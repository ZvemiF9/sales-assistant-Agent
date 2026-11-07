"""Owned child service: fail-closed metadata observations, unchanged responses."""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import socket
from time import perf_counter
from urllib.parse import urlsplit

import httpx

from app.agent.model import _loads, _parse_reply
from app.telemetry import request_id
from scripts.chat_smoke_test import _contains_secret
from scripts.day9_cases import QUERY_KEYS


IDENTIFIER = re.compile(r"[a-f0-9]{32}\Z")
CATEGORIES = {"model", "shop_read", "shop_write_blocked", "unexpected_blocked"}
SAFE_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}


def _origin(url):
    parts = urlsplit(str(url))
    return parts.scheme, parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)


class OutboundObserver:
    def __init__(self, *, shop_url, model_url, credential, record, run_id):
        if not IDENTIFIER.fullmatch(run_id):
            raise ValueError("invalid run identifier")
        self.shop_url, self.model_url = shop_url, model_url
        self.credential, self.record, self.run_id = credential, record, run_id
        self.private_secrets = [credential] if credential else []

    def _record(self, event, request):
        try:
            self.record(validate_event(event))
        except Exception:
            raise httpx.RequestError("day9_observation_failed", request=request) from None

    def _record_local(self, event):
        try:
            self.record(validate_event(event))
        except Exception:
            raise RuntimeError("day9_observation_failed") from None

    def answer_policy(self, original, service, question):
        identity = request_id()
        event = {"kind": "local", "phase": "start", "run_id": self.run_id,
                 "request_id": identity, "category": "rag", "operation": "policy_answer"}
        self._record_local(event)
        started = perf_counter()
        try:
            result = original(service, question)
        except BaseException:
            self._record_local({**event, "phase": "complete", "status": "failed",
                                "elapsed_ms": max(0, round((perf_counter() - started) * 1000))})
            raise
        self._record_local({**event, "phase": "complete", "status": "succeeded",
                            "elapsed_ms": max(0, round((perf_counter() - started) * 1000))})
        return result

    def _classify(self, request):
        method = request.method if request.method in SAFE_METHODS else "OTHER"
        value = str(request.url)
        parts = urlsplit(value)
        base = {"category": "unexpected_blocked", "method": method}
        if _origin(value) == _origin(self.shop_url) and method not in {"GET", "HEAD"}:
            return {**base, "category": "shop_write_blocked"}
        if value == self.model_url and method == "POST":
            return {**base, "category": "model"}
        if _origin(value) != _origin(self.shop_url) or parts.username or parts.password or parts.fragment:
            return base
        prefix = urlsplit(self.shop_url).path
        if not parts.path.startswith(prefix):
            return base
        path = parts.path[len(prefix):]
        if method not in {"GET", "HEAD"}:
            return {**base, "category": "shop_write_blocked"}
        query = dict(request.url.params)
        if len(query) != len(list(request.url.params.multi_items())):
            return base
        if path == "agent/products":
            if (query.keys() - QUERY_KEYS or any(len(v) > 100 or any(ord(c) < 32 for c in v) for v in query.values())
                    or _contains_secret(query, tuple(self.private_secrets))):
                return base
            return {**base, "category": "shop_read", "path": path, "query": query}
        if path == "userinfo" and not query:
            return {**base, "category": "shop_read", "path": path}
        if re.fullmatch(r"product/detail/[1-9][0-9]{0,18}", path) and not query:
            return {**base, "category": "shop_read", "path": "product/detail",
                    "product_id": int(path.rsplit("/", 1)[1])}
        return base

    async def send(self, original, client, request, **kwargs):
        authorization = request.headers.get("Authorization", "")
        for secret in (authorization, authorization.partition(" ")[2]):
            if secret and secret not in self.private_secrets:
                self.private_secrets.append(secret)
        identity = request_id()
        event = {"kind": "outbound", "phase": "start", "run_id": self.run_id,
                 "request_id": identity, "forwarded": False, **self._classify(request)}
        self._record(event, request)
        if event["category"] in {"shop_write_blocked", "unexpected_blocked"}:
            raise httpx.RequestError("day9_outbound_blocked", request=request)
        started = perf_counter()
        completed = {**event, "phase": "complete", "forwarded": True, "http_status": None,
                     "elapsed_ms": 0, "protocol": None, "usage": None, "transport_error": None}
        try:
            response = await original(client, request, **kwargs)
        except Exception as error:
            completed.update(elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
                             transport_error="timeout" if isinstance(error, httpx.TimeoutException) else "request_failed")
            self._record(completed, request)
            raise
        completed.update(http_status=response.status_code, elapsed_ms=max(0, round((perf_counter() - started) * 1000)))
        if event["category"] == "model":
            completed["protocol"] = "invalid"
            try:
                # The app still receives and independently parses this exact response.
                await response.aread()
                parsed = _parse_reply(_loads(response.content), self.credential)
                completed.update(protocol="valid", usage=parsed.usage_metadata)
            except (ValueError, TypeError, KeyError, RecursionError, OverflowError, UnicodeError):
                pass
        elif event.get("path") == "userinfo":
            try:
                await response.aread()
                payload = _loads(response.content)
                nickname = payload["data"].get("nickname")
                if type(nickname) is str and nickname:
                    self.private_secrets.append(nickname)
            except (ValueError, TypeError, KeyError, UnicodeError):
                pass
        self._record(completed, request)
        return response

    @contextmanager
    def installed(self):
        from app.rag.service import PolicyService
        original = httpx.AsyncClient.send
        original_policy_answer = PolicyService.answer
        observer = self
        async def observed(client, request, **kwargs):
            return await observer.send(original, client, request, **kwargs)
        def observed_policy_answer(service, question):
            return observer.answer_policy(original_policy_answer, service, question)
        httpx.AsyncClient.send = observed
        PolicyService.answer = observed_policy_answer
        try:
            yield
        finally:
            PolicyService.answer = original_policy_answer
            httpx.AsyncClient.send = original


def validate_event(event):
    if type(event) is not dict or type(event.get("run_id")) is not str or not IDENTIFIER.fullmatch(event["run_id"]):
        raise ValueError("invalid observation")
    if event.get("kind") == "startup":
        if (set(event) != {"kind", "run_id", "pid", "port", "configured", "timeout_seconds", "enable_thinking"}
                or type(event["pid"]) is not int or event["pid"] <= 0
                or type(event["port"]) is not int or not 1024 <= event["port"] <= 65535
                or type(event["configured"]) is not bool or type(event["timeout_seconds"]) not in {int, float}
                or not 0 < event["timeout_seconds"] <= 600
                or event["enable_thinking"] is not None and type(event["enable_thinking"]) is not bool):
            raise ValueError("invalid startup observation")
        return event
    if event.get("kind") == "local":
        keys = {"kind", "phase", "run_id", "request_id", "category", "operation"}
        if (event.get("phase") not in {"start", "complete"} or event.get("category") != "rag"
                or event.get("operation") != "policy_answer"
                or type(event.get("request_id")) is not str or not IDENTIFIER.fullmatch(event["request_id"])):
            raise ValueError("invalid local observation")
        if event["phase"] == "complete":
            keys |= {"status", "elapsed_ms"}
            if (event.get("status") not in {"succeeded", "failed"}
                    or type(event.get("elapsed_ms")) is not int
                    or not 0 <= event["elapsed_ms"] <= 600000):
                raise ValueError("invalid local completion observation")
        if set(event) != keys:
            raise ValueError("unexpected local observation field")
        return event
    keys = {"kind", "phase", "run_id", "request_id", "forwarded", "category", "method"}
    if (event.get("kind") != "outbound" or event.get("phase") not in {"start", "complete"}
            or event.get("category") not in CATEGORIES or event.get("method") not in SAFE_METHODS | {"OTHER"}
            or type(event.get("forwarded")) is not bool
            or event.get("request_id") is not None and (
                type(event["request_id"]) is not str or not IDENTIFIER.fullmatch(event["request_id"]))):
        raise ValueError("invalid HTTP observation")
    if event["category"] == "shop_read":
        keys.add("path")
        if event.get("path") not in {"userinfo", "agent/products", "product/detail"}:
            raise ValueError("invalid observed shop path")
        if event["path"] == "agent/products":
            keys.add("query")
            query = event.get("query")
            if (type(query) is not dict or query.keys() - QUERY_KEYS
                    or any(type(k) is not str or type(v) is not str or len(v) > 100
                           or any(ord(c) < 32 for c in v) for k, v in query.items())):
                raise ValueError("invalid observed query")
        if event["path"] == "product/detail":
            keys.add("product_id")
            if type(event.get("product_id")) is not int or not 0 < event["product_id"] < 10**19:
                raise ValueError("invalid observed product")
    if event["phase"] == "complete":
        keys |= {"http_status", "elapsed_ms", "protocol", "usage", "transport_error"}
        if (event.get("http_status") is not None and (
                type(event["http_status"]) is not int or not 100 <= event["http_status"] <= 599)
                or type(event.get("elapsed_ms")) is not int or not 0 <= event["elapsed_ms"] <= 600000
                or event.get("protocol") not in {None, "valid", "invalid"}
                or event.get("transport_error") not in {None, "timeout", "request_failed"}):
            raise ValueError("invalid completion observation")
        usage = event.get("usage")
        if usage is not None and (type(usage) is not dict or set(usage) != {"input_tokens", "output_tokens", "total_tokens"}
                or any(type(v) is not int or v < 0 for v in usage.values())
                or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]):
            raise ValueError("invalid observed usage")
    elif event["forwarded"]:
        raise ValueError("invalid start observation")
    if set(event) != keys:
        raise ValueError("unexpected observation field")
    return event


def read_events(path, run_id):
    if not Path(path).is_file() or Path(path).stat().st_size > 16 * 1024 * 1024:
        raise ValueError("missing or oversized observations")
    result = []
    for line in Path(path).read_bytes().splitlines():
        if len(line) > 4096:
            raise ValueError("oversized observation")
        event = validate_event(_loads(line))
        if event["run_id"] != run_id:
            raise ValueError("foreign observation")
        result.append(event)
    return result


def counts(events):
    complete = [e for e in events if e.get("kind") == "outbound" and e.get("phase") == "complete" and e.get("forwarded") is True]
    model = [e for e in complete if e["category"] == "model"]
    starts = [e for e in events if e.get("kind") == "outbound" and e.get("phase") == "start" and e.get("category") == "model"]
    complete_pairs = Counter(e.get("request_id") for e in model) == Counter(e.get("request_id") for e in starts)
    correlated = all(e.get("request_id") is not None for e in model)
    valid = sum(e.get("protocol") == "valid" for e in model)
    successes = sum(type(e.get("http_status")) is int and 200 <= e["http_status"] < 300 for e in model)
    blocked = [e for e in events if e.get("phase") == "start" and e.get("category") in {"shop_write_blocked", "unexpected_blocked"}]
    local_rag = [e for e in events if e.get("kind") == "local" and e.get("category") == "rag"]
    return {"model_http_requests": len(model), "model_http_successes": successes,
            "model_protocol_valid": valid, "model_protocol_invalid": sum(e.get("protocol") == "invalid" for e in model),
            "shop_read_requests": sum(e["category"] == "shop_read" for e in complete),
            "business_write_requests": 0 if complete else None,
            "blocked_shop_writes": sum(e["category"] == "shop_write_blocked" for e in blocked),
            "blocked_unexpected_requests": sum(e["category"] == "unexpected_blocked" for e in blocked),
            "external_model_verified": bool(model) and complete_pairs and correlated and len(model) == valid == successes,
            "model_requests_correlated": correlated if model else None,
            "model_started_requests": len(starts), "model_observation_pairs_complete": complete_pairs if starts else None,
            "local_rag_requests": sum(e.get("phase") == "start" for e in local_rag),
            "local_rag_completed": sum(e.get("phase") == "complete" for e in local_rag),
            "local_rag_failures": sum(e.get("phase") == "complete" and e.get("status") == "failed"
                                      for e in local_rag)}


def main(argv=None):
    from app.agent.config import AgentSettings
    from app.config import Settings
    from app.main import create_app
    from app.rag.config import RagSettings
    from app.storage.config import SessionSettings
    from scripts.eval_policy import _project_path
    import uvicorn

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    runtime = _project_path(args.runtime)
    if (runtime.parent.name != ".tmp" or not re.fullmatch(r"day9-live-[a-f0-9]{32}", runtime.name)
            or not IDENTIFIER.fullmatch(args.run_id) or not 1024 <= args.port <= 65535
            or (runtime / "sessions.sqlite3").exists()):
        return 1
    settings, model = Settings(), AgentSettings()
    rag = RagSettings(chroma_dir=runtime / "index")
    sessions = SessionSettings(db_path=runtime / "sessions.sqlite3")
    application = create_app(settings, rag_settings=rag, agent_settings=model, session_settings=sessions)
    try:
        with (runtime / "observations.jsonl").open("x", encoding="utf-8") as output:
            def record(event):
                output.write(json.dumps(event, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            observer = OutboundObserver(shop_url=settings.yshop_api_url,
                model_url=model.llm_base_url + "chat/completions", credential=model.llm_api_key.get_secret_value(),
                record=record, run_id=args.run_id)
            with socket.socket() as listener:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind(("127.0.0.1", args.port))
                listener.listen(128)
                record(validate_event({"kind": "startup", "run_id": args.run_id, "pid": os.getpid(),
                    "port": args.port, "configured": model.configured, "timeout_seconds": model.llm_timeout_seconds,
                    "enable_thinking": model.llm_enable_thinking}))
                with observer.installed():
                    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=args.port,
                                                         log_config=None, access_log=False))
                    server.run(sockets=[listener])
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
