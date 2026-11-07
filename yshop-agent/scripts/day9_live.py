"""Fresh owned TCP service, bounded SSE reader, and independent live cases."""

import asyncio
import codecs
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from time import perf_counter

import httpx

from app.agent.model import _loads
from app.config import Settings
from app.rag.config import RagSettings, SERVICE_ROOT
from app.rag.documents import load_policy_chunks
from app.schemas import ProductDetail, UserIdentity
from app.tools.yshop import YShopClient
from scripts.action_smoke_test import _http_error_code, _public_session_response
from scripts.chat_smoke_test import _contains_secret, _crop
from scripts.day9_cases import product_checks, policy_checks, tool_checks
from scripts.day9_observer import counts, read_events
from scripts.eval_policy import _project_path


class SSEReader:
    """Committed business-text timing, not model-token TTFT."""

    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer, self.total, self.events = "", 0, []
        self.status_first_ms = self.business_text_first_ms = self.terminal_ms = None

    def feed(self, value, elapsed_ms):
        self.total += len(value)
        if self.total > 2 * 1024 * 1024:
            raise ValueError("SSE total capacity")
        try:
            self.buffer += self.decoder.decode(value)
        except UnicodeError:
            raise ValueError("SSE invalid UTF8") from None
        if len(self.buffer) > 262144:
            raise ValueError("SSE frame capacity")
        # CRLF can itself be split between input buffers.
        while True:
            match = re.search(r"\r?\n\r?\n", self.buffer)
            if match is None:
                break
            frame, self.buffer = self.buffer[:match.start()], self.buffer[match.end():]
            self._frame(frame, elapsed_ms)

    def _frame(self, frame, elapsed_ms):
        if not frame.strip() or frame.startswith(":") and all(line.startswith(":") for line in frame.splitlines()):
            return
        kinds, data = [], []
        for line in frame.splitlines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                kinds.append(line[6:].lstrip(" "))
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            else:
                raise ValueError("SSE unexpected field")
        if len(kinds) != 1 or not data or kinds[0] not in {
                "status", "text", "products", "sources", "confirmation", "done", "error"}:
            raise ValueError("SSE invalid frame")
        if len(self.events) >= 32 or self.terminal_ms is not None:
            raise ValueError("SSE event after terminal")
        value = _loads("\n".join(data))
        if type(value) is not dict:
            raise ValueError("SSE payload not object")
        kind = kinds[0]
        if kind == "status":
            if value != {"status": "processing"} or self.status_first_ms is not None:
                raise ValueError("SSE invalid status")
            self.status_first_ms = elapsed_ms
        elif kind == "text":
            if set(value) != {"text"} or type(value["text"]) is not str:
                raise ValueError("SSE invalid text")
            self.business_text_first_ms = elapsed_ms
        elif kind in {"done", "error"}:
            self.terminal_ms = elapsed_ms
        self.events.append((kind, value))

    def finish(self):
        try:
            self.buffer += self.decoder.decode(b"", final=True)
        except UnicodeError:
            raise ValueError("SSE truncated UTF8") from None
        if self.buffer.strip() or self.terminal_ms is None:
            raise ValueError("SSE missing terminal")
        terminal = [(k, v) for k, v in self.events if k in {"done", "error"}]
        if len(terminal) != 1:
            raise ValueError("SSE terminal count")
        kind, value = terminal[0]
        if kind == "error":
            if set(value) != {"error"} or type(value["error"]) is not dict:
                raise ValueError("SSE invalid error")
            raise ValueError("SSE public error")
        if self.status_first_ms is None:
            raise ValueError("SSE missing status")
        fields = {"text": {"text": value.get("answer")}, "products": {"products": value.get("products")},
                  "sources": {"sources": value.get("sources"), "notice": value.get("notice")}}
        if value.get("pending_action") is not None:
            fields["confirmation"] = {"pending_action": value["pending_action"]}
        for field, expected in fields.items():
            matches = [v for k, v in self.events if k == field]
            if matches != [expected]:
                raise ValueError("SSE public mismatch")
        if value.get("pending_action") is None and any(k == "confirmation" for k, _ in self.events):
            raise ValueError("SSE unexpected confirmation")
        return value


def observed_usage(events):
    model = [e for e in events if e.get("phase") == "complete" and e.get("category") == "model"]
    if not model or any(e.get("usage") is None for e in model):
        return None
    return {key: sum(e["usage"][key] for e in model) for key in ("input_tokens", "output_tokens")}


def first_use_labels(seen, observations, tool_calls):
    complete = [e for e in observations if e.get("phase") == "complete" and e.get("forwarded") is True]
    activity = {
        "model": any(e.get("category") == "model" for e in complete),
        "product": any(e.get("category") == "shop_read"
                       and e.get("path") in {"agent/products", "product/detail"} for e in complete),
        "rag": (any(e.get("kind") == "local" and e.get("phase") == "start"
                    and e.get("category") == "rag" for e in observations)
                or any(t.get("name") == "search_policy" for t in tool_calls)),
    }
    labels = [label for label in ("model", "product", "rag") if activity[label] and label not in seen]
    seen.update(labels)
    return labels


def free_port(port):
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("invalid loopback port")
    with socket.socket() as sock:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise ValueError("dedicated_port_unavailable") from None


class OwnedAgent:
    def __init__(self, runtime, *, port, run_id, environ):
        self.runtime = _project_path(runtime)
        if self.runtime.parent != SERVICE_ROOT / ".tmp" or self.runtime.name != "day9-live-" + run_id:
            raise ValueError("invalid owned runtime")
        self.port, self.run_id, self.environment, self.process = port, run_id, dict(environ), None

    def start(self):
        if self.process is not None:
            raise ValueError("owned process already started")
        free_port(self.port)
        self.runtime.mkdir(parents=True, exist_ok=False)
        rag = RagSettings()
        if not rag.chroma_dir.is_dir():
            raise ValueError("missing original index")
        for path in rag.chroma_dir.rglob("*"):
            if path.is_symlink() or path.is_junction():
                raise ValueError("linked original index")
        shutil.copytree(rag.chroma_dir, self.runtime / "index")
        environment = {**self.environment, "TEMP": str(self.runtime), "TMP": str(self.runtime),
                       "PYTHONPYCACHEPREFIX": str(self.runtime / "pycache"), "HF_HUB_OFFLINE": "1",
                       "LLM_ENABLE_THINKING": "false", "LLM_TIMEOUT_SECONDS": "60",
                       "SESSION_DB_PATH": str(self.runtime / "sessions.sqlite3")}
        executable = sys.executable
        if sys.platform == "win32" and sys.executable != sys._base_executable:
            executable = sys._base_executable
            environment["__PYVENV_LAUNCHER__"] = sys.executable
        self.process = subprocess.Popen([executable, "-X", "utf8", "-W", "error", "-m", "scripts.day9_observer",
            "--port", str(self.port), "--runtime", str(self.runtime), "--run-id", self.run_id],
            cwd=SERVICE_ROOT, env=environment, shell=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def stop(self):
        if self.process is None:
            return {"pid": None, "exit_code": None, "awaited": False,
                    "termination_requested": False, "exited_before_shutdown": False,
                    "unexpected_exit": False}
        exited_before_shutdown = self.process.poll() is not None
        if not exited_before_shutdown:
            self.process.terminate()
            try:
                exit_code = self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                exit_code = self.process.wait(timeout=20)
        else:
            exit_code = self.process.wait(timeout=20)
        return {"pid": getattr(self.process, "pid", None), "exit_code": exit_code, "awaited": True,
                "termination_requested": not exited_before_shutdown,
                "exited_before_shutdown": exited_before_shutdown,
                "unexpected_exit": exited_before_shutdown}

    def events(self):
        return read_events(self.runtime / "observations.jsonl", self.run_id)

    async def ready(self, client):
        for _ in range(90):
            if self.process is None or self.process.poll() is not None:
                raise ValueError("owned_service_exited")
            try:
                response = await client.get("/health", timeout=2)
                startup = [e for e in self.events() if e["kind"] == "startup"]
                if (response.status_code == 200 and _loads(response.content) == {"status": "ok", "service": "yshop-agent"}
                        and len(startup) == 1 and startup[0]["pid"] == self.process.pid
                        and startup[0]["port"] == self.port and startup[0]["configured"] is True):
                    return
            except (httpx.HTTPError, OSError, ValueError):
                pass
            await asyncio.sleep(1)
        raise ValueError("owned_service_not_ready")


async def run_case(case, execute, save=None):
    row = {k: case[k] for k in ("id", "group", "mode")}
    row.update(status="not_run", steps=[{"status": "not_run"} for _ in case["steps"]])
    session_id, previous = None, []
    for index, step in enumerate(case["steps"]):
        try:
            result, session_id, current = await execute(step, session_id, previous)
        except (asyncio.CancelledError, KeyboardInterrupt):
            row.update(status="failed", interrupted=True)
            if save is not None:
                save(row)
            raise
        except Exception:
            result, current = {"status": "failed", "failures": ["live_step_failed"]}, []
        row["steps"][index] = result
        row["status"] = ("failed" if result["status"] != "passed" else
                         "passed" if index == len(case["steps"]) - 1 else "not_run")
        if save is not None:
            save(row)
        if result["status"] != "passed":
            break
        previous = current
    return row


class LiveRunner:
    def __init__(self, agent, *, authorization, secrets):
        self.agent, self.authorization = agent, authorization
        self.secrets = tuple(secrets)
        rag = RagSettings()
        self.originals = {c.chunk_id: c for c in load_policy_chunks(rag.knowledge_dir,
            chunk_size=rag.chunk_size, chunk_overlap=rag.chunk_overlap)}
        self.verifications = []
        self.first_uses = set()

    def _record_observations(self, row, tool_calls=()):
        identity = row.get("request_id")
        observations = [e for e in self.agent.events() if e.get("request_id") == identity] if identity else []
        row["observations"] = counts(observations)
        row["usage"] = observed_usage(observations)
        model_durations = [e["elapsed_ms"] for e in observations
            if e.get("phase") == "complete" and e.get("category") == "model"]
        row["model_elapsed_ms"] = sum(model_durations) if model_durations else None
        labels = first_use_labels(self.first_uses, observations, tool_calls)
        if labels or "first_use_labels" not in row:
            row["first_use_labels"] = labels
        return observations

    async def execute(self, step, sid, previous, client, shop):
        began = perf_counter()
        row = {"status": "failed", "sse": step["sse"], "phase": "local_http", "failures": []}
        try:
            return await self._execute(step, sid, previous, client, shop, began, row)
        except Exception:
            row["status"] = "failed"
            row["failures"].append("live_step_failed")
            row.setdefault("client_elapsed_ms", round((perf_counter() - began) * 1000))
            identity = row.get("request_id")
            if identity:
                try:
                    self._record_observations(row)
                except Exception:
                    row["failures"].append("step_observation_failed")
            return row, sid, previous

    async def _execute(self, step, sid, previous, client, shop, began, row):
        payload = {"message": step["message"]}
        if sid:
            payload["session_id"] = sid
        if _contains_secret(step["message"], self.secrets):
            raise ValueError("private frozen message")
        headers = {"Authorization": self.authorization}
        endpoint = "/agent/chat/stream" if step["sse"] else "/agent/chat"
        if step["sse"]:
            reader = SSEReader()
            try:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    row.update(http_status=response.status_code, request_id=response.headers.get("X-Request-ID"))
                    if response.status_code != 200 or response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
                        raise ValueError("stream HTTP failure")
                    row["phase"] = "sse_read"
                    async for data in response.aiter_bytes():
                        reader.feed(data, round((perf_counter() - began) * 1000))
                raw = reader.finish()
            finally:
                row.update(status_first_ms=reader.status_first_ms, business_text_first_ms=reader.business_text_first_ms,
                           terminal_ms=reader.terminal_ms, timing_scope="committed_business_text_not_model_token_ttft")
        else:
            response = await client.post(endpoint, json=payload, headers=headers)
            row.update(http_status=response.status_code, request_id=response.headers.get("X-Request-ID"))
            if response.status_code != 200:
                row.update(client_elapsed_ms=round((perf_counter() - began) * 1000),
                           failures=[_http_error_code(response)])
                self._record_observations(row)
                return row, sid, previous
            raw = _loads(response.content)
        row["client_elapsed_ms"] = round((perf_counter() - began) * 1000)
        row["phase"] = "public_projection"
        result = _public_session_response(raw, self.secrets, session_id=sid).model_dump(mode="json")
        if not re.fullmatch(r"[a-f0-9]{32}", row.get("request_id") or ""):
            raise ValueError("missing request correlation")
        row["phase"] = "observations"
        observations = self._record_observations(row, result["tool_calls"])
        observed = row["observations"]
        row["tool_elapsed_ms"] = sum(t["elapsed_ms"] for t in result["tool_calls"]) if result["tool_calls"] else None
        row["public"] = result
        expect = step["expect"]
        failures = row["failures"]
        if result["status"] != expect["status"]:
            failures.append("public_status_mismatch")
        traces = [t for t in result["tool_calls"] if t["status"] == "succeeded"]
        if expect["tool"] != "none" and not any(t["name"] == expect["tool"] for t in traces):
            failures.append("expected_tool_missing")
        if expect["tool"] == "none" and result["tool_calls"]:
            failures.append("unexpected_tool_trace")
        if any(t["status"] != "succeeded" for t in result["tool_calls"]):
            failures.append("tool_execution_failure")
        if any(t["name"] != expect["tool"] for t in result["tool_calls"]):
            failures.append("unexpected_tool_trace")
        argument_failures = tool_checks(expect, observations, previous)
        row["tool_parameters_match"] = not argument_failures if expect["tool"] in {"search_products", "get_product_detail"} else None
        failures.extend(argument_failures)
        complete = [e for e in observations if e.get("phase") == "complete"]
        if not complete or not any(e["category"] == "shop_read" and e.get("path") == "userinfo" for e in complete):
            failures.append("missing_identity_observation")
        if observed["blocked_shop_writes"] or observed["blocked_unexpected_requests"]:
            failures.append("blocked_outbound_attempt")
        if result["model_rounds"] and (not observed["external_model_verified"]
                or observed["model_http_requests"] != result["model_rounds"]):
            failures.append("model_execution_unverified")
        truth = {}
        row["phase"] = "independent_verification"
        verification_began = perf_counter()
        for product in result["products"]:
            detail = await shop.get_product_detail(product["product_id"], self.authorization)
            public = detail.model_dump(mode="json")
            if _contains_secret(public, self.secrets):
                raise ValueError("private verification response")
            truth[product["product_id"]] = public
            self.verifications.append({"scope": "independent_runner_get_not_agent_tool", "request_id": row["request_id"],
                                       "product_id": product["product_id"], "method": "GET", "verified_response": True})
        if expect["tool"] == "search_products":
            from app.schemas import ProductSearchInput
            query = dict(expect["query"])
            for key in ("page", "limit", "category_id"):
                if key in query:
                    query[key] = int(query[key])
            matched = await shop.search_products(ProductSearchInput(**query), self.authorization)
            if [p.model_dump(mode="json") for p in matched.products] != result["products"]:
                failures.append("independent_search_membership_mismatch")
            self.verifications.append({"scope": "independent_runner_get_not_agent_tool",
                "request_id": row["request_id"], "query": expect["query"], "method": "GET", "verified_response": True})
        row["independent_verification_elapsed_ms"] = round((perf_counter() - verification_began) * 1000)
        row["phase"] = "assertions"
        failures.extend(product_checks(result["products"], truth, expect))
        if "reference_index" in expect:
            index = expect["reference_index"]
            if len(previous) <= index or [p["product_id"] for p in result["products"]] != [previous[index]["product_id"]]:
                failures.append("reference_product_mismatch")
        citation = policy_checks(result, self.originals, expect)
        row["citations"] = citation
        failures.extend(citation["failures"])
        row["status"] = "failed" if failures else "passed"
        row["phase"] = "completed"
        return row, result["session_id"], previous if expect.get("preserve_product_ids") else result["products"]

    async def run(self, cases, update):
        settings = Settings()
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.agent.port}", timeout=600,
                                     trust_env=False, follow_redirects=False) as client:
            await self.agent.ready(client)
            identity_response = await client.get("/agent/me", headers={"Authorization": self.authorization})
            if identity_response.status_code != 200:
                raise ValueError("identity_verification_failed")
            identity = UserIdentity.model_validate(_crop(_loads(identity_response.content), UserIdentity))
            self.secrets = (*self.secrets, identity.nickname)
            async with httpx.AsyncClient(base_url=settings.yshop_api_url, timeout=settings.http_timeout_seconds,
                                         trust_env=False, follow_redirects=False) as shop_http:
                shop = YShopClient(shop_http)
                async def execute(step, sid, previous):
                    return await self.execute(step, sid, previous, client, shop)
                for case in cases:
                    row = await run_case(case, execute, lambda partial: update(partial))
                    update(row)
