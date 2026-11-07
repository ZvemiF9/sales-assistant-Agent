"""Send completed chat turns to one local session and retain public evidence."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Mapping, Sequence, TextIO

from dotenv import dotenv_values
import httpx
from pydantic import TypeAdapter, ValidationError

from app.agent.schemas import ChatRequest, SessionId
from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from app.rag.config import SERVICE_ROOT
from scripts.chat_smoke_test import _public_response
from scripts.eval_policy import _arguments, _report_path
from scripts.policy_smoke_test import API_URL, _local_url
from scripts.smoke_test import _SafeParser, _authorization, _redacted


REPORT_PATH = SERVICE_ROOT / "data" / "day5-session-smoke.json"
_SESSION_CODES = {"session_not_found", "session_busy", "session_conflict", "session_unavailable"}
_ERRORS = {
    "chat_http_error": "本地会话接口请求失败，请检查服务状态。",
    "chat_invalid_response": "会话接口返回的数据无效。",
    "report_write_failed": "无法保存会话联调报告，请检查项目数据目录权限。",
    "session_smoke_error": "连续对话联调失败，请检查本地服务配置。",
}


def _failure(report, code):
    report.update(status="failed", error={"code": code,
        "message": _ERRORS[code] if code in _ERRORS else ShopError(code).message})


def _save_owned_report(path: Path, report: dict, authorization: str) -> None:
    """Atomically save through a temporary file exclusively owned by this call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(_redacted(report, authorization), ensure_ascii=False,
                                    indent=2) + "\n")
        temporary.replace(path)
        replaced = True
    finally:
        if not replaced and temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_file: Path = SERVICE_ENV_FILE, transport: httpx.AsyncBaseTransport | None = None,
         report_path: Path = REPORT_PATH, stdout: TextIO | None = None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    environment = os.environ if environ is None else environ
    path, owns_report = None, False
    authorization = ""
    report = {"schema_version": 1, "status": "pending", "turns": [],
              "business_write_requests": 0,
              "write_measurement_scope": "CLI sends only local chat POSTs; downstream writes require separate evidence"}
    began = perf_counter()
    try:
        parser = _SafeParser(description="使用现有商城身份连续 POST 本地会话接口。", allow_abbrev=False)
        parser.add_argument("--messages", required=True, help="包含 1–12 条消息的 JSON 数组")
        parser.add_argument("--session-id")
        parser.add_argument("--api-url", default=API_URL)
        parser.add_argument("--report", default=str(report_path))
        parser.add_argument("--mode", choices=("real_model", "test_double", "unverified"),
                            default="test_double" if transport is not None else "unverified")
        options = _arguments(argv, parser=parser,
            flags={"--messages", "--session-id", "--api-url", "--report", "--mode"})
        messages = json.loads(options.messages)
        if not isinstance(messages, list) or not 1 <= len(messages) <= 12:
            raise ShopError("invalid_request")
        queries = [ChatRequest(message=message) for message in messages]
        session_id = TypeAdapter(SessionId).validate_python(options.session_id) if options.session_id is not None else None
        if any(ord(character) < 32 or ord(character) == 127 for character in options.api_url):
            raise ShopError("invalid_request")
        base_url = _local_url(options.api_url)
        if transport is not None and options.mode == "real_model":
            raise ShopError("invalid_request")
        path = _report_path(options.report, environ=environment, env_file=env_file)
        if path.exists() or path.with_name(path.name + ".tmp").exists() or path.name.lower().startswith(("day1-", "day2-", "day3-", "day4-")):
            raise ShopError("invalid_request")
        authorization = _authorization(environment, Path(env_file))
        model_key = (environment["LLM_API_KEY"] if "LLM_API_KEY" in environment else
                     dotenv_values(env_file, interpolate=False).get("LLM_API_KEY", ""))
        secrets = (authorization, authorization.partition(" ")[2], model_key if isinstance(model_key, str) else "")
        if any(secret and secret in query.message for query in queries for secret in secrets):
            raise ShopError("invalid_request")
        report.update(mode=options.mode, provenance={"declared_mode": options.mode,
            "declaration_source": "runner", "external_model_verified": False, "injected_transport": transport is not None})
        if session_id is not None:
            report["session_id"] = session_id
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive reservation rejects concurrent runs sharing an evidence filename.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        owns_report = True

        async def chat_turns():
            nonlocal session_id
            async with httpx.AsyncClient(base_url=base_url, timeout=600.0, trust_env=False,
                                         follow_redirects=False, transport=transport) as client:
                for query in queries:
                    payload = {"message": query.message}
                    if session_id is not None:
                        payload["session_id"] = session_id
                    started = perf_counter()
                    response = await client.post("/agent/chat", json=payload,
                                                 headers={"Authorization": authorization})
                    if response.status_code != 200:
                        code = "login_required" if response.status_code == 401 else "chat_http_error"
                        try:
                            body = response.json()
                            candidate = body.get("error", {}).get("code") if isinstance(body, dict) else None
                            # Only known public codes survive, never upstream messages.
                            if isinstance(candidate, str) and (candidate in _SESSION_CODES or
                                    ShopError(candidate).code == candidate):
                                code = candidate
                        except (ValueError, TypeError, AttributeError):
                            pass
                        report["http_status"] = response.status_code
                        _failure(report, code)
                        return
                    try:
                        body = response.json()
                        returned_id = TypeAdapter(SessionId).validate_python(body.get("session_id"))
                        if session_id is not None and returned_id != session_id:
                            raise ValueError
                        result = _public_response(body, secrets)
                    except (ValueError, TypeError, AttributeError, ValidationError):
                        _failure(report, "chat_invalid_response")
                        return
                    session_id = returned_id
                    report["session_id"] = session_id
                    report["turns"].append({"message": query.message, "http_status": 200,
                        "elapsed_seconds": perf_counter() - started, "result": result.model_dump(mode="json")})
                    _save_owned_report(_report_path(path, environ=environment, env_file=env_file), report, authorization)
                report["status"] = "passed"

        asyncio.run(chat_turns())
    except SystemExit as error:
        return int(error.code or 0)
    except ShopError as error:
        _failure(report, error.code)
    except (ValidationError, ValueError, TypeError):
        _failure(report, "invalid_request")
    except OSError:
        _failure(report, "report_write_failed")
    except httpx.HTTPError:
        _failure(report, "chat_http_error")
    except Exception:
        _failure(report, "session_smoke_error")
    report["elapsed_seconds"] = perf_counter() - began
    if owns_report:
        try:
            _save_owned_report(_report_path(path, environ=environment, env_file=env_file), report, authorization)
        except (OSError, ShopError):
            _failure(report, "report_write_failed")
    summary = {"status": report["status"], "turn_count": len(report["turns"])}
    if "error" in report:
        summary["error_code"] = report["error"]["code"]
    print(json.dumps(summary), file=stdout)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
