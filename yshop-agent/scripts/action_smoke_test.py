"""One local prepare or explicit saved-ID decision, without automatic replay."""

import asyncio
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence, TextIO

from dotenv import dotenv_values
import httpx
from pydantic import TypeAdapter, ValidationError

from app.actions.schemas import ActionDecision, ActionId, ActionResponse, CartActionCard, CartActionResult
from app.agent.model import _loads
from app.agent.schemas import ChatRequest, SessionChatResponse, TokenUsage
from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from app.rag.config import SERVICE_ROOT
from scripts.chat_smoke_test import _contains_secret, _crop, _public_response
from scripts.eval_policy import _arguments, _report_path
from scripts.policy_smoke_test import API_URL, _local_url
from scripts.session_smoke_test import _save_owned_report
from scripts.smoke_test import _SafeParser, _authorization, _redacted


REPORT_PATH = SERVICE_ROOT / "data" / "day6-action-smoke.json"
_READ_STATUSES = {"answered", "needs_input", "no_evidence", "limit_reached"}
_ERRORS = {
    "action_http_error": "Local action request failed. Do not repeat an uncertain decision.",
    "action_invalid_response": "The local action response is invalid.",
    "action_reconfirmation_required": "The card changed; a new explicit decision is required.",
    "action_not_completed": "The requested action did not complete.",
    "report_write_failed": "Cannot save the action report.",
    "action_smoke_error": "Action verification failed. Do not repeat an uncertain decision.",
}


def _action_report_path(value, *, environ, env_file, initial=False):
    path = _report_path(value, environ=environ, env_file=env_file)
    if path.name.lower().startswith(("day1-", "day2-", "day3-", "day4-", "day5-")):
        raise ShopError("invalid_request")
    if initial:
        if path.is_dir():
            raise OSError("report destination is a directory")
        if path.exists() or path.with_name(path.name + ".tmp").exists():
            raise ShopError("invalid_request")
    return path


def _public_action_response(value, secrets, *, action_id=None):
    if not isinstance(value, dict) or not set(ActionResponse.model_fields) <= set(value):
        raise ValueError("incomplete action response")
    cropped = _crop(value, ActionResponse)
    for name, model in (("pending_action", CartActionCard), ("action_result", CartActionResult)):
        item = cropped[name]
        if item is not None:
            if not isinstance(item, dict) or not set(model.model_fields) <= set(item):
                raise ValueError("incomplete public action")
            cropped[name] = _crop(item, model)
    if _contains_secret(cropped, secrets):
        raise ValueError("credential in public action")
    result = ActionResponse.model_validate(cropped)
    # A replacement is display-only. Every result must bind to the requested ID.
    if action_id is not None and result.action_result is not None and result.action_result.action_id != action_id:
        raise ValueError("action ID mismatch")
    return result


def _public_session_response(value, secrets, *, session_id=None):
    if not isinstance(value, dict) or not set(SessionChatResponse.model_fields) <= set(value):
        raise ValueError("incomplete session response")
    returned_id = TypeAdapter(ActionId).validate_python(value["session_id"])
    if session_id is not None and returned_id != session_id:
        raise ValueError("session ID mismatch")
    if value["status"] in _READ_STATUSES and value["model_rounds"] != 0:
        if value["pending_action"] is not None or value["action_result"] is not None:
            raise ValueError("read response carries an action")
        direct = _public_response(value, secrets)
        return SessionChatResponse(**direct.model_dump(), session_id=returned_id)
    action = _public_action_response(value, secrets)
    if (type(value["model_rounds"]) is not int or value["model_rounds"] != 0
            or value["products"] != [] or value["sources"] != [] or value["tool_calls"] != []
            or value["notice"] is not None
            or _crop(value["usage"], TokenUsage) !=
            {"input_tokens": None, "output_tokens": None}):
        raise ValueError("action response fabricates model evidence")
    return SessionChatResponse(**action.model_dump(), model_rounds=0)


def _secrets(environment, env_file, authorization):
    key = (environment["LLM_API_KEY"] if "LLM_API_KEY" in environment else
           dotenv_values(env_file, interpolate=False).get("LLM_API_KEY", ""))
    return (authorization, authorization.partition(" ")[2], key if isinstance(key, str) else "")


def _redact_secrets(value, secrets):
    for secret in secrets:
        if secret:
            value = _redacted(value, secret)
    return value


def _http_error_code(response, fallback="action_http_error"):
    if response.status_code == 401:
        return "login_required"
    if response.status_code == 404:
        return "action_not_found"
    if 400 <= response.status_code <= 599:
        try:
            value = _loads(response.content)
            error = value.get("error") if isinstance(value, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            if isinstance(code, str) and ShopError(code).code == code:
                return code
        except (ValueError, TypeError, UnicodeError, RecursionError):
            pass
    return fallback


def _failure(report, code):
    report.update(status="failed", error={
        "code": code, "message": _ERRORS.get(code, ShopError(code).message),
    })


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_file: Path = SERVICE_ENV_FILE, transport: httpx.AsyncBaseTransport | None = None,
         report_path: Path = REPORT_PATH, stdout: TextIO | None = None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    environment = os.environ if environ is None else environ
    path, owns_report = None, False
    authorization, secrets = "", ()
    report = {
        "schema_version": 1, "status": "pending", "local_http_requests": 0,
        "business_write_requests": None,
        "write_measurement_scope": "Local CLI HTTP only; downstream writes require separate observations.",
    }
    began = perf_counter()
    try:
        parser = _SafeParser(description="Prepare a local card or explicitly decide one saved action ID.",
                             allow_abbrev=False)
        parser.add_argument("--message")
        parser.add_argument("--session-id")
        parser.add_argument("--action-id")
        parser.add_argument("--decision", choices=("confirm", "cancel"))
        parser.add_argument("--api-url", default=API_URL)
        parser.add_argument("--report", default=str(report_path))
        parser.add_argument("--mode", choices=("real_model", "test_double", "unverified"),
                            default="test_double" if transport is not None else "unverified")
        options = _arguments(argv, parser=parser, flags={
            "--message", "--session-id", "--action-id", "--decision", "--api-url", "--report", "--mode",
        })
        preparing = options.message is not None
        if preparing:
            if options.action_id is not None or options.decision is not None:
                raise ShopError("invalid_request")
            query = ChatRequest(message=options.message, session_id=options.session_id)
            action_id, decision = None, None
        else:
            if options.action_id is None or options.decision is None or options.session_id is not None:
                raise ShopError("invalid_request")
            action_id = TypeAdapter(ActionId).validate_python(options.action_id)
            decision = ActionDecision(decision=options.decision)
            query = None
        if any(ord(character) < 32 or ord(character) == 127 for character in options.api_url):
            raise ShopError("invalid_request")
        base_url = _local_url(options.api_url)
        if transport is not None and options.mode == "real_model":
            raise ShopError("invalid_request")
        path = _action_report_path(options.report, environ=environment, env_file=env_file, initial=True)
        authorization = _authorization(environment, Path(env_file))
        secrets = _secrets(environment, env_file, authorization)
        if query is not None and _contains_secret(query.message, secrets):
            raise ShopError("invalid_request")
        report.update(operation="prepare" if preparing else "decision", mode=options.mode,
            provenance={"declared_mode": options.mode, "declaration_source": "runner",
                        "external_model_verified": False, "injected_transport": transport is not None})
        if decision is not None:
            report.update(action_id=action_id, decision={
                "value": decision.decision, "request_status": "not_attempted", "retry_permitted": False,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            owns_report = True
            stream.write(json.dumps(_redact_secrets(report, secrets), ensure_ascii=False, indent=2) + "\n")

        async def request_once():
            if decision is not None:
                report["decision"]["request_status"] = "sending"
                # Persist uncertainty before the sole POST, including report-path revalidation.
                _save_owned_report(_action_report_path(path, environ=environment, env_file=env_file),
                                   _redact_secrets(report, secrets), authorization)
            async with httpx.AsyncClient(base_url=base_url, timeout=600.0, trust_env=False,
                                         follow_redirects=False, transport=transport) as client:
                report["local_http_requests"] = 1
                return await client.post(
                    "/agent/chat" if preparing else f"/agent/actions/{action_id}/decision",
                    json=query.model_dump(mode="json", exclude_none=True) if preparing else
                    decision.model_dump(mode="json"),
                    headers={"Authorization": authorization},
                )

        response = asyncio.run(request_once())
        if decision is not None:
            report["decision"]["request_status"] = "completed"
        report["http_status"] = response.status_code
        if response.status_code != 200:
            _failure(report, _http_error_code(response))
        else:
            try:
                value = _loads(response.content)
                result = (_public_session_response(value, secrets, session_id=query.session_id) if preparing else
                          _public_action_response(value, secrets, action_id=action_id))
            except (ValueError, TypeError, ValidationError, UnicodeError, RecursionError):
                _failure(report, "action_invalid_response")
            else:
                report["result"] = result.model_dump(mode="json")
                if not preparing and result.pending_action is not None:
                    _failure(report, "action_reconfirmation_required")
                elif not preparing and result.status != ("succeeded" if decision.decision == "confirm" else "cancelled"):
                    _failure(report, result.action_result.error_code if result.action_result is not None
                             and result.action_result.error_code is not None else "action_not_completed")
                elif preparing and result.status in {"executing", "succeeded", "failed", "unknown", "cancelled"}:
                    _failure(report, "action_not_completed")
                else:
                    report["status"] = "passed"
    except SystemExit as error:
        return int(error.code or 0)
    except ShopError as error:
        _failure(report, error.code)
    except (ValidationError, ValueError, TypeError):
        _failure(report, "invalid_request")
    except OSError:
        _failure(report, "report_write_failed")
    except httpx.HTTPError:
        if "decision" in report and report["local_http_requests"]:
            report["decision"]["request_status"] = "unknown"
        _failure(report, "action_http_error")
    except Exception:
        if "decision" in report and report["local_http_requests"]:
            report["decision"]["request_status"] = "unknown"
        _failure(report, "action_smoke_error")
    report["elapsed_seconds"] = perf_counter() - began
    if owns_report:
        try:
            _save_owned_report(_action_report_path(path, environ=environment, env_file=env_file),
                               _redact_secrets(report, secrets), authorization)
        except (OSError, ShopError):
            _failure(report, "report_write_failed")
    summary = {"status": report["status"]}
    if "result" in report:
        result = report["result"]
        summary.update(action_status=result["status"], session_id=result["session_id"])
        action = result.get("pending_action") or result.get("action_result")
        if action is not None:
            summary["action_id"] = action["action_id"]
    if "error" in report:
        summary["error_code"] = report["error"]["code"]
    print(json.dumps(_redact_secrets(summary, secrets), ensure_ascii=False), file=stdout)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
