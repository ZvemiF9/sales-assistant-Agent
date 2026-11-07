"""One local read-only chat POST; runner mode is not model attestation."""

import asyncio
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence, TextIO

from dotenv import dotenv_values
import httpx
from pydantic import ValidationError

from app.agent.schemas import ChatRequest, ChatResponse, TokenUsage, ToolTrace
from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from app.rag.config import SERVICE_ROOT
from app.rag.schemas import PolicySource
from app.rag.service import DEMO_NOTICE
from app.schemas import ProductDetail, ProductSKU
from scripts import eval_policy
from scripts.eval_policy import NO_EVIDENCE_ANSWER, _arguments, _report_path
from scripts.policy_smoke_test import API_URL, _local_url
from scripts.smoke_test import _SafeParser, _authorization, _save_report


REPORT_PATH = SERVICE_ROOT / "data" / "day4-chat-smoke.json"
_READ_TOOLS = {"search_products", "get_product_detail", "search_policy"}
_KNOWN_HTTP_CODES = {
    "login_required", "invalid_request", "shop_unavailable", "shop_invalid_response", "shop_rejected",
    "model_not_configured", "model_timeout", "model_unavailable", "model_rejected", "model_invalid_response",
    "rag_not_built", "rag_index_incompatible", "rag_index_corrupt", "rag_model_unavailable", "rag_unavailable",
}
_ERROR_MESSAGES = {
    "chat_http_error": "本地查询接口请求失败，请检查服务状态。",
    "chat_invalid_response": "查询接口返回的数据无效。",
    "report_write_failed": "无法保存查询联调报告，请检查项目数据目录权限。",
    "chat_smoke_error": "查询联调失败，请检查本地服务配置。",
}


def _chat_report_path(value, *, environ, env_file, initial=False):
    path = _report_path(value, environ=environ, env_file=env_file)
    # Preserve all prior-day reports, including dated final runs not enumerated
    # by the day-three helper. Only initial resolution checks existing day-four
    # reports: the final write must update this invocation's new pending file.
    name = path.name.lower()
    if name.startswith(("day1-", "day2-", "day3-")) or (initial and path.exists()
            and name.startswith("day4-") and path.relative_to(eval_policy.SERVICE_ROOT.resolve()).parts[0] == "data"):
        raise ShopError("invalid_request")
    return path


def _crop(value, model):
    if not isinstance(value, dict):
        raise ValueError
    return {key: value[key] for key in model.model_fields if key in value}


def _contains_secret(value, secrets):
    if isinstance(value, str):
        return any(secret and secret in value for secret in secrets)
    if isinstance(value, list):
        return any(_contains_secret(item, secrets) for item in value)
    if isinstance(value, dict):
        return any(_contains_secret(item, secrets) for item in value.values())
    return False


def _public_response(value, secrets):
    if not isinstance(value, dict) or not set(ChatResponse.model_fields) <= set(value):
        raise ValueError
    cropped = _crop(value, ChatResponse)
    for key, model in (("products", ProductDetail), ("sources", PolicySource), ("tool_calls", ToolTrace)):
        if not isinstance(value[key], list):
            raise ValueError
        cropped[key] = [_crop(item, model) for item in value[key]]
    for product in cropped["products"]:
        if not isinstance(product.get("skus"), list):
            raise ValueError
        product["skus"] = [_crop(sku, ProductSKU) for sku in product["skus"]]
    cropped["usage"] = _crop(value["usage"], TokenUsage)
    # Crop first: ignored upstream extras are never accepted as public data.
    # Public echoes fail the run rather than turning redaction into acceptance.
    if _contains_secret(cropped, secrets):
        raise ValueError
    for source in cropped["sources"]:
        if (any(not isinstance(source.get(key), str) or not source[key].strip()
                for key in PolicySource.model_fields if key != "score")
                or type(source.get("score")) not in {int, float}):
            raise ValueError
    result = ChatResponse.model_validate(cropped)
    if (not result.answer.strip() or len(result.tool_calls) > 3 * (result.model_rounds - 1)
            or (result.status == "limit_reached" and result.model_rounds != 5)
            or any(trace.status == "succeeded" and trace.name not in _READ_TOOLS for trace in result.tool_calls)):
        raise ValueError
    policy_succeeded = any(trace.name == "search_policy" and trace.status == "succeeded"
                           for trace in result.tool_calls)
    if result.notice is not None and (result.notice != DEMO_NOTICE or not policy_succeeded):
        raise ValueError
    if result.sources and (result.notice != DEMO_NOTICE or not policy_succeeded
                           or result.status == "no_evidence"
                           or any(source.text not in result.answer for source in result.sources)):
        raise ValueError
    refusals = (NO_EVIDENCE_ANSWER, "演示资料中未找到足够依据，请咨询客服。")
    if result.status == "no_evidence" and (result.sources or result.notice != DEMO_NOTICE
            or not policy_succeeded or not any(refusal in result.answer for refusal in refusals)):
        raise ValueError
    return result


def _failure(report, code):
    message = ShopError(code).message if code in _KNOWN_HTTP_CODES else _ERROR_MESSAGES[code]
    report.pop("result", None)
    report.update(status="failed", error={"code": code, "message": message})


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_file: Path = SERVICE_ENV_FILE, transport: httpx.AsyncBaseTransport | None = None,
         report_path: Path = REPORT_PATH, stdout: TextIO | None = None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    environment = os.environ if environ is None else environ
    path = None
    authorization = ""
    report = {"schema_version": 1, "status": "pending", "method": "POST", "endpoint": "/agent/chat",
              "scope": "one local authenticated read-only chat request",
              "business_write_requests": 0,
              "write_measurement_scope": "CLI sends only the local chat POST; downstream writes require separate acceptance evidence"}
    began = perf_counter()
    try:
        parser = _SafeParser(description="使用现有商城身份 POST 本地单次只读查询接口。", allow_abbrev=False)
        parser.add_argument("--message", required=True)
        parser.add_argument("--api-url", default=API_URL)
        parser.add_argument("--report", default=str(report_path))
        parser.add_argument("--mode", choices=("real_model", "test_double", "unverified"),
                            default="test_double" if transport is not None else "unverified")
        options = _arguments(argv, parser=parser, flags={"--message", "--api-url", "--report", "--mode"})
        query = ChatRequest(message=options.message)
        # urlsplit strips some controls; refuse them before reusing loopback validation.
        if any(ord(character) < 32 or ord(character) == 127 for character in options.api_url):
            raise ShopError("invalid_request")
        base_url = _local_url(options.api_url)
        if transport is not None and options.mode == "real_model":
            raise ShopError("invalid_request")
        path = _chat_report_path(options.report, environ=environment, env_file=env_file, initial=True)
        authorization = _authorization(environment, Path(env_file))
        model_key = (environment["LLM_API_KEY"] if "LLM_API_KEY" in environment else
                     dotenv_values(env_file, interpolate=False).get("LLM_API_KEY", ""))
        secrets = (authorization, authorization.partition(" ")[2], model_key if isinstance(model_key, str) else "")
        report.update(mode=options.mode, provenance={"declared_mode": options.mode, "declaration_source": "runner",
                      "external_model_verified": False, "injected_transport": transport is not None})
        _save_report(path, report, authorization)

        async def request_chat():
            # Five model rounds each have their own timeout; leave time for tools.
            async with httpx.AsyncClient(base_url=base_url, timeout=600.0, trust_env=False,
                                         follow_redirects=False, transport=transport) as client:
                return await client.post("/agent/chat", json=query.model_dump(mode="json", exclude_none=True),
                                         headers={"Authorization": authorization})

        response = asyncio.run(request_chat())
        report["http_status"] = response.status_code
        if response.status_code != 200:
            code = "login_required" if response.status_code == 401 else "chat_http_error"
            if 400 <= response.status_code <= 599 and response.status_code != 401:
                try:
                    body = response.json()
                    error = body.get("error") if isinstance(body, dict) else None
                    candidate = error.get("code") if isinstance(error, dict) else None
                    if isinstance(candidate, str) and candidate in _KNOWN_HTTP_CODES:
                        code = candidate
                except (ValueError, TypeError):
                    pass
            _failure(report, code)
        else:
            try:
                result = _public_response(response.json(), secrets)
            except (ValueError, TypeError, ValidationError):
                _failure(report, "chat_invalid_response")
            else:
                report.update(status="passed", result=result.model_dump(mode="json"))
    except SystemExit as error:
        return int(error.code or 0)
    except ShopError as error:
        _failure(report, error.code)
    except ValidationError:
        _failure(report, "invalid_request")
    except OSError:
        _failure(report, "report_write_failed")
    except httpx.HTTPError:
        _failure(report, "chat_http_error")
    except Exception:
        _failure(report, "chat_smoke_error")
    report["elapsed_seconds"] = perf_counter() - began
    try:
        if path is not None:
            _save_report(_chat_report_path(path, environ=environment, env_file=env_file), report, authorization)
    except (OSError, ShopError):
        _failure(report, "report_write_failed")
    summary = {"status": report["status"]}
    if "error" in report:
        summary["error_code"] = report["error"]["code"]
    elif "result" in report:
        result = report["result"]
        summary.update(chat_status=result["status"], model_rounds=result["model_rounds"],
                       tools=[trace["name"] for trace in result["tool_calls"]],
                       product_count=len(result["products"]), source_count=len(result["sources"]))
    print(json.dumps(summary, ensure_ascii=False), file=stdout)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
