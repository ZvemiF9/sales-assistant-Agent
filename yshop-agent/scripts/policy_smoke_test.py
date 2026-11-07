"""One authenticated GET to the local FastAPI policy endpoint; no business POST."""

import asyncio
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence, TextIO
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from app.rag.config import SERVICE_ROOT
from app.rag.schemas import PolicyAnswer, PolicyQuery
from scripts.eval_policy import _answer_contract_failures, _arguments, _report_path
from scripts.smoke_test import _SafeParser, _authorization, _save_report


REPORT_PATH = SERVICE_ROOT / "data" / "day3-policy-smoke.json"
API_URL = "http://127.0.0.1:8010"
_KNOWN_HTTP_CODES = {"login_required", "invalid_request", "shop_unavailable", "shop_invalid_response",
                     "shop_rejected", "rag_not_built", "rag_index_incompatible", "rag_index_corrupt",
                     "rag_model_unavailable", "rag_unavailable"}


def _local_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parts.username is not None or parts.password is not None or parts.query or parts.fragment
                or parts.path not in {"", "/"} or (parts.port is not None and not 0 < parts.port <= 65535)
                or any(character.isspace() for character in value) or "\\" in value):
            raise ValueError
    except ValueError:
        raise ShopError("invalid_request") from None
    return value.rstrip("/")


def _cropped_answer(value: dict) -> PolicyAnswer:
    # Discard upstream extras before validating the public model.
    if not isinstance(value, dict) or not isinstance(value.get("sources"), list):
        raise ValueError
    fields = set(PolicyAnswer.model_fields)
    from app.rag.schemas import PolicySource
    source_fields = set(PolicySource.model_fields)
    cropped = {key: value[key] for key in fields if key in value}
    cropped["sources"] = [{key: source[key] for key in source_fields if key in source}
                          for source in value["sources"] if isinstance(source, dict)]
    if len(cropped["sources"]) != len(value["sources"]):
        raise ValueError
    return PolicyAnswer.model_validate(cropped)


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_file: Path = SERVICE_ENV_FILE, transport: httpx.AsyncBaseTransport | None = None,
         report_path: Path = REPORT_PATH, stdout: TextIO | None = None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    path = None
    authorization = ""
    report = {"schema_version": 1, "status": "pending", "method": "GET", "endpoint": "/agent/policies",
              "scope": "one local authenticated policy request; no business write request"}
    began = perf_counter()
    try:
        parser = _SafeParser(description="使用现有本地商城身份 GET 查询演示规则。", allow_abbrev=False)
        parser.add_argument("--question", required=True)
        parser.add_argument("--api-url", default=API_URL, help="本地 FastAPI origin，默认 http://127.0.0.1:8010")
        parser.add_argument("--report", default=str(report_path))
        options = _arguments(argv, parser=parser, flags={"--question", "--api-url", "--report"})
        question = PolicyQuery(question=options.question).question
        base_url = _local_url(options.api_url)
        path = _report_path(options.report, environ=environ, env_file=env_file)
        authorization = _authorization(os.environ if environ is None else environ, Path(env_file))
        _save_report(path, report, authorization)

        async def request_policy():
            async with httpx.AsyncClient(base_url=base_url, timeout=30.0, trust_env=False,
                                         follow_redirects=False, transport=transport) as client:
                return await client.get("/agent/policies", params={"question": question},
                                        headers={"Authorization": authorization})

        response = asyncio.run(request_policy())
        report["http_status"] = response.status_code
        if response.status_code != 200:
            code = "login_required" if response.status_code == 401 else "policy_http_error"
            try:
                body = response.json()
                candidate = body.get("error", {}).get("code") if isinstance(body, dict) else None
                if candidate in _KNOWN_HTTP_CODES:
                    code = candidate
            except (ValueError, TypeError, AttributeError):
                pass
            if code in _KNOWN_HTTP_CODES:
                raise ShopError(code)
            report.update(status="failed", error={"code": code, "message": "本地规则接口请求失败，请检查服务状态。"})
        else:
            try:
                answer = _cropped_answer(response.json())
                if _answer_contract_failures(question, answer) or "演示" not in answer.notice:
                    raise ValueError
            except (ValueError, TypeError, ValidationError):
                report.update(status="failed", error={"code": "policy_invalid_response", "message": "规则接口返回的数据无效。"})
            else:
                report.update(status="passed", result=answer.model_dump(mode="json"))
    except SystemExit as error:
        return int(error.code or 0)
    except ShopError as error:
        report.update(status="failed", error={"code": error.code, "message": error.message})
    except ValidationError:
        report.update(status="failed", error={"code": "invalid_request", "message": "请求参数无效。"})
    except OSError:
        report.update(status="failed", error={"code": "report_write_failed", "message": "无法保存规则联调报告，请检查项目数据目录权限。"})
    except httpx.HTTPError:
        report.update(status="failed", error={"code": "policy_http_error", "message": "本地规则接口请求失败，请检查服务状态。"})
    except Exception:
        report.update(status="failed", error={"code": "policy_smoke_error", "message": "规则联调失败，请检查本地服务配置。"})
    report["elapsed_seconds"] = perf_counter() - began
    try:
        if path is not None:
            _save_report(_report_path(path, environ=environ, env_file=env_file), report, authorization)
    except (OSError, ShopError):
        report.update(status="failed", error={"code": "report_write_failed", "message": "无法保存规则联调报告，请检查项目数据目录权限。"})
    summary = {key: report[key] for key in ("status", "http_status", "error") if key in report}
    if "result" in report:
        summary["policy_status"] = report["result"]["status"]
        summary["citation_count"] = len(report["result"]["sources"])
    print(json.dumps(summary, ensure_ascii=False), file=stdout)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
