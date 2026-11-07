"""CLI contracts: fail closed, persist honest metrics, never expose credentials."""

import importlib
import importlib.util
import io
import json
from pathlib import Path

import httpx
import pytest

from app.errors import ShopError
from app.rag.config import RagSettings
from app.rag.documents import load_policy_chunks
from app.rag.schemas import PolicyAnswer, PolicySource
from app.rag.service import DEMO_NOTICE


AUTH = "Bearer cli-test-only-secret"


def _module(name):
    assert importlib.util.find_spec(name) is not None, f"missing required CLI: {name}"
    return importlib.import_module(name)


def _case(**updates):
    result = {"id": "return", "kind": "policy", "topic": "returns",
              "question": "不合适能退吗？", "expected_status": "answered",
              "expected_document_ids": ["06-return-eligibility"],
              "relevant_document_ids": ["06-return-eligibility"]}
    result.update(updates)
    return result


def _cases(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}, ensure_ascii=False), encoding="utf-8")
    return path


def _answer(*document_ids):
    chunks = {chunk.document_id: chunk for chunk in load_policy_chunks(RagSettings(_env_file=None).knowledge_dir)}
    sources = [PolicySource(**chunks[doc].metadata, text=chunks[doc].text, score=0.8) for doc in document_ids]
    return PolicyAnswer(question="不合适能退吗？", status="answered" if sources else "no_evidence",
                        answer=("以下为检索到的演示规则原文摘录：\n\n" +
                                "\n\n".join(f"[{index}] {source.text}" for index, source in enumerate(sources, 1)))
                               if sources else "演示资料中未找到足够依据，请通过商城已有客服入口咨询。",
                        sources=sources, notice=DEMO_NOTICE)


class RecordedService:
    def __init__(self, answer=None, error=None):
        self.settings = RagSettings(_env_file=None)
        self.result = _answer("06-return-eligibility") if answer is None else answer
        self.error = error
        self.calls = []

    def search_policy(self, question, *, top_k):
        # Explicit pre-selection candidates for the deterministic CLI contract.
        return self.result.sources[:top_k]

    def answer(self, question):
        self.calls.append(question)
        if self.error:
            raise self.error
        return self.result.model_copy(update={"question": question})


@pytest.mark.parametrize("extra", [["--token", AUTH], ["--report", "a", "--report", "b"], ["--unknown", AUTH]])
def test_eval_invalid_arguments_are_sanitized_and_nonzero(tmp_path, extra):
    module = _module("scripts.eval_policy")
    output = io.StringIO()
    service = RecordedService()
    assert module.main(extra, service=service, report_path=tmp_path / "report.json", stdout=output) != 0
    assert not service.calls
    assert AUTH not in output.getvalue()
    assert json.loads(output.getvalue())["error"]["code"] == "invalid_request"


def test_eval_reports_irrelevant_exact_citation_as_failure(tmp_path):
    module = _module("scripts.eval_policy")
    path = _cases(tmp_path, [_case()])
    report = tmp_path / "report.json"
    service = RecordedService(_answer("06-return-eligibility", "12-coupons"))
    assert module.main(["--cases", str(path), "--report", str(report)], service=service, stdout=io.StringIO()) == 1
    result = json.loads(report.read_text(encoding="utf-8"))
    row = result["cases"][0]
    assert row["top_k_hit"] is True
    assert row["exact_text_fidelity"] is True
    assert row["all_citations_relevant"] is False
    assert row["citation_relevance_precision"] == 0.5
    assert row["status"] == "failed"
    assert result["summary"]["failed"] == 1


def test_eval_absent_policy_with_similar_source_fails(tmp_path):
    module = _module("scripts.eval_policy")
    path = _cases(tmp_path, [_case(expected_status="no_evidence", expected_document_ids=[], relevant_document_ids=[])])
    report = tmp_path / "report.json"
    assert module.main(["--cases", str(path), "--report", str(report)], service=RecordedService(), stdout=io.StringIO()) == 1
    row = json.loads(report.read_text(encoding="utf-8"))["cases"][0]
    assert row["status_matches"] is False
    assert "unexpected_evidence" in row["failures"]


def test_eval_continues_after_query_error_and_records_safe_failure(tmp_path):
    module = _module("scripts.eval_policy")
    path = _cases(tmp_path, [_case(id="first"), _case(id="second")])
    report = tmp_path / "report.json"
    service = RecordedService(error=ShopError("rag_index_corrupt"))
    assert module.main(["--cases", str(path), "--report", str(report)], service=service, stdout=io.StringIO()) == 1
    data = json.loads(report.read_text(encoding="utf-8"))
    assert len(service.calls) == 2
    assert data["summary"]["failed"] == 2
    assert all(row["error"]["code"] == "rag_index_corrupt" for row in data["cases"])


def test_eval_success_and_elapsed_are_saved(tmp_path):
    module = _module("scripts.eval_policy")
    path = _cases(tmp_path, [_case()])
    report = tmp_path / "report.json"
    assert module.main(["--cases", str(path), "--report", str(report)], service=RecordedService(), stdout=io.StringIO()) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["cases"][0]["elapsed_seconds"] >= 0
    assert data["summary"]["passed"] == 1


def test_eval_abstention_does_not_inflate_citation_metrics(tmp_path):
    module = _module("scripts.eval_policy")
    path = _cases(tmp_path, [_case(kind="absent_policy", expected_status="no_evidence",
                                 expected_document_ids=[], relevant_document_ids=[])])
    report = tmp_path / "report.json"
    assert module.main(["--cases", str(path), "--report", str(report)],
                       service=RecordedService(_answer()), stdout=io.StringIO()) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["cases"][0]["exact_text_fidelity"] is None
    assert data["cases"][0]["all_citations_relevant"] is None
    assert data["summary"]["exact_text_fidelity_cases"] == 0
    assert data["summary"]["all_citations_relevant_cases"] == 0


@pytest.mark.parametrize("bad", [{"cases": []}, {"schema_version": 1, "cases": [_case(), _case()]},
                                 {"schema_version": 1, "cases": [_case(expected_status="maybe")]}])
def test_eval_rejects_invalid_case_file_before_query(tmp_path, bad):
    module = _module("scripts.eval_policy")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    service = RecordedService()
    output = io.StringIO()
    assert module.main(["--cases", str(path)], service=service, report_path=tmp_path / "report.json", stdout=output) != 0
    assert not service.calls
    assert json.loads(output.getvalue())["error"]["code"] == "invalid_cases"


def test_eval_report_write_failure_prevents_query(tmp_path):
    module = _module("scripts.eval_policy")
    path = _cases(tmp_path, [_case()])
    blocked = tmp_path / "blocked.json"
    blocked.mkdir()
    service = RecordedService()
    output = io.StringIO()
    assert module.main(["--cases", str(path), "--report", str(blocked)], service=service, stdout=output) != 0
    assert not service.calls
    assert json.loads(output.getvalue())["error"]["code"] == "report_write_failed"
    assert str(blocked) not in output.getvalue()


def _smoke(tmp_path, handler, argv=None, **kwargs):
    output = io.StringIO()
    path = tmp_path / "smoke.json"
    code = _module("scripts.policy_smoke_test").main(
        ["--question", "不合适能退吗？"] if argv is None else argv,
        environ={"YSHOP_AUTHORIZATION": AUTH}, env_file=tmp_path / "unused.env",
        transport=httpx.MockTransport(handler), report_path=path, stdout=output, **kwargs)
    return code, json.loads(output.getvalue()), path


@pytest.mark.parametrize("argv", [[], ["--question", " "], ["--question", "x" * 1001],
                                 ["--question", "a", "--question", "b"], ["--token", AUTH]])
def test_smoke_invalid_arguments_never_send_http(tmp_path, argv):
    def handler(request):
        pytest.fail("invalid arguments must not send HTTP")
    code, output, _ = _smoke(tmp_path, handler, argv)
    assert code != 0
    assert output["error"]["code"] == "invalid_request"
    assert AUTH not in json.dumps(output)


def test_smoke_only_gets_local_authenticated_policy_and_crops_fields(tmp_path):
    requests = []
    def handler(request):
        requests.append(request)
        body = _answer("06-return-eligibility").model_dump(mode="json")
        body["internal"] = {"authorization": AUTH}
        body["sources"][0]["internal"] = AUTH
        return httpx.Response(200, json=body)
    code, output, path = _smoke(tmp_path, handler)
    assert code == 0
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.host == "127.0.0.1"
    assert requests[0].url.path == "/agent/policies"
    assert requests[0].headers["authorization"] == AUTH
    report = path.read_text(encoding="utf-8")
    assert AUTH not in report
    assert "internal" not in report
    assert output["status"] == "passed"


@pytest.mark.parametrize("status,code", [(401, "login_required"), (503, "rag_not_built"), (302, "policy_http_error")])
def test_smoke_http_failures_nonzero_and_do_not_echo_response(tmp_path, status, code):
    def handler(request):
        return httpx.Response(status, json={"error": {"code": code, "message": AUTH}}, headers={"Location": "https://example.com"})
    exit_code, output, path = _smoke(tmp_path, handler)
    assert exit_code != 0
    assert output["error"]["code"] == code
    assert AUTH not in path.read_text(encoding="utf-8")


def test_smoke_malformed_success_is_nonzero(tmp_path):
    exit_code, output, _ = _smoke(tmp_path, lambda request: httpx.Response(200, json={"answer": AUTH}))
    assert exit_code != 0
    assert output["error"]["code"] == "policy_invalid_response"
    assert AUTH not in json.dumps(output)


def test_smoke_missing_authorization_fails_before_http(tmp_path):
    module = _module("scripts.policy_smoke_test")
    output = io.StringIO()
    assert module.main(["--question", "退货条件"], environ={}, env_file=tmp_path / "absent.env",
                       transport=httpx.MockTransport(lambda request: pytest.fail("no login")),
                       report_path=tmp_path / "report.json", stdout=output) != 0
    assert json.loads(output.getvalue())["error"]["code"] == "login_required"


def test_smoke_report_failure_is_nonzero_before_http(tmp_path):
    module = _module("scripts.policy_smoke_test")
    blocked = tmp_path / "directory.json"
    blocked.mkdir()
    output = io.StringIO()
    assert module.main(["--question", "退货条件"], environ={"YSHOP_AUTHORIZATION": AUTH},
                       transport=httpx.MockTransport(lambda request: pytest.fail("report failure before HTTP")),
                       report_path=blocked, stdout=output) != 0
    assert json.loads(output.getvalue())["error"]["code"] == "report_write_failed"
    assert str(blocked) not in output.getvalue()
