"""Task-three review regressions: protect inputs and measure actual contracts."""

import io
import json
import socket
from types import SimpleNamespace

import httpx
import pytest

from app.rag.config import RagSettings
from app.rag.schemas import PolicyAnswer, PolicySource
from app.rag.service import DEMO_NOTICE
from scripts import eval_policy, policy_smoke_test


QUESTION = "不合适能退吗？"
NO_EVIDENCE = "演示资料中未找到足够依据，请通过商城已有客服入口咨询。"


def _case(**updates):
    case = {"id": "returns", "kind": "policy", "topic": "returns", "question": QUESTION,
            "expected_status": "answered", "expected_document_ids": ["return-policy"],
            "relevant_document_ids": ["return-policy"]}
    case.update(updates)
    return case


def _source(document_id="return-policy"):
    return PolicySource(document_id=document_id, title="演示退货", section="退货条件",
                        source="knowledge/policies/return.md", version="review.v1", kind="demo",
                        chunk_id=document_id + "-chunk", text="演示规则：符合条件可申请退货。", score=0.8)


def _answer(source=None):
    source = _source() if source is None else source
    return PolicyAnswer(question=QUESTION, status="answered", sources=[source], notice=DEMO_NOTICE,
                        answer="以下为检索到的演示规则原文摘录：\n\n[1] " + source.text)


def _originals(*sources):
    return {source.chunk_id: SimpleNamespace(text=source.text, document_id=source.document_id,
             chunk_id=source.chunk_id, metadata=source.model_dump(exclude={"text", "score"}))
            for source in sources}


class ReviewService:
    def __init__(self, project, answer=None, candidates=None):
        self.settings = RagSettings(_env_file=None, knowledge_dir=project / "knowledge" / "policies",
                                   chroma_dir=project / "data" / "chroma",
                                   model_cache_dir=project / ".cache" / "models")
        self.result = _answer() if answer is None else answer
        self.candidates = [_source()] if candidates is None else candidates
        self.calls = []
        self.search_calls = []

    def search_policy(self, question, *, top_k):
        self.search_calls.append((question, top_k))
        return self.candidates

    def answer(self, question):
        self.calls.append(question)
        return self.result


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "service"
    root.mkdir()
    monkeypatch.setattr(eval_policy, "SERVICE_ROOT", root)
    monkeypatch.setattr(policy_smoke_test, "SERVICE_ROOT", root)
    monkeypatch.setattr(eval_policy, "load_policy_chunks", lambda *args, **kwargs: list(_originals(_source()).values()))
    return root


def _cases(project, case=None, relative="evals/cases.json"):
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "cases": [_case() if case is None else case]},
                               ensure_ascii=False), encoding="utf-8")
    return path


PROTECTED = [".env", "evals/policy_cases.json", "knowledge/policies/return.md",
             ".cache/models/model.json", "data/chroma/manifest.json", "data/chroma/chroma.sqlite3",
             "data/day3-policy-eval-first.json", "data/day3-policy-eval-tuned-01-20261025.json",
             "README.md"]


@pytest.mark.parametrize("cli", ["eval", "smoke"])
@pytest.mark.parametrize("relative", PROTECTED)
def test_report_never_overwrites_protected_file_or_starts_query(project, cli, relative):
    path = _cases(project)
    destination = project / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = b"protected review fixture, no credentials"
    destination.write_bytes(original)
    service = ReviewService(project)
    requests = []
    output = io.StringIO()
    if cli == "eval":
        code = eval_policy.main(["--cases", str(path), "--report", str(destination)],
                                service=service, stdout=output)
    else:
        code = policy_smoke_test.main(["--question", QUESTION, "--report", str(destination)],
            environ={"YSHOP_AUTHORIZATION": "Bearer test-only"}, env_file=project / "unused.env",
            transport=httpx.MockTransport(lambda request: requests.append(request) or
                                          httpx.Response(200, json=_answer().model_dump())), stdout=output)
    assert destination.read_bytes() == original
    assert code != 0
    assert service.calls == [] and service.search_calls == [] and requests == []
    assert json.loads(output.getvalue())["error"]["code"] == "invalid_request"


def test_eval_report_cannot_overwrite_its_case_input_inside_data(project):
    case_path = _cases(project, relative="data/input.json")
    original = case_path.read_bytes()
    service = ReviewService(project)
    assert eval_policy.main(["--cases", str(case_path), "--report", str(case_path)],
                            service=service, stdout=io.StringIO()) != 0
    assert case_path.read_bytes() == original
    assert service.calls == [] and service.search_calls == []


@pytest.mark.parametrize("cli", ["eval", "smoke"])
@pytest.mark.parametrize("directory", ["RAG_CHROMA_DIR", "RAG_KNOWLEDGE_DIR", "RAG_MODEL_CACHE_DIR"])
def test_reports_respect_configured_protected_directory_before_any_query(project, monkeypatch, cli, directory):
    protected = project / "data" / "custom-protected"
    monkeypatch.setenv(directory, str(protected))
    destination = protected / "report.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"keep-configured-data")
    case_path = _cases(project)
    service = ReviewService(project)
    output = io.StringIO()
    if cli == "eval":
        code = eval_policy.main(["--cases", str(case_path), "--report", str(destination)],
                                service=service, stdout=output)
    else:
        code = policy_smoke_test.main(["--question", QUESTION, "--report", str(destination)],
            environ={"YSHOP_AUTHORIZATION": "Bearer test-only", directory: str(protected)},
            env_file=project / "unused.env", stdout=output,
            transport=httpx.MockTransport(lambda request: pytest.fail("protected report must precede HTTP")))
    assert code != 0
    assert destination.read_bytes() == b"keep-configured-data"
    assert service.calls == [] and service.search_calls == []
    assert json.loads(output.getvalue())["error"]["code"] == "invalid_request"


def test_measure_without_raw_candidates_does_not_claim_retrieval_hit():
    measured = eval_policy._measure(_case(), _answer(), _originals(_source()), 0.53)
    assert measured["top_k_hit"] is None
    assert measured["selected_citation_expected_document_hit"] is True


def test_eval_reports_raw_retrieval_hit_separately_from_selected_citation(project):
    case_path = _cases(project)
    report_path = project / "data" / "review.json"
    service = ReviewService(project, answer=_answer(_source("unrelated")), candidates=[_source()])
    assert eval_policy.main(["--cases", str(case_path), "--report", str(report_path)],
                            service=service, stdout=io.StringIO()) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = report["cases"][0]
    assert row["top_k_hit"] is True
    assert row["selected_citation_expected_document_hit"] is False
    assert report["summary"]["top_k_hits"] == 1
    assert report["summary"]["top_k_measured_cases"] == 1
    assert report["summary"]["selected_citation_expected_document_hits"] == 0
    assert service.search_calls == [(QUESTION, service.settings.top_k)]


def test_summary_records_unmeasured_retrieval_without_inventing_zero_hits():
    row = eval_policy._measure(_case(), _answer(), _originals(_source()), 0.53)
    row.update(expected_status="answered", elapsed_seconds=0.1)
    summary = eval_policy._summary([row])
    assert summary["top_k_measured_cases"] == 0
    assert summary["top_k_hits"] is None
    assert summary["selected_citation_expected_document_hits"] == 1


@pytest.mark.parametrize("change,failure", [
    ({"question": "发票怎么开？"}, "question_mismatch"),
    ({"answer": _answer().answer + "\n商城保证无条件永久退款。"}, "invalid_extractive_answer"),
])
def test_measure_rejects_wrong_question_or_unsupported_added_answer(change, failure):
    measured = eval_policy._measure(_case(), _answer().model_copy(update=change), _originals(_source()), 0.53)
    assert measured["status"] == "failed"
    assert failure in measured["failures"]


def test_measure_rejects_no_evidence_text_that_only_mentions_keywords():
    answer = PolicyAnswer(question=QUESTION, status="no_evidence", sources=[], notice=DEMO_NOTICE,
                          answer="未找到客服，但保证无条件退款。")
    measured = eval_policy._measure(_case(expected_status="no_evidence", expected_document_ids=[],
        relevant_document_ids=[]), answer, {}, 0.53)
    assert measured["status"] == "failed"
    assert "invalid_no_evidence_response" in measured["failures"]


@pytest.mark.parametrize("body", [
    _answer().model_copy(update={"answer": _answer().answer + "\n商城保证无条件永久退款。"}),
    PolicyAnswer(question=QUESTION, status="no_evidence", sources=[], notice=DEMO_NOTICE,
                 answer="未找到客服，但保证无条件退款。"),
])
def test_smoke_rejects_text_outside_the_extractive_or_abstention_contract(project, body):
    output = io.StringIO()
    assert policy_smoke_test.main(["--question", QUESTION],
        environ={"YSHOP_AUTHORIZATION": "Bearer test-only"}, env_file=project / "unused.env",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body.model_dump())),
        report_path=project / "data" / "smoke-review.json", stdout=output) != 0
    assert json.loads(output.getvalue())["error"]["code"] == "policy_invalid_response"


@pytest.mark.parametrize("phase", ["retrieval", "answer"])
@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_user_injection_blocks_and_records_connection_attempts(project, monkeypatch, phase, method):
    reached_socket = []
    monkeypatch.setattr(socket.socket, method, lambda *args: reached_socket.append(True) or 0)
    service = ReviewService(project)
    operation = service.search_policy if phase == "retrieval" else service.answer

    def attempt_then_return(*args, **kwargs):
        with socket.socket() as sock:
            try:
                getattr(sock, method)(("127.0.0.1", 9))
            except OSError:
                pass
        return operation(*args, **kwargs)

    monkeypatch.setattr(service, "search_policy" if phase == "retrieval" else "answer", attempt_then_return)
    case_path = _cases(project, _case(kind="user_injection"))
    destination = project / "data" / "injection-review.json"
    code = eval_policy.main(["--cases", str(case_path), "--report", str(destination)],
                            service=service, stdout=io.StringIO())
    assert code != 0
    assert reached_socket == []
    row = json.loads(destination.read_text(encoding="utf-8"))["cases"][0]
    assert row["injection"]["network_connection_attempts"] == 1
    assert "network_execution_attempted" in row["failures"]
    assert "no live-shop write audit" in row["injection"]["scope"]
