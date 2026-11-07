"""Offline, pre-labelled development evaluation; never calls business tools."""

import json
from contextlib import contextmanager
from pathlib import Path
import socket
import sys
from time import perf_counter
from typing import Sequence, TextIO
from unittest.mock import patch
import uuid

from pydantic import ValidationError

from app.errors import ShopError
from app.rag.config import RagSettings, SERVICE_ROOT
from app.rag.documents import load_policy_chunks
from app.rag.embeddings import LocalEmbeddings
from app.rag.evidence import CHUNK_WEIGHT, MAX_SOURCES, SENTENCE_WEIGHT
from app.rag.schemas import PolicyQuery
from app.rag.service import PolicyService
from app.rag.store import PolicyStore
from scripts.smoke_test import _SafeParser, _save_report


CASES_PATH = SERVICE_ROOT / "evals" / "policy_cases.json"
REPORT_PATH = SERVICE_ROOT / "data" / "day3-policy-eval.json"
NO_EVIDENCE_ANSWER = "演示资料中未找到足够依据，请通过商城已有客服入口咨询。"
EXTRACTIVE_PREFIX = "以下为检索到的演示规则原文摘录：\n\n"
# Historical evidence is immutable. New runs use a new report filename.
_PRESERVED_REPORT_NAMES = {
    "day1-http-checks.json", "day1-java-checks.json", "day1-smoke-report-add.json",
    "day1-smoke-report.json", "day2-http-checks.json", "day2-search-huawei.json",
    "day2-search-makeup.json", "day2-search-smoke-report.json",
    "day3-evidence-resume-pilot.json", "day3-http-checks.json", "day3-model-sanity.json",
    "day3-policy-eval-first.json", "day3-policy-eval-tuned-01-20261025.json",
    "day3-policy-pilot.json", "day3-query-prefix-pilot.json", "day3-real-rag-smoke.json",
    "day3-resume-preservation-20261025.json", "day3-sentence-evidence-pilot.json",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else SERVICE_ROOT / path
    if (not path.resolve().is_relative_to(SERVICE_ROOT)
            or any(p.is_symlink() or p.is_junction() for p in (path, *path.parents))):
        raise ShopError("invalid_request")
    return path.resolve()


def _report_path(value: str | Path, *, settings=None, case_path: Path | None = None,
                 environ=None, env_file=None) -> Path:
    """Validate report and atomic temporary destinations before any write/query."""
    path = _project_path(value)
    relative = path.relative_to(SERVICE_ROOT.resolve())
    if (path.suffix.lower() != ".json" or not relative.parts or relative.parts[0] not in {"data", ".tmp"}
            or path.name.lower() in _PRESERVED_REPORT_NAMES
            or path.name.lower() == "manifest.json"
            or any(part.lower() in {"knowledge", "chroma", ".cache", "fastembed"} for part in relative.parts[:-1])):
        raise ShopError("invalid_request")
    # Settings parsing performs no model/index/service construction. Its errors
    # remain sanitized; supplied service settings do not override env protection.
    directory_fields = ("knowledge_dir", "chroma_dir", "model_cache_dir")
    overrides = {field: environ["RAG_" + field.upper()] for field in directory_fields
                 if environ is not None and "RAG_" + field.upper() in environ}
    try:
        configured = RagSettings(**overrides) if env_file is None else RagSettings(_env_file=env_file, **overrides)
    except (ValidationError, ValueError, OSError):
        raise ShopError("invalid_request") from None
    directories = [SERVICE_ROOT / "knowledge", SERVICE_ROOT / ".cache", SERVICE_ROOT / "data" / "chroma"]
    for current in (configured, settings):
        if current is not None:
            directories.extend(getattr(current, field) for field in directory_fields)
    protected = [CASES_PATH, SERVICE_ROOT / ".env"]
    if case_path is not None:
        protected.append(case_path)
    if env_file is not None:
        protected.append(Path(env_file))
    temporary = _project_path(path.with_name(path.name + ".tmp"))
    for destination in (path, temporary):
        if (any(destination.is_relative_to(Path(directory).resolve()) for directory in directories)
                or any(destination == Path(file).resolve() for file in protected)):
            raise ShopError("invalid_request")
    return path


def _answer_contract_failures(question: str, answer) -> list[str]:
    failures = []
    if PolicyQuery(question=answer.question).question != PolicyQuery(question=question).question:
        failures.append("question_mismatch")
    if answer.status == "answered":
        expected = EXTRACTIVE_PREFIX + "\n\n".join(
            f"[{index}] {source.text}" for index, source in enumerate(answer.sources, 1))
        if not answer.sources or answer.answer != expected:
            failures.append("invalid_extractive_answer")
    elif answer.status == "no_evidence" and (answer.sources or answer.answer != NO_EVIDENCE_ANSWER):
        failures.append("invalid_no_evidence_response")
    return failures


@contextmanager
def _network_guard(attempts: list):
    def refuse_connection(sock, address):
        attempts.append(True)  # Never save destinations, payloads or credentials.
        raise OSError("evaluation_network_disabled")

    with patch.object(socket.socket, "connect", refuse_connection), patch.object(socket.socket, "connect_ex", refuse_connection):
        yield


def _arguments(argv: Sequence[str] | None, *, parser: _SafeParser, flags: set[str]):
    arguments = list(sys.argv[1:] if argv is None else argv)
    seen = set()
    for argument in arguments:
        flag = argument.partition("=")[0]
        if flag in flags:
            if flag in seen:
                raise ShopError("invalid_request")
            seen.add(flag)
    return parser.parse_args(arguments)


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or not isinstance(data.get("cases"), list) or not data["cases"]):
        raise ValueError
    required = {"id", "kind", "topic", "question", "expected_status",
                "expected_document_ids", "relevant_document_ids"}
    seen = set()
    for case in data["cases"]:
        if (not isinstance(case, dict) or not required <= set(case)
                or set(case) - required - {"document_text"}
                or any(not isinstance(case[key], str) or not case[key].strip()
                       for key in ("id", "kind", "topic", "question", "expected_status"))):
            raise ValueError
        PolicyQuery(question=case["question"])
        if case["id"] in seen or case["expected_status"] not in {"answered", "no_evidence"}:
            raise ValueError
        seen.add(case["id"])
        if case["kind"] not in {"policy", "absent_policy", "user_injection", "document_injection"}:
            raise ValueError
        for key in ("expected_document_ids", "relevant_document_ids"):
            ids = case[key]
            if (not isinstance(ids, list) or any(not isinstance(item, str) or not item.strip() for item in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError
        if not set(case["expected_document_ids"]) <= set(case["relevant_document_ids"]):
            raise ValueError
        if case["expected_status"] == "answered" and not case["expected_document_ids"]:
            raise ValueError
        if case["expected_status"] == "no_evidence" and (case["expected_document_ids"] or case["relevant_document_ids"]):
            raise ValueError
        if case["kind"] == "absent_policy" and case["expected_status"] != "no_evidence":
            raise ValueError
        if case["kind"] == "document_injection":
            if (case["expected_document_ids"] != ["99-document-injection"]
                    or not isinstance(case.get("document_text"), str) or not case["document_text"].strip()):
                raise ValueError
        elif "document_text" in case:
            raise ValueError
    return data["cases"]


def _measure(case: dict, answer, chunks: dict, min_score: float, *, retrieval_candidates=None) -> dict:
    sources = answer.sources
    ids = [source.document_id for source in sources]
    expected = set(case["expected_document_ids"])
    relevant = set(case["relevant_document_ids"])
    status_matches = answer.status == case["expected_status"]
    selected_hit = bool(expected.intersection(ids)) if expected else None
    retrieval_ids = [candidate.document_id for candidate in retrieval_candidates] if retrieval_candidates is not None else None
    hit = bool(expected.intersection(retrieval_ids)) if expected and retrieval_ids is not None else None
    relevant_count = sum(doc in relevant for doc in ids)
    precision = relevant_count / len(ids) if ids else None
    citation_rows = []
    for source in sources:
        chunk = chunks.get(source.chunk_id)
        exact = (chunk is not None and source.text == chunk.text
                 and all(getattr(source, key) == value for key, value in chunk.metadata.items()))
        citation_rows.append({
            "document_id": source.document_id, "chunk_id": source.chunk_id,
            "source": source.source, "section": source.section, "version": source.version,
            "score": source.score, "relevant": source.document_id in relevant,
            "exact_original_text": exact, "excerpt_in_answer": source.text in answer.answer,
            "above_threshold": source.score >= min_score,
        })
    fidelity = all(row["exact_original_text"] and row["excerpt_in_answer"] for row in citation_rows)
    all_relevant = all(row["relevant"] for row in citation_rows)
    failures = _answer_contract_failures(case["question"], answer)
    if not status_matches:
        failures.append("status_mismatch")
    if expected and not selected_hit:
        failures.append("expected_document_missed")
    if expected and not expected <= set(ids):
        failures.append("expected_documents_incomplete")
    if case["expected_status"] == "no_evidence" and sources:
        failures.append("unexpected_evidence")
    if answer.status == "answered" and not sources:
        failures.append("answered_without_citations")
    if not fidelity:
        failures.append("text_fidelity_failure")
    if not all_relevant:
        failures.append("irrelevant_citation")
    if not all(row["above_threshold"] for row in citation_rows):
        failures.append("below_threshold_citation")
    if "演示" not in answer.notice:
        failures.append("missing_demo_notice")
    return {
        "status": "failed" if failures else "passed", "actual_status": answer.status,
        "status_matches": status_matches, "top_k_hit": hit,
        "retrieval_top_k_measured": retrieval_candidates is not None,
        "retrieval_candidates": [{"document_id": candidate.document_id, "chunk_id": candidate.chunk_id,
                                  "score": candidate.score} for candidate in retrieval_candidates]
                                if retrieval_candidates is not None else None,
        "selected_citation_expected_document_hit": selected_hit,
        "all_expected_documents_hit": expected <= set(ids) if expected else None,
        "extractive_composition_matches": "invalid_extractive_answer" not in failures if answer.status == "answered" else None,
        "exact_text_fidelity": fidelity if sources else None,
        "all_citations_relevant": all_relevant if sources else None,
        "citation_relevance_precision": precision, "citations": citation_rows,
        "failures": failures,
    }


def _query_measure(case: dict, service, originals: dict) -> dict:
    search = getattr(service, "search_policy", None)
    candidates = search(case["question"], top_k=service.settings.top_k) if callable(search) else None
    answer = service.answer(case["question"])
    return _measure(case, answer, originals, service.settings.min_score, retrieval_candidates=candidates)


def _document_injection(case: dict, settings: RagSettings) -> tuple[dict, dict]:
    # This fixture and index cannot alter the main corpus or manifest.
    root = SERVICE_ROOT / ".tmp" / "data" / ("day3-document-injection-" + uuid.uuid4().hex)
    knowledge = root / "knowledge" / "policies"
    knowledge.mkdir(parents=True)
    metadata = {"document_id": "99-document-injection", "title": "隔离文档注入测试",
                "source": "knowledge/policies/99-document-injection.md", "version": "2026-10-24.eval.v1", "kind": "demo"}
    (knowledge / "99-document-injection.md").write_text(
        "---\n" + json.dumps(metadata, ensure_ascii=False) + "\n---\n# 隔离文档注入测试\n\n## 退货与不可信指令数据\n\n"
        + case["document_text"] + "\n", encoding="utf-8")
    isolated = RagSettings(_env_file=None, knowledge_dir=knowledge, chroma_dir=root / "chroma",
                           model_cache_dir=settings.model_cache_dir, min_score=settings.min_score,
                           evidence_min_score=settings.evidence_min_score,
                           top_k=settings.top_k, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    attempts = []

    with _network_guard(attempts):
        model = LocalEmbeddings(isolated, allow_download=False)
        chunks = load_policy_chunks(knowledge, chunk_size=isolated.chunk_size, chunk_overlap=isolated.chunk_overlap)
        manifest = PolicyStore(isolated, model).build(chunks)
        service = PolicyService(isolated, embeddings=model)
        candidates = service.search_policy(case["question"], top_k=isolated.top_k)
        answer = service.answer(case["question"])
    measured = _measure(case, answer, {chunk.chunk_id: chunk for chunk in chunks}, isolated.min_score,
                        retrieval_candidates=candidates)
    if attempts:
        measured["status"] = "failed"
        measured["failures"].append("network_execution_attempted")
    measured["injection"] = {
        "isolated_corpus": True, "real_local_embeddings": True,
        "network_connection_attempts": len(attempts),
        "scope": "only this process during isolated index build and query; no live-shop write audit",
        "untrusted_instruction_in_excerpt": any("注入测试数据" in source.text for source in answer.sources),
        "index_fingerprint": manifest["index_fingerprint"],
        "fixture_directory": root.relative_to(SERVICE_ROOT).as_posix(),
    }
    return measured, manifest


def _bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _summary(rows: list[dict]) -> dict:
    labelled = [row for row in rows if row["expected_status"] == "answered"]
    sources = [source for row in rows for source in row.get("citations", [])]
    times = [row["elapsed_seconds"] for row in rows if "elapsed_seconds" in row]
    measured = [row for row in labelled if row.get("top_k_hit") is not None]
    return {
        "total": len(rows), "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "expected_answer_cases": len(labelled),
        "top_k_measured_cases": len(measured),
        "top_k_hits": sum(row["top_k_hit"] is True for row in measured) if measured else None,
        "selected_citation_expected_document_hits": sum(row.get("selected_citation_expected_document_hit") is True for row in labelled),
        "citation_count": len(sources), "relevant_citations": sum(source["relevant"] for source in sources),
        "citation_bearing_cases": sum(bool(row.get("citations")) for row in rows),
        "citation_relevance_precision": sum(source["relevant"] for source in sources) / len(sources) if sources else None,
        "exact_text_fidelity_cases": sum(row.get("exact_text_fidelity") is True for row in rows),
        "all_citations_relevant_cases": sum(row.get("all_citations_relevant") is True for row in rows),
        "elapsed_seconds_min": min(times) if times else None,
        "elapsed_seconds_max": max(times) if times else None,
    }


def main(argv: Sequence[str] | None = None, *, service=None, report_path: Path = REPORT_PATH,
         stdout: TextIO | None = None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    path = None
    case_path = None
    settings = getattr(service, "settings", None)
    report = {"schema_version": 1, "status": "pending", "cases": [],
              "evaluation_scope": "pre-labelled development fixed set; not independent generalization",
              "execution_scope": "local policy queries only; no business client or write tool"}
    started = perf_counter()
    try:
        parser = _SafeParser(description="用已缓存的真实中文模型评估固定演示规则；失败非零退出。", allow_abbrev=False)
        parser.add_argument("--cases", default=str(CASES_PATH))
        parser.add_argument("--report", default=str(report_path))
        options = _arguments(argv, parser=parser, flags={"--cases", "--report"})
        case_path = _project_path(options.cases)
        path = _report_path(options.report, settings=settings, case_path=case_path)
        try:
            cases = _load_cases(case_path)
        except (OSError, ValueError, ValidationError, UnicodeError):
            report["error"] = {"code": "invalid_cases", "message": "固定用例无效，请检查项目内评估 JSON。"}
            raise ValueError from None
        _save_report(_report_path(path, settings=settings, case_path=case_path), report, "")
        current = service if service is not None else PolicyService()
        settings = current.settings
        chunks = load_policy_chunks(settings.knowledge_dir, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        originals = {chunk.chunk_id: chunk for chunk in chunks}
        manifest_path = settings.chroma_dir / "manifest.json"
        # The service validates this manifest again before every actual query.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        report["configuration"] = {"model": settings.embedding_model, "dimension": 512,
                                   "min_score": settings.min_score, "top_k": settings.top_k,
                                   "evidence_min_score": settings.evidence_min_score,
                                   "evidence_selection": {"chunk_weight": CHUNK_WEIGHT,
                                                          "sentence_weight": SENTENCE_WEIGHT,
                                                          "max_sources": MAX_SOURCES,
                                                          "sentence_vectors": "transient; same embedding model; exact original spans",
                                                          "missing_information": "refuse unless a boolean question has a matching independent assertion"},
                                   "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
                                   "document_count": len({chunk.document_id for chunk in chunks}), "chunk_count": len(chunks),
                                   "index_fingerprint": manifest.get("index_fingerprint"),
                                   "model_fingerprint": manifest.get("model_fingerprint"),
                                   "model_cache_bytes": _bytes(settings.model_cache_dir), "index_bytes_before": _bytes(settings.chroma_dir)}
        for case in cases:
            row = {key: case[key] for key in ("id", "kind", "topic", "question", "expected_status",
                                             "expected_document_ids", "relevant_document_ids")}
            began = perf_counter()
            user_attempts = None
            try:
                if case["kind"] == "document_injection":
                    measured, _ = _document_injection(case, settings)
                elif case["kind"] == "user_injection":
                    user_attempts = []
                    with _network_guard(user_attempts):
                        measured = _query_measure(case, current, originals)
                else:
                    measured = _query_measure(case, current, originals)
                row.update(measured)
            except ShopError as error:
                row.update(status="failed", failures=["query_error"], error={"code": error.code, "message": error.message})
            except Exception:
                row.update(status="failed", failures=["evaluation_error"],
                           error={"code": "evaluation_error", "message": "评估执行失败，请检查本地模型、索引和规则配置。"})
            if user_attempts is not None:
                row["injection"] = {"isolated_corpus": False,
                    "network_connection_attempts": len(user_attempts),
                    "scope": "only this evaluation process during raw retrieval and answer; no live-shop write audit"}
                if user_attempts:
                    row["status"] = "failed"
                    row.setdefault("failures", []).append("network_execution_attempted")
            row["elapsed_seconds"] = perf_counter() - began
            report["cases"].append(row)
            _save_report(_report_path(path, settings=settings, case_path=case_path), report, "")
        report["summary"] = _summary(report["cases"])
        report["configuration"]["index_bytes_after"] = _bytes(settings.chroma_dir)
        report["status"] = "passed" if not report["summary"]["failed"] else "failed"
    except SystemExit as error:
        return int(error.code or 0)
    except ShopError as error:
        report.update(status="failed", error={"code": error.code, "message": error.message})
    except OSError:
        report.update(status="failed", error={"code": "report_write_failed", "message": "无法保存评估报告，请检查项目数据目录权限。"})
    except Exception:
        report["status"] = "failed"
        report.setdefault("error", {"code": "evaluation_error", "message": "评估执行失败，请检查本地模型、索引和规则配置。"})
    report["elapsed_seconds"] = perf_counter() - started
    try:
        if path is not None:
            _save_report(_report_path(path, settings=settings, case_path=case_path), report, "")
    except (OSError, ShopError):
        report.update(status="failed", error={"code": "report_write_failed", "message": "无法保存评估报告，请检查项目数据目录权限。"})
    summary = {key: report[key] for key in ("status", "summary", "error") if key in report}
    print(json.dumps(summary, ensure_ascii=False), file=stdout)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
