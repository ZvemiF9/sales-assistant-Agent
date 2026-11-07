"""Day 9 evaluator contracts; transport substitutions occur only at HTTP edges."""

import asyncio
import importlib
import importlib.util
import io
import json
from pathlib import Path
import socket

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    assert importlib.util.find_spec("scripts." + name) is not None, "Day 9 module missing: " + name
    return importlib.import_module("scripts." + name)


def manifest():
    return json.loads((ROOT / "evals/day9_cases.json").read_text(encoding="utf-8"))


def test_frozen_manifest_has_exact_groups_modes_and_existing_unique_selectors():
    cases = module("day9_cases").load_cases()
    assert len(cases) == 50
    assert {group: sum(c["group"] == group for c in cases) for group in
            ("product", "policy", "multi_turn", "confirmation", "identity_failure")} == {
                "product": 10, "policy": 10, "multi_turn": 10,
                "confirmation": 10, "identity_failure": 10}
    assert sum(c["mode"] == "live" for c in cases) == 30
    assert sum(c["mode"] == "deterministic" for c in cases) == 20
    assert sum(step["sse"] for c in cases if c["group"] == "product" for step in c["steps"]) >= 2


@pytest.mark.parametrize("mutation", ["bool_schema", "count", "duplicate", "bool_reference", "unknown_expectation"])
def test_manifest_rejects_malformed_before_requests(tmp_path, mutation):
    day = module("day9_cases")
    value = manifest()
    if mutation == "bool_schema":
        value["schema_version"] = True
    elif mutation == "count":
        value["cases"].pop()
    elif mutation == "duplicate":
        value["cases"][1]["id"] = value["cases"][0]["id"]
    elif mutation == "bool_reference":
        value["cases"][0]["steps"][0]["expect"]["reference_index"] = True
    else:
        value["cases"][0]["steps"][0]["expect"]["arbitrary_success"] = True
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        day.load_cases(path)


def test_summary_keeps_failed_skipped_not_run_and_missing_usage_null():
    rows = [{"id": str(i), "group": "product", "mode": "live", "status": s}
            for i, s in enumerate(("passed", "failed", "skipped", "not_run"))]
    result = module("day9_cases").summarize(rows)
    assert result["case_success_rate"] == 0.25
    assert result["total"] == 4
    assert result["modes"]["live"]["total"] == 4
    assert result["timings"]["client_elapsed_ms"] == {
        "count": 0, "median": None, "p95": None, "max": None}
    assert result["usage"]["input_tokens"] is None
    assert result["citations"]["fidelity_rate"] is None


def test_summary_nearest_rank_p95_and_median_count():
    rows = [{"group": "product", "mode": "live", "status": "passed",
             "steps": [{"client_elapsed_ms": n}]} for n in range(1, 21)]
    stats = module("day9_cases").summarize(rows)["timings"]["client_elapsed_ms"]
    assert stats == {"count": 20, "median": 10.5, "p95": 19, "max": 20}


def test_summary_separates_live_phases_deterministic_cases_and_first_use():
    rows = [
        {"id": "live-first", "group": "product", "mode": "live", "status": "passed",
         "steps": [{"client_elapsed_ms": 5, "model_elapsed_ms": 3,
                    "independent_verification_elapsed_ms": 2,
                    "first_use_labels": ["model", "product"]}]},
        {"id": "live-rag", "group": "policy", "mode": "live", "status": "passed",
         "steps": [{"client_elapsed_ms": 9, "first_use_labels": ["rag"]}]},
        {"id": "det-a", "group": "confirmation", "mode": "deterministic", "status": "passed",
         "elapsed_ms": 10},
        {"id": "det-b", "group": "confirmation", "mode": "deterministic", "status": "failed",
         "elapsed_ms": 30},
        {"id": "det-unmeasured", "group": "confirmation", "mode": "deterministic", "status": "not_run",
         "elapsed_ms": None},
    ]
    result = module("day9_cases").summarize(rows)
    assert result["timings_by_mode"]["live"]["client_elapsed_ms"] == {
        "count": 2, "median": 7.0, "p95": 9, "max": 9}
    assert result["timings_by_mode"]["live"]["independent_verification_elapsed_ms"] == {
        "count": 1, "median": 2, "p95": 2, "max": 2}
    assert result["timings_by_mode"]["deterministic"]["case_elapsed_ms"] == {
        "count": 2, "median": 20.0, "p95": 30, "max": 30}
    assert result["first_use"] == {
        "model": {"case_id": "live-first", "step_index": 0},
        "product": {"case_id": "live-first", "step_index": 0},
        "rag": {"case_id": "live-rag", "step_index": 0},
    }


def test_first_use_labels_follow_observed_activity_only():
    live = module("day9_live")
    seen = set()
    observations = [
        {"phase": "complete", "category": "model", "forwarded": True},
        {"phase": "complete", "category": "shop_read", "path": "agent/products", "forwarded": True},
    ]
    tool_calls = [{"name": "search_policy", "status": "succeeded"}]
    assert live.first_use_labels(seen, observations, tool_calls) == ["model", "product", "rag"]
    assert live.first_use_labels(seen, observations, tool_calls) == []
    assert live.first_use_labels(set(), [], [{"name": "search_policy", "status": "failed"}]) == ["rag"]


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed_ms", [0, 4])
async def test_failed_first_model_response_consumes_first_use_label(elapsed_ms):
    live = module("day9_live")
    request_ids = ["b" * 32, "c" * 32]

    class Agent:
        port = 8019

        def events(self):
            return [event for request_id in request_ids for event in (
                {"kind": "outbound", "phase": "start", "category": "model", "request_id": request_id},
                {"kind": "outbound", "phase": "complete", "category": "model", "request_id": request_id,
                 "forwarded": True, "protocol": "invalid", "http_status": 503, "elapsed_ms": elapsed_ms,
                 "usage": None},
            )]

    responses = iter(request_ids)

    def fail_model(request):
        return httpx.Response(503, headers={"X-Request-ID": next(responses)},
                              json={"error": {"code": "model_timeout"}}, request=request)

    runner = live.LiveRunner(Agent(), authorization="Bearer unit-test-token", secrets=("unit-test-token",))
    step = {"message": "query", "sse": False, "expect": {"status": "answered", "tool": "none"}}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8019", transport=httpx.MockTransport(fail_model),
                                 trust_env=False) as client:
        first, _, _ = await runner.execute(step, None, [], client, None)
        second, _, _ = await runner.execute(step, None, [], client, None)
    assert first["first_use_labels"] == ["model"]
    assert first["model_elapsed_ms"] == elapsed_ms
    assert first["usage"] is None
    assert second["first_use_labels"] == []


def test_product_truth_rejects_sku_price_stock_and_image_mismatch():
    day = module("day9_cases")
    product = {"product_id": 36, "name": "test", "description": "", "image": "image",
               "skus": [{"unique_id": "sku", "name": "A4", "price": "1.00", "stock": 5, "image": "sku-image"}]}
    assert day.product_checks([product], {36: product}, {"product_ids": [36]}) == []
    for field, value in (("price", "2.00"), ("stock", 4), ("image", "other")):
        changed = json.loads(json.dumps(product))
        changed["skus"][0][field] = value
        assert "product_truth_mismatch" in day.product_checks([changed], {36: product}, {"product_ids": [36]})


def test_policy_checks_require_original_metadata_text_and_expected_ids():
    from app.rag.documents import PolicyChunk
    day = module("day9_cases")
    chunk = PolicyChunk(document_id="01-demo-scope", title="demo", section="scope",
                        source="knowledge/policies/demo.md", version="v1", kind="demo",
                        chunk_id="chunk", text="original")
    source = {**chunk.metadata, "text": chunk.text, "score": 0.9}
    expect = {"document_ids": ["01-demo-scope"]}
    result = {"status": "answered", "sources": [source], "answer": "original", "notice": "demo"}
    assert day.policy_checks(result, {"chunk": chunk}, expect)["failures"] == []
    source["version"] = "altered"
    assert "citation_fidelity_failed" in day.policy_checks(result, {"chunk": chunk}, expect)["failures"]
    result["sources"] = []
    checked = day.policy_checks(result, {"chunk": chunk}, expect)
    assert checked["fidelity"] is None
    assert "expected_documents_missing" in checked["failures"]


def test_tool_arguments_use_observed_queries_not_declarations():
    day = module("day9_cases")
    expect = {"tool": "search_products", "query": {"keyword": "test", "max_price": "3", "limit": "3", "page": "1"}}
    observed = [{"category": "shop_read", "path": "agent/products", "query": dict(expect["query"]), "forwarded": True}]
    assert day.tool_checks(expect, observed, []) == []
    observed[0]["query"]["max_price"] = "4"
    assert day.tool_checks(expect, observed, []) == ["tool_arguments_mismatch"]
    assert day.tool_checks(expect, [], []) == ["missing_tool_observation"]


def frame(kind, value):
    return ("event: " + kind + "\ndata: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode("utf-8")


def test_sse_incremental_utf8_and_public_event_consistency():
    stream = module("day9_live").SSEReader()
    done = {"answer": "\u4f60\u597d", "products": [], "sources": [], "notice": None,
            "pending_action": None}
    raw = (frame("status", {"status": "processing"}) + frame("text", {"text": done["answer"]})
           + frame("products", {"products": []}) + frame("sources", {"sources": [], "notice": None})
           + frame("done", done))
    for byte in raw:
        stream.feed(bytes([byte]), 12)
    assert stream.finish() == done
    assert stream.status_first_ms == stream.business_text_first_ms == stream.terminal_ms == 12


@pytest.mark.parametrize("raw", [
    b"event: done\ndata: {}\n\n",
    b"event: unknown\ndata: {}\n\n",
    b"event: status\ndata: broken\n\n",
    b"event: status\ndata: {}\n",
    b"\xff",
    b"x" * 262145,
], ids=["missing", "unknown", "json", "partial", "utf8", "oversized"])
def test_sse_rejects_malformed_missing_or_unbounded_frames(raw):
    stream = module("day9_live").SSEReader()
    with pytest.raises(ValueError):
        stream.feed(raw, 1)
        stream.finish()


def test_sse_duplicate_terminal_and_mismatch_are_rejected():
    live = module("day9_live")
    for raw in (frame("error", {"error": {"code": "model_timeout"}}) * 2,
                frame("text", {"text": "different"}) + frame("products", {"products": []})
                + frame("sources", {"sources": [], "notice": None})
                + frame("done", {"answer": "answer", "products": [], "sources": [], "notice": None, "pending_action": None})):
        reader = live.SSEReader()
        with pytest.raises(ValueError):
            reader.feed(raw, 1)
            reader.finish()


@pytest.mark.asyncio
async def test_observer_blocks_every_shop_write_and_unexpected_destination_before_forward():
    observer = module("day9_observer")
    events, forwarded = [], []
    async def send(client, request, **kwargs):
        forwarded.append(request)
        return httpx.Response(200, json={}, request=request)
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/v1/chat/completions", credential="private-key",
        record=events.append, run_id="a" * 32)
    async with httpx.AsyncClient(trust_env=False) as client:
        for url, method in (("http://127.0.0.1:8008/api/cart/add", "POST"),
                            ("https://unexpected.example/", "GET"),
                            ("http://127.0.0.1:8008/api/unknown", "GET"),
                            ("https://model.example/v1/chat/completions?extra=1", "POST")):
            with pytest.raises(httpx.RequestError):
                await guard.send(send, client, httpx.Request(method, url))
    assert not forwarded
    assert len(events) == 4 and all(e["forwarded"] is False for e in events)
    assert "private-key" not in json.dumps(events)


@pytest.mark.asyncio
async def test_observer_returns_same_response_and_strict_protocol_usage_without_raw_text(monkeypatch):
    observer = module("day9_observer")
    monkeypatch.setattr(observer, "request_id", lambda: "b" * 32)
    events = []
    raw = {"choices": [{"message": {"role": "assistant", "content": "never-log-model-body",
                                   "reasoning_content": "never-log-reasoning"}}],
           "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
    response = httpx.Response(200, json=raw)
    async def send(client, request, **kwargs):
        return response
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/v1/chat/completions", credential="private-key",
        record=events.append, run_id="a" * 32)
    async with httpx.AsyncClient(trust_env=False) as client:
        result = await guard.send(send, client, httpx.Request("POST", "https://model.example/v1/chat/completions"))
    assert result is response and response.json() == raw
    assert events[-1]["protocol"] == "valid"
    assert events[-1]["usage"]["input_tokens"] == 2
    assert "never-log" not in json.dumps(events)
    assert observer.counts(events)["external_model_verified"] is True
    assert observer.counts([])["external_model_verified"] is False


@pytest.mark.asyncio
async def test_observer_recording_failure_stops_forwarding():
    observer = module("day9_observer")
    async def send(*args, **kwargs):
        pytest.fail("forwarded after observation storage failed")
    def record(event):
        raise OSError("private path")
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/v1/chat/completions", credential="", record=record, run_id="a" * 32)
    with pytest.raises(httpx.RequestError):
        await guard.send(send, None, httpx.Request("GET", "http://127.0.0.1:8008/api/userinfo"))


@pytest.mark.asyncio
async def test_local_rag_observer_preserves_result_exception_and_restores_wrapper(monkeypatch):
    observer = module("day9_observer")
    from app.rag.service import PolicyService
    from starlette.concurrency import run_in_threadpool

    marker = object()
    raised = RuntimeError("private-rag-error")

    def original(service, question):
        if question == "raise-private-question":
            raise raised
        return marker

    monkeypatch.setattr(PolicyService, "answer", original)
    monkeypatch.setattr(observer, "request_id", lambda: "b" * 32)
    events = []
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/chat/completions", credential="secret",
        record=events.append, run_id="a" * 32)
    service = object.__new__(PolicyService)
    with guard.installed():
        assert await run_in_threadpool(service.answer, "private-question") is marker
        with pytest.raises(RuntimeError) as caught:
            await run_in_threadpool(service.answer, "raise-private-question")
        assert caught.value is raised
    assert PolicyService.answer is original
    assert [(event["phase"], event.get("status")) for event in events] == [
        ("start", None), ("complete", "succeeded"),
        ("start", None), ("complete", "failed")]
    assert all(event["category"] == "rag" and event["request_id"] == "b" * 32 for event in events)
    assert "private-question" not in json.dumps(events)
    assert "private-rag-error" not in json.dumps(events)


@pytest.mark.asyncio
async def test_local_rag_observer_is_fail_closed_before_invocation(monkeypatch):
    observer = module("day9_observer")
    from app.rag.service import PolicyService
    from starlette.concurrency import run_in_threadpool

    called = []

    def original(service, question):
        called.append(question)

    monkeypatch.setattr(PolicyService, "answer", original)
    monkeypatch.setattr(observer, "request_id", lambda: "b" * 32)
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/chat/completions", credential="secret",
        record=lambda event: (_ for _ in ()).throw(OSError("private-path")), run_id="a" * 32)
    service = object.__new__(PolicyService)
    with guard.installed():
        with pytest.raises(RuntimeError, match="day9_observation_failed"):
            await run_in_threadpool(service.answer, "private-question")
    assert called == []


@pytest.mark.asyncio
async def test_first_local_rag_use_survives_later_model_http_failure(monkeypatch):
    observer = module("day9_observer")
    live = module("day9_live")
    import app.telemetry as telemetry
    from app.rag.service import PolicyService
    from starlette.concurrency import run_in_threadpool
    from types import SimpleNamespace

    request_ids = iter(("b" * 32, "c" * 32))
    events = []
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/chat/completions", credential="secret",
        record=events.append, run_id="a" * 32)
    service = object.__new__(PolicyService)
    service.settings = SimpleNamespace(top_k=4, evidence_min_score=.5)
    service._store = SimpleNamespace(embeddings=None)
    service.search_policy = lambda question, top_k=4: []

    async def observe_failed_request(request_id):
        token = telemetry._REQUEST_ID.set(request_id)
        try:
            with guard.installed():
                answer = await run_in_threadpool(service.answer, "配送范围是什么？")
                assert answer.status == "no_evidence"
            async def fail_model(client, request, **kwargs):
                return httpx.Response(503, json={"error": {"code": "model_timeout"}}, request=request)
            await guard.send(fail_model, None, httpx.Request("POST", "https://model.example/chat/completions"))
        finally:
            telemetry._REQUEST_ID.reset(token)

    await observe_failed_request("b" * 32)
    await observe_failed_request("c" * 32)

    class Agent:
        port = 8019

        def events(self):
            return events

    def fail_public_request(request):
        return httpx.Response(503, headers={"X-Request-ID": next(request_ids)},
                              json={"error": {"code": "model_timeout"}}, request=request)

    runner = live.LiveRunner(Agent(), authorization="Bearer unit-test-token", secrets=("unit-test-token",))
    step = {"message": "policy query", "sse": False, "expect": {"status": "answered", "tool": "search_policy"}}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8019",
                                 transport=httpx.MockTransport(fail_public_request), trust_env=False) as client:
        first, _, _ = await runner.execute(step, None, [], client, None)
        second, _, _ = await runner.execute(step, None, [], client, None)
    assert first["first_use_labels"] == ["model", "rag"]
    assert second["first_use_labels"] == []
    rows = [
        {"id": "first", "group": "policy", "mode": "live", "status": "failed", "steps": [first]},
        {"id": "second", "group": "policy", "mode": "live", "status": "failed", "steps": [second]},
    ]
    assert module("day9_cases").summarize(rows)["first_use"]["rag"] == {
        "case_id": "first", "step_index": 0}
    assert "配送范围是什么" not in json.dumps(events, ensure_ascii=False)


def test_report_reservation_never_overwrites_or_accepts_outside_protected_paths():
    evaluator = module("eval_day9")
    from uuid import uuid4
    path = ROOT / ".tmp" / ("day9-report-test-" + uuid4().hex) / "report.json"
    owner = evaluator.OwnedReport(path, environ={}, env_file=ROOT / ".env")
    owner.reserve({"status": "pending"})
    with pytest.raises((ValueError, OSError)):
        evaluator.OwnedReport(path, environ={}, env_file=ROOT / ".env").reserve({"status": "other"})
    assert json.loads(path.read_text()) == {"status": "pending"}
    for unsafe in (ROOT / ".env", ROOT / "knowledge/report.json", ROOT.parent / "report.json",
                   ROOT / "data/chroma/report.json"):
        with pytest.raises((ValueError, OSError)):
            evaluator.OwnedReport(unsafe, environ={}, env_file=ROOT / ".env")
    owner.close()


def test_occupied_port_is_rejected_without_stopping_listener():
    live = module("day9_live")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(ValueError):
            live.free_port(listener.getsockname()[1])
        assert listener.fileno() != -1


@pytest.mark.asyncio
async def test_dependent_turns_stop_but_next_case_can_run():
    live = module("day9_live")
    requests = []
    async def execute(step, sid, previous):
        requests.append(step["message"])
        return {"status": "failed", "failures": ["fixed_failure"]}, None, []
    case = {"id": "multi-test", "group": "multi_turn", "mode": "live",
            "steps": [{"message": "first"}, {"message": "second"}]}
    row = await live.run_case(case, execute)
    assert requests == ["first"]
    assert row["steps"][1]["status"] == "not_run"
    assert row["status"] == "failed"
    await live.run_case({"id": "next", "group": "product", "mode": "live", "steps": [{"message": "next"}]}, execute)
    assert requests == ["first", "next"]


@pytest.mark.asyncio
async def test_interrupted_multi_turn_case_never_persists_as_passed():
    live = module("day9_live")
    saved = []
    calls = 0

    async def execute(step, sid, previous):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.CancelledError
        return {"status": "passed"}, "session", []

    case = {"id": "multi-interrupted", "group": "multi_turn", "mode": "live",
            "steps": [{"message": "first"}, {"message": "second"}]}
    with pytest.raises(asyncio.CancelledError):
        await live.run_case(case, execute, lambda row: saved.append(json.loads(json.dumps(row))))

    assert saved[0]["status"] == "not_run"
    assert saved[-1]["status"] == "failed"
    assert saved[-1]["interrupted"] is True
    assert saved[-1]["steps"] == [{"status": "passed"}, {"status": "not_run"}]
    assert module("day9_cases").summarize([saved[-1]])["case_success_rate"] == 0


def test_deterministic_worker_missing_network_or_exit_failure_cannot_pass(tmp_path):
    evaluator = module("eval_day9")
    cases = [{"id": "c", "group": "confirmation", "mode": "deterministic",
              "selector": "tests/test_actions.py::test_action_modules_exist"}]
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite><testcase classname="tests.test_actions" name="test_action_modules_exist" time="0"/></testsuite>')
    network = tmp_path / "network.json"
    assert evaluator.deterministic_results(cases, junit, network, 0)["gate_passed"] is False
    network.write_text(json.dumps({"verified": True, "connection_attempts": 0, "local_socketpair_connections": 1}))
    assert evaluator.deterministic_results(cases, junit, network, 1)["gate_passed"] is False
    assert evaluator.deterministic_results(cases, junit, network, 0)["gate_passed"] is True


def test_partial_junit_excludes_absent_selector_timing_but_keeps_measured_zero(tmp_path):
    evaluator = module("eval_day9")
    cases = [
        {"id": "present", "group": "confirmation", "mode": "deterministic",
         "selector": "tests/test_actions.py::test_present"},
        {"id": "absent", "group": "confirmation", "mode": "deterministic",
         "selector": "tests/test_actions.py::test_absent"},
    ]
    junit = tmp_path / "partial.xml"
    junit.write_text('<testsuite><testcase classname="tests.test_actions" name="test_present" time="0"/>'
                     '</testsuite>')
    network = tmp_path / "network.json"
    network.write_text(json.dumps({"verified": True, "connection_attempts": 0,
                                   "local_socketpair_connections": 0}))
    result = evaluator.deterministic_results(cases, junit, network, 0)
    assert [(row["status"], row["test_count"], row["elapsed_ms"]) for row in result["rows"]] == [
        ("passed", 1, 0), ("not_run", 0, None)]
    assert module("day9_cases").summarize(result["rows"])["timings_by_mode"]["deterministic"] == {
        "case_elapsed_ms": {"count": 1, "median": 0, "p95": 0, "max": 0}}


def test_parameterized_selector_skip_is_not_passed(tmp_path):
    evaluator = module("eval_day9")
    cases = [{"id": "c", "group": "confirmation", "mode": "deterministic",
              "selector": "tests/test_actions.py::test_action_modules_exist"}]
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite><testcase classname="tests.test_actions" name="test_action_modules_exist[a]" time="0"/>'
                     '<testcase classname="tests.test_actions" name="test_action_modules_exist[b]" time="0"><skipped/></testcase></testsuite>')
    network = tmp_path / "network.json"
    network.write_text(json.dumps({"verified": True, "connection_attempts": 0, "local_socketpair_connections": 0}))
    result = evaluator.deterministic_results(cases, junit, network, 0)
    assert result["rows"][0]["status"] == "skipped" and result["gate_passed"] is False


def test_manifest_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path):
    day = module("day9_cases")
    for content in ('{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}'):
        path = tmp_path / "manifest.json"
        path.write_text(content)
        with pytest.raises(ValueError):
            day.load_cases(path)


def test_manifest_rejects_unhashable_expectation_with_fixed_value_error(tmp_path):
    day = module("day9_cases")
    value = manifest()
    value["cases"][0]["steps"][0]["expect"]["tool"] = []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        day.load_cases(path)


@pytest.mark.asyncio
async def test_model_invalid_protocol_and_missing_usage_remain_visible(monkeypatch):
    observer = module("day9_observer")
    monkeypatch.setattr(observer, "request_id", lambda: "b" * 32)
    events = []
    replies = [httpx.Response(200, json={"choices": []}), httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "ok"}}]})]
    async def send(*args, **kwargs):
        return replies.pop(0)
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/chat/completions", credential="secret", record=events.append, run_id="a" * 32)
    for _ in range(2):
        await guard.send(send, None, httpx.Request("POST", "https://model.example/chat/completions"))
    assert observer.counts(events)["model_protocol_invalid"] == 1
    assert observer.counts(events)["external_model_verified"] is False
    assert events[-1]["usage"] is None


@pytest.mark.asyncio
async def test_observer_rejects_authorization_echo_in_query_without_persisting_it():
    observer = module("day9_observer")
    events = []
    async def send(*args, **kwargs):
        pytest.fail("private query must not be forwarded")
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.example/chat/completions", credential="model-secret", record=events.append, run_id="a" * 32)
    request = httpx.Request("GET", "http://127.0.0.1:8008/api/agent/products",
                           params={"keyword": "shop-private-token"}, headers={"Authorization": "Bearer shop-private-token"})
    with pytest.raises(httpx.RequestError):
        await guard.send(send, None, request)
    assert "shop-private-token" not in json.dumps(events)


def test_report_save_cannot_follow_replaced_symlink_or_hardlink(tmp_path):
    from uuid import uuid4
    evaluator = module("eval_day9")
    path = ROOT / ".tmp" / ("day9-link-test-" + uuid4().hex) / "report.json"
    owner = evaluator.OwnedReport(path, environ={}, env_file=ROOT / ".env")
    owner.reserve({"status": "pending"})
    try:
        alias = path.with_name("alias.json")
        import os
        os.link(path, alias)
        with pytest.raises(ValueError):
            owner.save({"status": "passed"})
        assert json.loads(path.read_text()) == {"status": "pending"}
    finally:
        owner.close()


def test_report_constructor_rejects_linked_parent(tmp_path):
    from uuid import uuid4
    evaluator = module("eval_day9")
    root = ROOT / ".tmp" / ("day9-parent-link-" + uuid4().hex)
    root.mkdir()
    linked = root / "link"
    try:
        linked.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Windows user cannot create symbolic links")
    with pytest.raises(ValueError):
        evaluator.OwnedReport(linked / "report.json", environ={}, env_file=ROOT / ".env")


@pytest.mark.asyncio
async def test_owned_process_early_exit_never_reads_foreign_health(monkeypatch):
    live = module("day9_live")
    from uuid import uuid4
    run_id = uuid4().hex
    owned = live.OwnedAgent(ROOT / ".tmp" / ("day9-live-" + run_id), port=8019, run_id=run_id, environ={})
    class Process:
        def poll(self):
            return 1
    owned.process = Process()
    async with httpx.AsyncClient(trust_env=False, transport=httpx.MockTransport(
            lambda request: pytest.fail("must not probe another service after child exit"))) as client:
        with pytest.raises(ValueError, match="owned_service_exited"):
            await owned.ready(client)


def test_owned_process_stop_awaits_exact_process_even_if_already_exited():
    live = module("day9_live")
    from uuid import uuid4
    run_id = uuid4().hex
    owned = live.OwnedAgent(ROOT / ".tmp" / ("day9-live-" + run_id), port=8019, run_id=run_id, environ={})
    calls = []
    class Process:
        def poll(self):
            return 1
        def wait(self, timeout):
            calls.append(timeout)
            return 1
    owned.process = Process()
    outcome = owned.stop()
    assert calls == [20]
    assert outcome == {"pid": None, "exit_code": 1, "awaited": True,
                       "termination_requested": False, "exited_before_shutdown": True,
                       "unexpected_exit": True}


def test_owner_requested_termination_is_not_an_unexpected_exit():
    live = module("day9_live")
    from uuid import uuid4
    run_id = uuid4().hex
    owned = live.OwnedAgent(ROOT / ".tmp" / ("day9-live-" + run_id), port=8019, run_id=run_id, environ={})

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 1

        def wait(self, timeout):
            return self.returncode

    owned.process = Process()
    outcome = owned.stop()
    assert outcome["exit_code"] == 1
    assert outcome["termination_requested"] is True
    assert outcome["exited_before_shutdown"] is False
    assert outcome["unexpected_exit"] is False


def test_unexpected_child_exit_fails_final_gate_with_valid_observations():
    evaluator = module("eval_day9")
    request_id = "b" * 32
    events = [
        {"kind": "outbound", "phase": "start", "category": "model", "request_id": request_id},
        {"kind": "outbound", "phase": "complete", "category": "model", "request_id": request_id,
         "forwarded": True, "protocol": "valid", "http_status": 200},
    ]

    class Agent:
        def stop(self):
            return {"pid": 123, "exit_code": 0, "awaited": True,
                    "termination_requested": False, "exited_before_shutdown": True,
                    "unexpected_exit": True}

        def events(self):
            return events

    report = {"status": "passed", "provenance": {"external_model_verified": False}}
    evaluator.finalize_live(report, Agent())
    assert report["observations"]["external_model_verified"] is True
    assert report["owned_process"]["unexpected_exit"] is True
    assert report["status"] == "failed"
    assert report["error_code"] == "owned_child_exited_unexpectedly"


def test_missing_observer_or_foreign_run_is_not_evidence(tmp_path):
    observer = module("day9_observer")
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        observer.read_events(path, "a" * 32)
    path.write_text(json.dumps({"kind": "startup", "run_id": "b" * 32, "pid": 1, "port": 8019,
                               "configured": True, "timeout_seconds": 60, "enable_thinking": False}) + "\n")
    with pytest.raises(ValueError):
        observer.read_events(path, "a" * 32)


@pytest.mark.asyncio
async def test_case_report_storage_failure_stops_independent_network_work():
    live = module("day9_live")
    async def execute(*args):
        return {"status": "passed"}, "session", []
    def save(row):
        raise OSError("cannot save")
    with pytest.raises(OSError):
        await live.run_case({"id": "test", "group": "multi_turn", "mode": "live",
                             "steps": [{}, {}]}, execute, save)


def test_unfinished_forwarded_model_start_prevents_attestation():
    observer = module("day9_observer")
    complete = {"kind": "outbound", "phase": "complete", "category": "model", "forwarded": True,
                "request_id": "b" * 32, "protocol": "valid", "http_status": 200}
    start = {**complete, "phase": "start", "forwarded": False}
    assert observer.counts([start, complete, start])["external_model_verified"] is False


def test_search_tool_checks_reject_extra_detail_tool_query():
    day = module("day9_cases")
    expect = {"tool": "search_products", "query": {"keyword": "test"}}
    events = [{"category": "shop_read", "path": "agent/products", "query": {"keyword": "test"}, "forwarded": True},
              {"category": "shop_read", "path": "product/detail", "product_id": 36, "forwarded": True}]
    assert "unexpected_product_tool" in day.tool_checks(expect, events, [])


@pytest.mark.asyncio
async def test_live_transport_failure_preserves_elapsed_and_step_metadata():
    live = module("day9_live")
    from uuid import uuid4
    run_id = uuid4().hex
    agent = live.OwnedAgent(ROOT / ".tmp" / ("day9-live-" + run_id), port=8019, run_id=run_id, environ={})
    runner = live.LiveRunner(agent, authorization="Bearer unit-test-token", secrets=("unit-test-token",))
    def fail(request):
        raise httpx.ConnectError("never-record-private-exception", request=request)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8019", transport=httpx.MockTransport(fail),
                                 trust_env=False) as client:
        result, sid, products = await runner.execute(
            {"message": "query", "sse": False, "expect": {"status": "answered", "tool": "none"}},
            None, [], client, None)
    assert result["status"] == "failed"
    assert result["client_elapsed_ms"] >= 0
    assert result["sse"] is False
    assert "never-record-private" not in json.dumps(result)


def test_empty_deterministic_evidence_does_not_claim_no_network(tmp_path):
    evaluator = module("eval_day9")
    result = evaluator.deterministic_results([], tmp_path / "missing.xml", tmp_path / "missing.json", 0)
    assert result["network"]["connection_attempts"] is None
    assert result["gate_passed"] is False


def test_detail_truth_requires_all_skus_not_a_valid_subset():
    day = module("day9_cases")
    product = {"product_id": 36, "name": "test", "description": "", "image": "",
               "skus": [{"unique_id": "a", "name": "A", "price": "1", "stock": 5, "image": ""},
                        {"unique_id": "b", "name": "B", "price": "3", "stock": 5, "image": ""}]}
    cropped = {**product, "skus": product["skus"][:1]}
    assert "product_truth_mismatch" in day.product_checks(
        [cropped], {36: product}, {"tool": "get_product_detail", "product_ids": [36]})


@pytest.mark.asyncio
async def test_all_shop_writes_blocked_even_if_model_origin_is_misconfigured():
    observer = module("day9_observer")
    events = []
    guard = observer.OutboundObserver(shop_url="http://127.0.0.1:8008/api/",
        model_url="http://127.0.0.1:8008/api/cart/add", credential="", record=events.append, run_id="a" * 32)
    async def send(*args, **kwargs):
        pytest.fail("shop origin writes always forbidden")
    with pytest.raises(httpx.RequestError):
        await guard.send(send, None, httpx.Request("POST", "http://127.0.0.1:8008/api/cart/add"))
    assert events[-1]["category"] == "shop_write_blocked"


def test_step_usage_is_null_when_any_model_response_omits_usage():
    live = module("day9_live")
    events = [{"phase": "complete", "category": "model", "usage": {"input_tokens": 2, "output_tokens": 3}},
              {"phase": "complete", "category": "model", "usage": None}]
    assert live.observed_usage(events) is None
    events[1]["usage"] = {"input_tokens": 5, "output_tokens": 7}
    assert live.observed_usage(events) == {"input_tokens": 7, "output_tokens": 10}


def test_child_startup_is_recorded_only_after_successful_loopback_binding(monkeypatch):
    from uuid import uuid4
    import uvicorn
    observer = module("day9_observer")
    run_id = uuid4().hex
    runtime = ROOT / ".tmp" / ("day9-live-" + run_id)
    runtime.mkdir()
    class Listener:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def setsockopt(self, *args):
            pass
        def bind(self, address):
            assert address == ("127.0.0.1", 8019)
            raise OSError("owned child occupied port")
    monkeypatch.setattr(observer.socket, "socket", lambda *args, **kwargs: Listener())
    def unavailable(*args, **kwargs):
        raise OSError("owned child occupied port")
    monkeypatch.setattr(uvicorn, "run", unavailable)
    assert observer.main(["--port", "8019", "--runtime", str(runtime), "--run-id", run_id]) == 1
    assert (runtime / "observations.jsonl").read_text() == ""
