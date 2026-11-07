"""Chat CLI contracts use an injected transport, never a live model or shop."""

import importlib
import importlib.util
import io
import json
from pathlib import Path

import httpx
import pytest

from app.rag.service import DEMO_NOTICE
from scripts import eval_policy


AUTH = "Bearer chat-cli-test-only-secret"
MODEL_KEY = "sk-chat-cli-test-only-model-secret"
MESSAGE = "预算3元，推荐华为手机"
ROOT = Path(__file__).resolve().parents[1]


def _module():
    name = "scripts.chat_smoke_test"
    assert importlib.util.find_spec(name) is not None, "missing required chat CLI"
    return importlib.import_module(name)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "service"
    root.mkdir()
    monkeypatch.setattr(eval_policy, "SERVICE_ROOT", root)
    return root


def _product():
    return {"product_id": 1, "name": "演示华为手机", "description": "公共商品详情", "image": "",
            "skus": [{"name": "默认", "unique_id": "sku-public", "price": "3.00", "stock": 7, "image": ""}]}


def _source():
    return {"document_id": "06-return-eligibility", "title": "演示退货", "section": "退货条件",
            "source": "knowledge/policies/06-return-eligibility.md", "version": "demo.v1", "kind": "demo",
            "chunk_id": "06-return-eligibility-0", "text": " 演示规则原文：符合条件可申请退货。\n", "score": 0.8}


def _body(**updates):
    body = {"answer": "请以商品卡的价格和库存为准。", "status": "answered", "products": [_product()],
            "sources": [], "notice": None,
            "tool_calls": [{"name": "search_products", "status": "succeeded", "elapsed_ms": 2}],
            "model_rounds": 2, "usage": {"input_tokens": 12, "output_tokens": 4}}
    body.update(updates)
    return body


def _run(project, handler, argv=None, *, environ=None, report_path=None):
    output = io.StringIO()
    destination = project / "data" / "chat-test.json" if report_path is None else report_path
    code = _module().main(["--message", MESSAGE] if argv is None else argv,
        environ={"YSHOP_AUTHORIZATION": AUTH, "LLM_API_KEY": MODEL_KEY} if environ is None else environ,
        env_file=project / "unused.env", transport=httpx.MockTransport(handler),
        report_path=destination, stdout=output)
    return code, json.loads(output.getvalue()), destination


def test_chat_cli_required_module_exists():
    _module()


def test_single_local_chat_post_and_public_report(project):
    requests = []
    body = _body()
    body["internal"] = {"authorization": AUTH, "model_key": MODEL_KEY}
    body["products"][0]["internal"] = AUTH
    body["products"][0]["skus"][0]["cost"] = MODEL_KEY
    body["tool_calls"][0]["arguments"] = MESSAGE
    body["usage"]["raw"] = AUTH

    def handler(request):
        requests.append(request)
        assert request.extensions["timeout"]["read"] >= 300
        return httpx.Response(200, json=body)

    code, output, path = _run(project, handler)
    assert code == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST" and request.url.host == "127.0.0.1"
    assert request.url.path == "/agent/chat" and not request.url.query
    assert json.loads(request.content) == {"message": MESSAGE}
    assert request.headers["authorization"] == AUTH
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["method"] == "POST" and report["endpoint"] == "/agent/chat"
    assert report["status"] == "passed" and report["mode"] == "test_double"
    assert report["provenance"]["external_model_verified"] is False
    assert report["provenance"]["injected_transport"] is True
    assert set(report["result"]) == set(_body())
    assert output == {"status": "passed", "chat_status": "answered", "model_rounds": 2,
                      "tools": ["search_products"], "product_count": 1, "source_count": 0}
    saved = path.read_text(encoding="utf-8")
    for secret in (AUTH, AUTH.split()[1], MODEL_KEY, MESSAGE, "internal", "cost", "arguments", "raw"):
        assert secret not in saved and secret not in json.dumps(output, ensure_ascii=False)


def test_http_client_ignores_proxy_env_and_disables_redirects(project, monkeypatch):
    module = _module()
    original = httpx.AsyncClient
    configurations = []

    def client(**kwargs):
        configurations.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client)
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted.example:99")
    assert _run(project, lambda request: httpx.Response(200, json=_body()))[0] == 0
    assert configurations[0]["trust_env"] is False
    assert configurations[0]["follow_redirects"] is False
    assert "proxy" not in configurations[0]


@pytest.mark.parametrize("argv", [[], ["--message", " "], ["--message", "x" * 1001],
    ["--message", "a", "--message=b"], ["--message", "a", "--api-url", "http://localhost", "--api-url=http://127.0.0.1"],
    ["--message", "a", "--report", "data/a.json", "--report=data/b.json"],
    ["--message", "a", "--mode", "test_double", "--mode=test_double"],
    ["--message", "a", "--mode", "unknown"], ["--token", AUTH], ["--mess", MESSAGE],
    ["--message", "a", "--unknown", MODEL_KEY], ["--message", "a", "--api-url"]])
def test_invalid_arguments_never_send_http(project, argv):
    code, output, _ = _run(project, lambda request: pytest.fail("invalid input sent HTTP"), argv)
    assert code == 1 and output == {"status": "failed", "error_code": "invalid_request"}


@pytest.mark.parametrize("url", ["https://127.0.0.1:8010", "http://example.com", "http://127.0.0.1.evil",
    "http://user:password@localhost:8010", "http://localhost:8010/agent/chat", "http://localhost:8010?x=1",
    "http://localhost:8010#frag", "http://localhost:0", "http://localhost:65536", "http://localhost:abc",
    "http://localhost:8010\\x", "http://localhost:8010\n", "http://localhost:8010\x00"])
def test_remote_or_decorated_url_refused_before_http(project, url):
    code, output, _ = _run(project, lambda request: pytest.fail("invalid origin sent HTTP"),
                           ["--message", MESSAGE, "--api-url", url])
    assert code == 1 and output["error_code"] == "invalid_request"


@pytest.mark.parametrize("url", ["http://localhost:8010/", "http://[::1]:8010", "http://127.0.0.1:8010"])
def test_loopback_origins_supported(project, url):
    requests = []
    code, _, _ = _run(project, lambda request: requests.append(request) or httpx.Response(200, json=_body()),
                      ["--message", MESSAGE, "--api-url", url])
    assert code == 0 and len(requests) == 1 and requests[0].url.path == "/agent/chat"


@pytest.mark.parametrize("authorization", [None, "", "Basic secret", AUTH + "\n", "Bearer two words"])
def test_missing_or_invalid_credential_never_sends_http(project, authorization):
    environ = {} if authorization is None else {"YSHOP_AUTHORIZATION": authorization}
    code, output, _ = _run(project, lambda request: pytest.fail("invalid credential sent HTTP"), environ=environ)
    assert code == 1 and output["error_code"] == "login_required"


@pytest.mark.parametrize("relative", [".env", "README.md", "evals/chat_cases.json", "knowledge/policies/demo.json",
    "data/chroma/report.json", ".cache/report.json", "data/fastembed/report.json", "data/manifest.json",
    "data/day1-http-checks.json", "data/day2-search-huawei.json", "data/day3-policy-eval-first.json",
    "data/day3-policy-eval-final-20261025.json", "data/day3-policy-smoke-final-20261025.json",
    "data/chat.txt", "../outside.json"])
def test_report_protection_precedes_http_and_preserves_evidence(project, relative):
    destination = project / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"frozen-evidence")
    code, output, _ = _run(project, lambda request: pytest.fail("protected report sent HTTP"),
                           ["--message", MESSAGE, "--report", str(destination)])
    assert code == 1 and output["error_code"] == "invalid_request"
    assert destination.read_bytes() == b"frozen-evidence"


@pytest.mark.parametrize("filename", ["day4-preflight.json", "day4-model-protocol-corrected-01.json",
                                     "day4-chat-smoke.json"])
def test_existing_day4_report_is_preserved_before_http(project, filename):
    destination = project / "data" / filename
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"frozen-day4-evidence")
    code, output, _ = _run(project, lambda request: pytest.fail("existing day-four report sent HTTP"),
                           ["--message", MESSAGE, "--report", str(destination)])
    assert code == 1 and output["error_code"] == "invalid_request"
    assert destination.read_bytes() == b"frozen-day4-evidence"


def test_new_day4_report_can_finish_its_pending_write(project):
    requests = []
    destination = project / "data" / "day4-chat-smoke.json"
    code, output, _ = _run(project,
        lambda request: requests.append(request) or httpx.Response(200, json=_body()),
        report_path=destination)
    assert code == 0 and output["status"] == "passed" and len(requests) == 1
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["status"] == "passed" and saved["result"] == _body()


@pytest.mark.parametrize("field", ["RAG_CHROMA_DIR", "RAG_KNOWLEDGE_DIR", "RAG_MODEL_CACHE_DIR"])
def test_configured_data_directory_is_protected(project, field):
    protected = project / "data" / "custom-private"
    protected.mkdir(parents=True)
    destination = protected / "chat.json"
    destination.write_bytes(b"configured-data")
    code, output, _ = _run(project, lambda request: pytest.fail("protected configured data sent HTTP"),
        ["--message", MESSAGE, "--report", str(destination)],
        environ={"YSHOP_AUTHORIZATION": AUTH, field: str(protected)})
    assert code == 1 and output["error_code"] == "invalid_request"
    assert destination.read_bytes() == b"configured-data"


def test_report_cannot_replace_directory(project):
    destination = project / "data" / "blocked.json"
    destination.mkdir(parents=True)
    code, output, _ = _run(project, lambda request: pytest.fail("report failure sent HTTP"), report_path=destination)
    assert code == 1 and output["error_code"] == "report_write_failed"


def test_final_report_write_failure_is_nonzero(project, monkeypatch):
    module = _module()
    save = module._save_report
    writes = []

    def fail_final(path, report, authorization):
        writes.append(report["status"])
        if report["status"] != "pending":
            raise OSError("private-path " + AUTH)
        save(path, report, authorization)

    monkeypatch.setattr(module, "_save_report", fail_final)
    code, output, _ = _run(project, lambda request: httpx.Response(200, json=_body()))
    assert code == 1 and output["error_code"] == "report_write_failed"
    assert len(writes) == 2 and AUTH not in json.dumps(output)


@pytest.mark.parametrize("status,error_code", [(401, "login_required"), (503, "model_not_configured"),
    (504, "model_timeout"), (502, "model_invalid_response"), (503, "rag_not_built"),
    (302, "chat_http_error"), (500, "chat_http_error")])
def test_http_failure_safe_code_no_redirect_or_retry(project, status, error_code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": error_code, "message": AUTH}},
                              headers={"Location": "https://untrusted.example"})

    code, output, path = _run(project, handler)
    assert code == 1 and output["error_code"] == error_code and len(requests) == 1
    assert AUTH not in path.read_text(encoding="utf-8")
    assert "result" not in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("error", [httpx.ReadTimeout("secret " + AUTH), httpx.ConnectError(MODEL_KEY)])
def test_network_failure_is_sanitized_and_not_retried(project, error):
    requests = []

    def handler(request):
        requests.append(request)
        raise error

    code, output, path = _run(project, handler)
    assert code == 1 and output["error_code"] == "chat_http_error" and len(requests) == 1
    assert AUTH not in path.read_text(encoding="utf-8") and MODEL_KEY not in json.dumps(output)


@pytest.mark.parametrize("change", [{"model_rounds": True}, {"model_rounds": 0}, {"model_rounds": 6},
    {"products": [_product()] * 4}, {"sources": [{}]}, {"products": ["invalid"]},
    {"tool_calls": [{"name": "add_to_cart", "status": "succeeded", "elapsed_ms": 1}]},
    {"tool_calls": [{"name": "unknown_tool", "status": "succeeded", "elapsed_ms": 1}]},
    {"tool_calls": [{"name": "search_products", "status": "succeeded", "elapsed_ms": -1}]},
    {"usage": {"input_tokens": True, "output_tokens": 0}}, {"status": "invented"}, {"answer": ""},
    {"status": "limit_reached", "model_rounds": 2},
    {"tool_calls": [{"name": "search_products", "status": "succeeded", "elapsed_ms": 1}] * 4}])
def test_invalid_public_response_is_refused(project, change):
    code, output, path = _run(project, lambda request: httpx.Response(200, json=_body(**change)))
    assert code == 1 and output["error_code"] == "chat_invalid_response"
    assert "result" not in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("field", ["products", "sources", "notice", "tool_calls", "usage"])
def test_success_must_contain_complete_public_shape(project, field):
    body = _body()
    del body[field]
    assert _run(project, lambda request: httpx.Response(200, json=body))[1]["error_code"] == "chat_invalid_response"


@pytest.mark.parametrize("target,field,value", [("product", "product_id", "1"), ("product", "name", " "),
    ("sku", "stock", True), ("sku", "price", "NaN"), ("sku", "price", "-1"),
    ("sku", "price", True), ("sku", "name", 9), ("sku", "image", 9),
    ("source", "score", True), ("source", "score", "0.8"), ("source", "text", " "),
    ("source", "document_id", ""), ("source", "kind", "official")])
def test_nested_public_types_and_metadata_are_validated(project, target, field, value):
    body = _body()
    if target == "product":
        body["products"][0][field] = value
    elif target == "sku":
        body["products"][0]["skus"][0][field] = value
    else:
        body.update(products=[], sources=[_source()], answer=_source()["text"], notice=DEMO_NOTICE,
            tool_calls=[{"name": "search_policy", "status": "succeeded", "elapsed_ms": 1}])
        body["sources"][0][field] = value
    assert _run(project, lambda request: httpx.Response(200, json=body))[1]["error_code"] == "chat_invalid_response"


@pytest.mark.parametrize("target", ["answer", "product", "sku", "source", "tool"])
@pytest.mark.parametrize("secret", [AUTH, AUTH.split()[1], MODEL_KEY])
def test_public_field_credential_echo_is_refused_not_accepted_as_redacted_success(project, target, secret):
    body = _body()
    if target == "answer":
        body["answer"] = secret
    elif target == "product":
        body["products"][0]["description"] = secret
    elif target == "sku":
        body["products"][0]["skus"][0]["image"] = secret
    elif target == "source":
        body["sources"] = [_source()]
        body["sources"][0]["text"] = secret
        body["answer"] = secret
        body["notice"] = DEMO_NOTICE
        body["tool_calls"].append({"name": "search_policy", "status": "succeeded", "elapsed_ms": 1})
    else:
        body["tool_calls"][0]["name"] = secret
    code, output, path = _run(project, lambda request: httpx.Response(200, json=body))
    assert code == 1 and output["error_code"] == "chat_invalid_response"
    assert secret not in path.read_text(encoding="utf-8") and secret not in json.dumps(output)


def test_policy_original_source_preserved_and_extras_cropped(project):
    source = _source()
    source["internal"] = MODEL_KEY
    body = _body(products=[], sources=[source], answer="以下为演示规则原文：\n" + source["text"], notice=DEMO_NOTICE,
        tool_calls=[{"name": "search_policy", "status": "succeeded", "elapsed_ms": 1}])
    code, output, path = _run(project, lambda request: httpx.Response(200, json=body))
    assert code == 0 and output["source_count"] == 1
    saved = json.loads(path.read_text(encoding="utf-8"))["result"]
    assert saved["sources"][0]["text"] == _source()["text"]
    assert "internal" not in saved["sources"][0]


@pytest.mark.parametrize("change", [{"notice": None}, {"notice": "全部退款保证"},
    {"answer": "没有任何来源原文"}, {"status": "no_evidence"},
    {"tool_calls": [{"name": "search_products", "status": "succeeded", "elapsed_ms": 1}]}])
def test_rule_evidence_contract_rejects_unsupported_sources(project, change):
    body = _body(products=[], sources=[_source()], answer=_source()["text"], notice=DEMO_NOTICE,
        tool_calls=[{"name": "search_policy", "status": "succeeded", "elapsed_ms": 1}])
    body.update(change)
    assert _run(project, lambda request: httpx.Response(200, json=body))[1]["error_code"] == "chat_invalid_response"


def test_missing_evidence_requires_explicit_service_refusal(project):
    body = _body(status="no_evidence", products=[], notice=DEMO_NOTICE, answer="保证无条件永久退款。",
        tool_calls=[{"name": "search_policy", "status": "succeeded", "elapsed_ms": 1}])
    assert _run(project, lambda request: httpx.Response(200, json=body))[1]["error_code"] == "chat_invalid_response"


@pytest.mark.parametrize("status,rounds,traces,notice", [
    ("needs_input", 1, [], None),
    ("needs_input", 2, [{"name": "unknown_tool", "status": "failed", "elapsed_ms": 1}], None),
    ("no_evidence", 2, [{"name": "search_policy", "status": "succeeded", "elapsed_ms": 1}], DEMO_NOTICE),
    ("limit_reached", 5, [{"name": "search_products", "status": "failed", "elapsed_ms": 1}] * 12, None)])
def test_public_clarification_missing_evidence_and_round_boundary_are_valid(project, status, rounds, traces, notice):
    body = _body(status=status, model_rounds=rounds, products=[], tool_calls=traces, notice=notice,
        answer="演示资料中未找到足够依据，请通过商城已有客服入口咨询。" if status == "no_evidence" else "请调整查询条件。")
    code, output, _ = _run(project, lambda request: httpx.Response(200, json=body))
    assert code == 0 and output["chat_status"] == status


def test_injected_transport_cannot_claim_real_model(project):
    code, output, _ = _run(project, lambda request: pytest.fail("false real-model claim sent HTTP"),
                           ["--message", MESSAGE, "--mode", "real_model"])
    assert code == 1 and output["error_code"] == "invalid_request"


def test_uninjected_mode_is_runner_declaration_without_server_attestation(project, monkeypatch):
    module = _module()
    original = httpx.AsyncClient

    def client(**kwargs):
        assert kwargs["transport"] is None
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, json=_body()))
        return original(**kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client)
    destination = project / "data" / "declared.json"
    assert module.main(["--message", MESSAGE, "--mode", "real_model"], environ={"YSHOP_AUTHORIZATION": AUTH},
        env_file=project / "unused.env", report_path=destination, stdout=io.StringIO()) == 0
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["mode"] == "real_model"
    assert saved["provenance"] == {"declared_mode": "real_model", "declaration_source": "runner",
                                   "external_model_verified": False, "injected_transport": False}


def test_frozen_cases_cover_real_acceptance_and_deterministic_boundaries():
    path = ROOT / "evals" / "chat_cases.json"
    assert path.is_file(), "missing frozen day-four query cases"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["scope"] == "single_request_read_only_query"
    assert data["live_acceptance_status"] in {"pending", "passed"}
    cases = data["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["kind"] for case in cases} >= {"product_budget", "product_detail", "policy", "absent_policy",
        "clarification", "mixed_tools", "injection", "failure", "round_boundary"}
    assert any(case["verification"] == "real_model_acceptance" for case in cases)
    assert any(case["verification"] == "deterministic_test_double" for case in cases)
    for case in cases:
        assert isinstance(case["message"], str) and 1 <= len(case["message"]) <= 1000
        assert case["expected"]["business_write_requests"] == 0
        assert case["expected"]["max_model_rounds"] == 5
        assert case["expected"]["max_products"] == 3
        assert case["verification"] in {"real_model_acceptance", "deterministic_test_double"}
    deterministic = [case for case in cases if case["verification"] == "deterministic_test_double"]
    assert all(case.get("test_selector", "").startswith("tests/") for case in deterministic)
