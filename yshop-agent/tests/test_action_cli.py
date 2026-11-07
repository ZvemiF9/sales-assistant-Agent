"""Day 6 CLI and verifier boundaries, using only injected local HTTP."""

import asyncio
from copy import deepcopy
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from scripts import eval_policy
from tests.test_chat_cli import AUTH, MODEL_KEY


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SID = "b" * 32
AID = "a" * 32
SECOND_AID = "c" * 32
SKU = "c38b4340485b465d92b787dd68dcc26e"
QUERY = "预算3元，查询华为商品，请使用商品查询工具核实实时数据。"
PREPARE = "选择第一个商品的A4,白色规格，加入一件购物车。"


def runner():
    name = "scripts.action_smoke_test"
    assert importlib.util.find_spec(name) is not None, "Day 6 action CLI is missing"
    return importlib.import_module(name)


def verifier():
    name = "scripts.day6_verify"
    assert importlib.util.find_spec(name) is not None, "Day 6 workflow verifier is missing"
    return importlib.import_module(name)


def observer():
    path = ROOT / "scripts" / "day6_observer.py"
    assert path.is_file(), "Day 6 metadata-only observer is missing"
    spec = importlib.util.spec_from_file_location("day6_observer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "service"
    root.mkdir()
    monkeypatch.setattr(eval_policy, "SERVICE_ROOT", root)
    return root


def card(aid=AID, **changes):
    value = {
        "kind": "add_to_cart", "action_id": aid, "session_id": SID,
        "product_id": 36, "product_name": "华为商品", "unique_id": SKU,
        "sku_name": "A4,白色", "quantity": 1, "unit_price": "1.00",
        "total_price": "1.00", "image": "", "expires_at": time.time() + 600.0,
    }
    value.update(changes)
    return value


def action_body(status="awaiting_confirmation", aid=AID, *, pending=None, **changes):
    value = {
        "session_id": SID, "status": status, "answer": "请确认本次动作。",
        "pending_action": (card(aid) if pending is None else pending)
        if status == "awaiting_confirmation" else None,
        "action_result": None if status in {"awaiting_confirmation", "needs_input"} else {
            "action_id": aid, "session_id": SID, "status": status,
            "cart_id": 41 if status == "succeeded" else None,
            "error_code": "cart_result_unknown" if status == "unknown" else
            "insufficient_stock" if status == "failed" else None,
            "cart_path": "/pages/shop/ShoppingCart/index",
        },
    }
    value.update(changes)
    return value


def chat_body(**changes):
    value = {
        **action_body(), "products": [], "sources": [], "notice": None,
        "tool_calls": [], "model_rounds": 0,
        "usage": {"input_tokens": None, "output_tokens": None},
    }
    value.update(changes)
    return value


def query_body():
    return chat_body(
        status="answered", answer="请以实时商品规格为准。",
        pending_action=None, action_result=None, model_rounds=2,
        tool_calls=[{"name": "search_products", "status": "succeeded", "elapsed_ms": 1}],
        products=[{
            "product_id": 36, "name": "华为商品", "description": "", "image": "",
            "skus": [{"unique_id": SKU, "name": "A4,白色", "price": "1.00",
                      "stock": 8882, "image": ""}],
        }],
        usage={"input_tokens": 10, "output_tokens": 2},
    )


def run(project, handler, argv=None, *, environ=None, destination=None):
    stream = io.StringIO()
    path = project / "data" / "day6-action-test.json" if destination is None else destination
    code = runner().main(
        ["--message", PREPARE, "--session-id", SID] if argv is None else argv,
        environ={"YSHOP_AUTHORIZATION": AUTH, "LLM_API_KEY": MODEL_KEY}
        if environ is None else environ,
        env_file=project / "unused.env", transport=httpx.MockTransport(handler),
        report_path=path, stdout=stream,
    )
    return code, json.loads(stream.getvalue()), path


def test_prepare_sends_one_local_chat_post_and_saves_only_public_card(project):
    requests = []
    body = chat_body()
    body["internal"] = {"authorization": AUTH, "model_key": MODEL_KEY, "reasoning": "private"}
    body["pending_action"]["executor"] = AUTH
    body["pending_action"]["nickname"] = MODEL_KEY

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=body)

    code, summary, path = run(project, handler)
    assert code == 0 and summary["action_status"] == "awaiting_confirmation"
    assert summary["action_id"] == AID and summary["session_id"] == SID
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST" and request.url.path == "/agent/chat"
    assert request.url.host == "127.0.0.1" and not request.url.query
    assert request.headers["authorization"] == AUTH
    assert json.loads(request.content) == {"message": PREPARE, "session_id": SID}
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["operation"] == "prepare" and report["mode"] == "test_double"
    assert report["provenance"]["external_model_verified"] is False
    assert report["provenance"]["injected_transport"] is True
    assert report["local_http_requests"] == 1 and report["business_write_requests"] is None
    assert report["result"]["pending_action"]["action_id"] == AID
    for secret in (AUTH, AUTH.partition(" ")[2], MODEL_KEY, "executor", "internal", "nickname", "reasoning"):
        assert secret not in path.read_text(encoding="utf-8")
        assert secret not in json.dumps(summary)


def test_prepare_can_query_without_session_and_retains_direct_read_contract(project):
    requests = []
    code, summary, path = run(
        project, lambda request: requests.append(request) or httpx.Response(200, json=query_body()),
        ["--message", QUERY],
    )
    assert code == 0 and summary["action_status"] == "answered"
    assert json.loads(requests[0].content) == {"message": QUERY}
    assert json.loads(path.read_text(encoding="utf-8"))["result"]["model_rounds"] == 2


@pytest.mark.parametrize("decision,status", [("confirm", "succeeded"), ("cancel", "cancelled")])
def test_saved_id_decision_posts_only_decision_and_never_reads_cart(project, decision, status):
    requests = []
    code, summary, path = run(
        project, lambda request: requests.append(request) or httpx.Response(200, json=action_body(status)),
        ["--action-id", AID, "--decision", decision],
    )
    assert code == 0 and summary["action_status"] == status and summary["action_id"] == AID
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == f"/agent/actions/{AID}/decision"
    assert json.loads(requests[0].content) == {"decision": decision}
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["operation"] == "decision" and report["decision"]["request_status"] == "completed"
    assert report["business_write_requests"] is None


def test_cli_rejects_existing_report_before_confirmation(project):
    report = project / "data" / "day6-existing.json"
    report.parent.mkdir()
    report.write_text("protected", encoding="utf-8")
    requests = []
    code, _, _ = run(project, lambda request: requests.append(request), [
        "--action-id", AID, "--decision", "confirm",
    ], destination=report)
    assert code == 1 and report.read_text(encoding="utf-8") == "protected"
    assert requests == []


@pytest.mark.parametrize("argv", [
    [], ["--action-id", AID], ["--decision", "confirm"],
    ["--message", PREPARE, "--decision", "confirm"],
    ["--message", PREPARE, "--action-id", AID, "--decision", "confirm"],
    ["--action-id", AID, "--decision", "confirm", "--session-id", SID],
    ["--action-id", AID, "--decision", "CONFIRM"],
    ["--action-id", AID.upper(), "--decision", "confirm"],
    ["--action-id", "a" * 31, "--decision", "confirm"],
    ["--action-id", "x" * 32, "--decision", "confirm"],
    ["--message", " "], ["--message", "x" * 1001],
    ["--message", PREPARE, "--message=x"],
    ["--action-id", AID, "--decision", "confirm", "--decision=cancel"],
    ["--action-id", AID, "--decision", "confirm", "--action-id=" + AID],
    ["--message", PREPARE, "--session-id", SID, "--session-id=" + SID],
    ["--message", PREPARE, "--session-id", "invalid"],
    ["--message", PREPARE, "--api-url", "http://localhost", "--api-url=http://127.0.0.1"],
    ["--message", PREPARE, "--report", "data/a.json", "--report=data/b.json"],
    ["--message", PREPARE, "--mode", "test_double", "--mode=test_double"],
    ["--message", PREPARE, "--mode", "unknown"], ["--message", PREPARE, "--quantity", "1"],
    ["--action-id", AID, "--decision", "confirm", "--parameters", '{"quantity":9999}'],
    ["--mess", PREPARE], ["--token", AUTH], ["--message", PREPARE, "--unknown", MODEL_KEY],
    ["--message", PREPARE, "--api-url"],
])
def test_invalid_or_duplicate_flags_never_send_http(project, argv):
    code, summary, path = run(project, lambda request: pytest.fail("invalid input sent HTTP"), argv)
    assert code == 1 and summary["error_code"] == "invalid_request"
    assert not path.exists() and AUTH not in json.dumps(summary) and MODEL_KEY not in json.dumps(summary)


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8010", "http://example.com", "http://127.0.0.1.evil",
    "http://user:secret@localhost", "http://localhost/agent/chat",
    "http://localhost?x=1", "http://localhost#frag", "http://localhost:0",
    "http://localhost:65536", "http://localhost:abc", "http://localhost\\x",
    "http://localhost\n", "http://localhost\x00",
])
def test_non_loopback_or_decorated_origin_is_rejected_before_decision(project, url):
    code, summary, _ = run(project, lambda request: pytest.fail("invalid origin sent HTTP"), [
        "--action-id", AID, "--decision", "confirm", "--api-url", url,
    ])
    assert code == 1 and summary["error_code"] == "invalid_request"


@pytest.mark.parametrize("url", ["http://localhost:8016/", "http://[::1]:8016", "http://127.0.0.1:8016"])
def test_loopback_origins_disable_redirects_and_environment_proxy(project, monkeypatch, url):
    module = runner()
    original = httpx.AsyncClient
    configurations = []

    def client(**kwargs):
        configurations.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client)
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted.invalid:9999")
    code, _, _ = run(project, lambda request: httpx.Response(200, json=chat_body()), [
        "--message", PREPARE, "--api-url", url,
    ])
    assert code == 0
    assert configurations[0]["trust_env"] is False
    assert configurations[0]["follow_redirects"] is False


@pytest.mark.parametrize("authorization", [None, "", "Basic secret", AUTH + "\n", "Bearer two words"])
def test_invalid_identity_never_sends_a_decision(project, authorization):
    environment = {} if authorization is None else {"YSHOP_AUTHORIZATION": authorization}
    assert run(project, lambda request: pytest.fail("invalid identity sent HTTP"), [
        "--action-id", AID, "--decision", "confirm",
    ], environ=environment)[1]["error_code"] == "login_required"


def test_environment_precedence_and_env_file_token_loading(project):
    env_file = project / "local.env"
    env_file.write_text("YSHOP_AUTHORIZATION=" + AUTH + "\nLLM_API_KEY=" + MODEL_KEY + "\n", encoding="utf-8")
    requests = []
    module = runner()
    stream = io.StringIO()
    destination = project / "data" / "day6-from-env.json"
    assert module.main(["--action-id", AID, "--decision", "cancel"], environ={},
        env_file=env_file, transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, json=action_body("cancelled"))),
        report_path=destination, stdout=stream) == 0
    assert requests[0].headers["authorization"] == AUTH
    assert AUTH not in destination.read_text(encoding="utf-8")
    assert module.main(["--action-id", AID, "--decision", "confirm"],
        environ={"YSHOP_AUTHORIZATION": ""}, env_file=env_file,
        transport=httpx.MockTransport(lambda request: pytest.fail("empty override sent HTTP")),
        report_path=project / "data" / "day6-empty-override.json", stdout=io.StringIO()) == 1


@pytest.mark.parametrize("relative", [
    ".env", "README.md", "evals/action_cases.json", "knowledge/policies/demo.json",
    "data/chroma/report.json", ".cache/report.json", "data/fastembed/report.json",
    "data/manifest.json", "data/day1-new.json", "data/day2-new.json",
    "data/day3-new.json", "data/day4-new.json", "data/day5-new.json",
    "data/action.txt", "../outside.json",
])
def test_protected_reports_and_prior_day_names_are_immutable(project, relative):
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"protected")
    code, _, _ = run(project, lambda request: pytest.fail("protected path sent HTTP"), [
        "--action-id", AID, "--decision", "confirm", "--report", str(path),
    ])
    assert code == 1 and path.read_bytes() == b"protected"


@pytest.mark.parametrize("field", ["RAG_CHROMA_DIR", "RAG_KNOWLEDGE_DIR", "RAG_MODEL_CACHE_DIR"])
def test_configured_protected_directories_are_rejected_before_decision(project, field):
    destination = project / "data" / "private" / "day6-action.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("private", encoding="utf-8")
    code, _, _ = run(project, lambda request: pytest.fail("private directory sent HTTP"), [
        "--action-id", AID, "--decision", "confirm", "--report", str(destination),
    ], environ={"YSHOP_AUTHORIZATION": AUTH, field: str(destination.parent)})
    assert code == 1 and destination.read_text(encoding="utf-8") == "private"


def test_preexisting_atomic_temporary_is_not_removed(project):
    destination = project / "data" / "day6-action-test.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.parent.mkdir()
    temporary.write_text("other invocation", encoding="utf-8")
    assert run(project, lambda request: pytest.fail("report collision sent HTTP"))[0] == 1
    assert temporary.read_text(encoding="utf-8") == "other invocation"


def test_foreign_temporary_created_during_request_is_preserved(project):
    temporary = project / "data" / "day6-action-test.json.tmp"

    def handler(request):
        temporary.write_text("other invocation", encoding="utf-8")
        return httpx.Response(200, json=action_body("cancelled"))

    assert run(project, handler, ["--action-id", AID, "--decision", "cancel"])[0] == 0
    assert temporary.read_text(encoding="utf-8") == "other invocation"


def test_report_failure_before_decision_prevents_the_sole_post(project, monkeypatch):
    module = runner()
    requests = []

    def fail(path, report, authorization):
        raise OSError("private " + AUTH + MODEL_KEY)

    monkeypatch.setattr(module, "_save_owned_report", fail)
    code, summary, path = run(project, lambda request: requests.append(request), [
        "--action-id", AID, "--decision", "confirm",
    ])
    assert code == 1 and summary["error_code"] == "report_write_failed"
    assert requests == []
    assert AUTH not in path.read_text(encoding="utf-8") and MODEL_KEY not in json.dumps(summary)


def test_final_report_failure_is_nonzero_without_another_request(project, monkeypatch):
    module = runner()
    original = module._save_owned_report
    requests = []

    def fail_final(path, report, authorization):
        if report["status"] == "passed":
            raise OSError("private " + MODEL_KEY)
        original(path, report, authorization)

    monkeypatch.setattr(module, "_save_owned_report", fail_final)
    code, summary, _ = run(
        project, lambda request: requests.append(request) or httpx.Response(200, json=action_body("succeeded")),
        ["--action-id", AID, "--decision", "confirm"],
    )
    assert code == 1 and summary["error_code"] == "report_write_failed" and len(requests) == 1


@pytest.mark.parametrize("status,code", [
    (401, "login_required"), (404, "action_not_found"), (409, "action_expired"),
    (409, "action_conflict"), (503, "session_unavailable"), (502, "model_invalid_response"),
    (302, "action_http_error"), (500, "action_http_error"),
])
def test_foreign_or_failed_http_is_sanitized_without_redirects_or_retry(project, status, code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": code, "message": AUTH + MODEL_KEY}},
                              headers={"Location": "https://untrusted.invalid"})

    result, summary, path = run(project, handler, ["--action-id", AID, "--decision", "confirm"])
    assert result == 1 and summary["error_code"] == code and len(requests) == 1
    assert AUTH not in path.read_text(encoding="utf-8") and MODEL_KEY not in json.dumps(summary)


@pytest.mark.parametrize("error", [httpx.ReadTimeout(AUTH), httpx.ConnectError(MODEL_KEY)])
def test_decision_transport_failure_stops_after_one_request_with_unknown_outcome(project, error):
    requests = []

    def handler(request):
        requests.append(request)
        raise error

    code, summary, path = run(project, handler, ["--action-id", AID, "--decision", "confirm"])
    assert code == 1 and summary["error_code"] == "action_http_error" and len(requests) == 1
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["decision"]["request_status"] == "unknown"
    assert report["decision"]["retry_permitted"] is False
    assert AUTH not in path.read_text(encoding="utf-8") and MODEL_KEY not in json.dumps(summary)


@pytest.mark.parametrize("raw", [
    'not-json', 'null', '[]',
    '{"status":"succeeded","status":"cancelled"}',
    json.dumps(action_body("succeeded")).replace('"cart_id": 41', '"cart_id": NaN'),
    json.dumps(chat_body()).replace('"quantity": 1', '"quantity": 1, "quantity": 2'),
    json.dumps(chat_body()).replace('"unit_price": "1.00"', '"unit_price": Infinity'),
    json.dumps(chat_body()).replace('"total_price": "1.00"', '"total_price": 1e999'),
])
def test_strict_response_json_rejects_duplicates_nonfinite_or_nonobjects(project, raw):
    requests = []
    code, summary, path = run(
        project, lambda request: requests.append(request) or httpx.Response(200, content=raw),
    )
    assert code == 1 and summary["error_code"] == "action_invalid_response" and len(requests) == 1
    assert "result" not in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("change", [
    {"action_id": "a" * 31}, {"session_id": "c" * 32}, {"quantity": True},
    {"quantity": "1"}, {"quantity": 0}, {"quantity": 10000}, {"unit_price": "-1.00"},
    {"total_price": "2.00"}, {"product_id": "36"}, {"sku_name": " "}, {"expires_at": True},
])
def test_malformed_public_card_is_refused(project, change):
    assert run(project, lambda request: httpx.Response(200, json=chat_body(
        pending_action=card(**change))))[1]["error_code"] == "action_invalid_response"


@pytest.mark.parametrize("field", ["pending_action", "action_result", "usage", "products", "session_id"])
def test_prepare_requires_complete_public_shape(project, field):
    body = chat_body()
    del body[field]
    assert run(project, lambda request: httpx.Response(200, json=body))[1]["error_code"] == "action_invalid_response"


@pytest.mark.parametrize("change", [
    {"model_rounds": True}, {"model_rounds": 1},
    {"tool_calls": [{"name": "add_to_cart", "status": "succeeded", "elapsed_ms": 1}]},
    {"products": query_body()["products"]}, {"usage": {"input_tokens": 1, "output_tokens": 1}},
    {"status": "succeeded"}, {"action_result": action_body("succeeded")["action_result"]},
])
def test_preparation_rejects_incoherent_action_or_fabricated_model_fields(project, change):
    assert run(project, lambda request: httpx.Response(200, json=chat_body(**change)))[1]["error_code"] == "action_invalid_response"


@pytest.mark.parametrize("secret", [AUTH, AUTH.partition(" ")[2], MODEL_KEY])
@pytest.mark.parametrize("field", ["answer", "product_name", "sku_name", "image"])
def test_public_credential_echo_fails_without_saving_it(project, secret, field):
    body = chat_body()
    if field == "answer":
        body[field] = secret
    else:
        body["pending_action"][field] = secret
    code, summary, path = run(project, lambda request: httpx.Response(200, json=body))
    assert code == 1 and summary["error_code"] == "action_invalid_response"
    assert secret not in path.read_text(encoding="utf-8") and secret not in json.dumps(summary)


def test_credential_in_prepare_message_is_rejected_before_reservation(project):
    code, summary, path = run(project, lambda request: pytest.fail("credential sent HTTP"), ["--message", MODEL_KEY])
    assert code == 1 and not path.exists() and MODEL_KEY not in json.dumps(summary)


def test_replacement_card_needs_a_new_explicit_decision_without_auto_confirm(project):
    requests = []
    code, summary, path = run(
        project, lambda request: requests.append(request) or httpx.Response(
            200, json=action_body(pending=card(SECOND_AID, unit_price="2.00", total_price="2.00"))),
        ["--action-id", AID, "--decision", "confirm"],
    )
    assert code == 1 and summary["error_code"] == "action_reconfirmation_required"
    assert len(requests) == 1 and summary["action_id"] == SECOND_AID
    assert json.loads(path.read_text(encoding="utf-8"))["result"]["pending_action"]["action_id"] == SECOND_AID


@pytest.mark.parametrize("status", ["executing", "failed", "unknown", "cancelled"])
def test_confirmation_not_succeeded_is_nonzero_without_replay(project, status):
    requests = []
    code, summary, _ = run(
        project, lambda request: requests.append(request) or httpx.Response(200, json=action_body(status)),
        ["--action-id", AID, "--decision", "confirm"],
    )
    assert code == 1 and summary["action_status"] == status and len(requests) == 1


def test_decision_response_must_refer_to_requested_action(project):
    assert run(project, lambda request: httpx.Response(200, json=action_body("succeeded", SECOND_AID)), [
        "--action-id", AID, "--decision", "confirm",
    ])[1]["error_code"] == "action_invalid_response"


def test_injected_transport_cannot_attest_real_model(project):
    assert run(project, lambda request: pytest.fail("false attestation sent HTTP"), [
        "--message", QUERY, "--mode", "real_model",
    ])[1]["error_code"] == "invalid_request"


def test_frozen_cases_have_exact_approved_target_and_real_steps():
    path = ROOT / "evals" / "action_cases.json"
    assert path.is_file(), "frozen Day 6 action cases are missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["frozen_before_live_requests"] is True
    assert data["live_acceptance_status"] == "pending"
    scenario = data["scenario"]
    assert scenario["query"] == QUERY and scenario["prepare"] == PREPARE
    assert scenario["target"] == {
        "product_id": 36, "unique_id": SKU, "sku_name": "A4,白色",
        "quantity": 1, "unit_price": "1.00",
    }
    assert scenario["preflight_declaration"]["stock"] == 8882
    assert scenario["steps"] == [
        "query", "cart_before_cancel", "prepare_cancel", "cancel", "cart_after_cancel",
        "prepare_saved", "stop_agent", "start_agent_same_database", "recover_saved_action",
        "cart_before_confirm", "confirm_once", "cart_after_confirm", "confirm_duplicate",
        "cart_after_duplicate",
    ]
    assert scenario["expected"] == {
        "cancellation_cart_writes": 0, "confirmation_cart_writes": 1,
        "duplicate_cart_writes": 0, "same_sku_quantity_growth": 1,
    }
    assert all(case["test_selector"].startswith("tests/") for case in data["deterministic_cases"])


class FakeAgent:
    """Lifecycle and HTTP boundary for exercising the real workflow decisions."""

    def __init__(self, *, changed_card=False, changed_query=False, timeout=False,
                 replacement=False, mismatch=False, duplicate_write=False, cancel_write=False):
        self.requests = []
        self.lifecycle = []
        self.observations = []
        self.quantity = 3
        self.prepare_count = 0
        self.confirm_count = 0
        self.alive = False
        self.first_card = card()
        self.saved_card = card(SECOND_AID)
        self.changed_card = changed_card
        self.changed_query = changed_query
        self.timeout = timeout
        self.replacement = replacement
        self.mismatch = mismatch
        self.duplicate_write = duplicate_write
        self.cancel_write = cancel_write

    def start(self, number):
        self.lifecycle.append(("start", number))
        self.alive = True
        self.observations.append({"kind": "startup", "pid": 100 + number, "port": 8016})
        return 100 + number

    def stop(self):
        self.lifecycle.append(("stop",))
        self.alive = False

    async def ready(self, client):
        self.lifecycle.append(("ready",))

    def events(self):
        return deepcopy(self.observations)

    def shop_event(self, method, path):
        self.observations.append({"kind": "shop_request", "method": method, "path": path,
                                  "status": 200, "elapsed_ms": 1})

    def __call__(self, request):
        self.requests.append(request)
        self.shop_event("GET", "/api/userinfo")
        path = request.url.path
        if path == "/agent/me":
            return httpx.Response(200, json={"user_id": 33, "nickname": "private-nickname"})
        if path == "/agent/chat":
            payload = json.loads(request.content)
            if payload["message"] == QUERY:
                self.observations.extend([
                    {"kind": "model_request", "method": "POST", "status": 200,
                     "elapsed_ms": 1, "enable_thinking": False},
                    {"kind": "model_protocol_check", "status": "valid"},
                ])
                self.shop_event("GET", "/api/agent/products")
                body = query_body()
                if self.changed_query:
                    body["products"][0]["product_id"] = 37
                return httpx.Response(200, json=body)
            assert payload == {"message": PREPARE, "session_id": SID}
            self.prepare_count += 1
            self.shop_event("GET", "/api/product/detail/36")
            self.shop_event("GET", "/api/cart/list")
            pending = self.first_card if self.prepare_count == 1 else self.saved_card
            if self.changed_card:
                pending = {**pending, "unit_price": "2.00", "total_price": "2.00"}
            return httpx.Response(200, json=chat_body(pending_action=pending))
        if path == "/agent/cart":
            self.shop_event("GET", "/api/cart/list")
            return httpx.Response(200, json={"valid": [{
                "cart_id": 41, "product_id": 36, "unique_id": SKU, "quantity": self.quantity,
                "name": "华为商品", "sku_name": "A4,白色", "image": "", "price": "1.00", "stock": 8882,
            }], "invalid": []})
        if path == f"/agent/actions/{SECOND_AID}":
            return httpx.Response(200, json=action_body(pending=self.saved_card))
        if path == f"/agent/actions/{AID}/decision":
            assert json.loads(request.content) == {"decision": "cancel"}
            if self.cancel_write:
                self.shop_event("POST", "/api/cart/add")
            return httpx.Response(200, json=action_body("cancelled"))
        if path == f"/agent/actions/{SECOND_AID}/decision":
            assert json.loads(request.content) == {"decision": "confirm"}
            self.confirm_count += 1
            if self.timeout:
                raise httpx.ReadTimeout(AUTH + MODEL_KEY, request=request)
            if self.replacement:
                return httpx.Response(200, json=action_body(
                    pending=card("d" * 32, unit_price="2.00", total_price="2.00")))
            if self.confirm_count == 1 or self.duplicate_write:
                self.shop_event("POST", "/api/cart/add")
                self.quantity += 2 if self.mismatch else 1
            return httpx.Response(200, json=action_body("succeeded", SECOND_AID))
        pytest.fail("unexpected local test path")


def exercise_workflow(agent, *, fail_save_at=None, saved_id=None, definition_unchanged=True):
    module = verifier()
    scenario, digest = module._load_cases(ROOT / "evals" / "action_cases.json")
    report = module._new_report(digest, observation_source="test_double")
    history = []

    def save():
        if fail_save_at is not None and report.get("phase") == fail_save_at:
            raise OSError(AUTH + MODEL_KEY)
        history.append(deepcopy(report))

    def load_saved():
        return saved_id if saved_id is not None else history[-1]["saved_action_id"]

    async def execute():
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8016", trust_env=False,
                                     transport=httpx.MockTransport(agent)) as client:
            await module._workflow(
                client, agent, report, scenario=scenario, authorization=AUTH,
                secrets=(AUTH, AUTH.partition(" ")[2], MODEL_KEY), save=save,
                load_saved=load_saved, definition_unchanged=lambda: definition_unchanged,
            )

    try:
        asyncio.run(execute())
    except module.VerificationStopped as error:
        return report, history, error.code
    except OSError:
        return report, history, "report_write_failed"
    return report, history, None


def test_workflow_cancels_restarts_same_saved_id_and_compares_exact_cart_growth():
    agent = FakeAgent()
    report, history, error = exercise_workflow(agent)
    assert error is None and agent.quantity == 4 and agent.confirm_count == 2
    assert agent.lifecycle == [("start", 1), ("ready",), ("stop",), ("start", 2), ("ready",)]
    assert report["saved_action_id"] == SECOND_AID
    assert report["cart_checks"]["cancel"]["before_quantity"] == 3
    assert report["cart_checks"]["cancel"]["after_quantity"] == 3
    assert report["cart_checks"]["confirm"]["expected_quantity"] == 4
    assert report["cart_checks"]["confirm"]["returned_cart_id"] == 41
    assert report["cart_checks"]["duplicate"]["before_quantity"] == 4
    assert report["cart_checks"]["duplicate"]["after_quantity"] == 4
    assert report["downstream_writes"] == {"cancel": 0, "confirm": 1, "duplicate": 0}
    assert report["provenance"]["external_model_verified"] is False
    assert report["provenance"]["observation_source"] == "test_double"
    assert all(request.url.host == "127.0.0.1" for request in agent.requests)
    assert all(request.url.path != "/api/cart/add" for request in agent.requests)
    assert any(row.get("saved_action_id") == SECOND_AID for row in history)
    for secret in (AUTH, AUTH.partition(" ")[2], MODEL_KEY, "private-nickname"):
        assert secret not in json.dumps(report, ensure_ascii=False)


@pytest.mark.parametrize("option", ["changed_card", "changed_query", "cancel_write"])
def test_changed_frozen_target_or_cancel_write_stops_before_any_confirmation(option):
    agent = FakeAgent(**{option: True})
    report, _, error = exercise_workflow(agent)
    assert error is not None and agent.confirm_count == 0
    assert not any(request.url.path == f"/agent/actions/{SECOND_AID}/decision" for request in agent.requests)
    assert any(not check["passed"] for check in report["checks"])


@pytest.mark.parametrize("option", ["timeout", "replacement", "mismatch"])
def test_failed_or_replacement_confirmation_stops_without_next_request_or_retry(option):
    agent = FakeAgent(**{option: True})
    report, _, error = exercise_workflow(agent)
    assert error is not None and agent.confirm_count == 1
    decision_index = next(index for index, request in enumerate(agent.requests)
                          if request.url.path == f"/agent/actions/{SECOND_AID}/decision")
    if option in {"timeout", "replacement"}:
        assert decision_index == len(agent.requests) - 1
    assert all(attempt["retry_permitted"] is False for attempt in report["decision_attempts"])


def test_duplicate_confirmation_write_is_a_failed_batch_not_acceptance():
    agent = FakeAgent(duplicate_write=True)
    report, _, error = exercise_workflow(agent)
    assert error is not None and report["downstream_writes"]["duplicate"] == 1
    assert agent.confirm_count == 2


def test_verifier_report_failure_before_confirmation_sends_no_decision():
    agent = FakeAgent()
    _, _, error = exercise_workflow(agent, fail_save_at="confirm_once")
    assert error == "report_write_failed" and agent.confirm_count == 0


def test_frozen_file_change_and_saved_id_mismatch_stop_before_confirmation():
    for settings in ({"definition_unchanged": False}, {"saved_id": "e" * 32}):
        agent = FakeAgent()
        _, _, error = exercise_workflow(agent, **settings)
        assert error is not None and agent.confirm_count == 0


@pytest.mark.parametrize("argv", [
    [], ["--suffix", "safe"], ["--suffix", "safe", "--allow-live-write", "--allow-live-write"],
    ["--suffix", "../escape", "--allow-live-write"],
    ["--suffix", "safe", "--allow-live-write", "--port", "0"],
    ["--suffix", "safe", "--allow-live-write", "--port", "8016.0"],
    ["--suffix", "safe", "--allow-live-write", "--port", "8016", "--port=8017"],
    ["--suffix", "safe", "--allow-live-write", "--message", PREPARE],
])
def test_verifier_invalid_arguments_or_missing_opt_in_do_not_launch_or_request(monkeypatch, argv):
    module = verifier()
    monkeypatch.setattr(module, "OwnedAgent", lambda *args, **kwargs: pytest.fail("invalid verifier launched"))
    stream = io.StringIO()
    assert module.main(argv, environ={}, env_file=ROOT / "absent.env", stdout=stream) != 0
    assert AUTH not in stream.getvalue() and MODEL_KEY not in stream.getvalue()


def test_verifier_preserves_existing_report_before_launch(tmp_path, monkeypatch):
    module = verifier()
    root = tmp_path / "project"
    service = root / "yshop-agent"
    evidence = root / "logs" / "agent-day6"
    evidence.mkdir(parents=True)
    service.mkdir()
    path = evidence / "live-action-safe.json"
    path.write_text("protected", encoding="utf-8")
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    monkeypatch.setattr(module, "SERVICE_ROOT", service)
    monkeypatch.setattr(module, "OwnedAgent", lambda *args, **kwargs: pytest.fail("collision launched process"))
    stream = io.StringIO()
    assert module.main(["--suffix", "safe", "--allow-live-write"],
        environ={"YSHOP_AUTHORIZATION": AUTH}, env_file=service / ".env", stdout=stream) != 0
    assert path.read_text(encoding="utf-8") == "protected"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CPython venv redirector regression")
def test_owned_windows_child_pid_matches_popen_and_preserves_project_venv(tmp_path, monkeypatch):
    module = verifier()
    root = tmp_path / "project"
    service = root / "yshop-agent"
    database = service / ".tmp" / "day6-live-safe" / "sessions.sqlite3"
    database.parent.mkdir(parents=True)
    evidence = root / "logs" / "agent-day6"
    evidence.mkdir(parents=True)
    script = service / "scripts" / "day6_observer.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse, json, os, sys\n"
        "from pathlib import Path\n"
        "import httpx\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--port', type=int)\n"
        "parser.add_argument('--events')\n"
        "parser.add_argument('--database')\n"
        "options = parser.parse_args()\n"
        "events = Path(options.events)\n"
        "with events.open('x', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps({'kind': 'startup', 'pid': os.getpid(), "
        "'port': options.port}) + '\\n')\n"
        "with events.with_suffix('.json').open('x', encoding='utf-8') as stream:\n"
        "    json.dump({'pid': os.getpid(), 'executable': sys.executable, "
        "'prefix': sys.prefix, 'httpx': httpx.__file__, "
        "'database': options.database, 'session_db': os.environ['SESSION_DB_PATH']}, stream)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    monkeypatch.setattr(module, "SERVICE_ROOT", service)
    paths = [evidence / f"events-safe-{number}.jsonl" for number in (1, 2)]
    agent = module.OwnedAgent(port=8016, database=database, event_paths=paths, environ=os.environ)
    for number, path in enumerate(paths, 1):
        try:
            pid = agent.start(number)
            process = agent.process
            assert isinstance(process, subprocess.Popen)
            # Reap the short-lived fixture before assertions, also under the broken redirector.
            assert process.wait(timeout=30) == 0
            observed = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            print(json.dumps({"launch": number, "popen_pid": pid, **observed}))
            assert observed["pid"] == pid == process.pid
            assert Path(observed["executable"]).resolve() == Path(sys.executable).resolve()
            assert Path(observed["prefix"]).resolve() == (ROOT / ".venv").resolve()
            assert Path(observed["httpx"]).resolve() == Path(httpx.__file__).resolve()
            assert observed["database"] == observed["session_db"] == str(database)
            assert {"kind": "startup", "pid": pid, "port": 8016} in agent.events()
        finally:
            agent.stop()
        assert process.poll() is not None


@pytest.mark.parametrize("platform,executable,base_executable", [
    ("win32", "D:/project/.venv/Scripts/python.exe", "C:/runtime/python.exe"),
    ("win32", "C:/runtime/python.exe", "C:/runtime/python.exe"),
    ("linux", "/project/.venv/bin/python", "/runtime/python"),
])
def test_owned_process_commands_reuse_database_apply_environment_and_stop_only_owned(
        tmp_path, monkeypatch, platform, executable, base_executable):
    module = verifier()
    root = tmp_path / "project"
    service = root / "yshop-agent"
    database = service / ".tmp" / "day6-live-safe" / "sessions.sqlite3"
    database.parent.mkdir(parents=True)
    evidence = root / "logs" / "agent-day6"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    monkeypatch.setattr(module, "SERVICE_ROOT", service)
    monkeypatch.setattr(module, "sys", SimpleNamespace(
        platform=platform, executable=executable, _base_executable=base_executable))
    calls = []
    processes = []

    class Process:
        def __init__(self):
            self.pid = 1000 + len(processes)
            self.running = True
            self.terminations = 0
            self.waits = []
            processes.append(self)

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminations += 1
            self.running = False

        def wait(self, timeout):
            self.waits.append(timeout)
            return 0

        def kill(self):
            pytest.fail("normal shutdown should not kill a process")

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    agent = module.OwnedAgent(port=8016, database=database,
        event_paths=[evidence / "events-safe-1.jsonl", evidence / "events-safe-2.jsonl"],
        environ={"LLM_ENABLE_THINKING": "true", "LLM_TIMEOUT_SECONDS": "1", "KEEP": "present",
                 "__PYVENV_LAUNCHER__": "inherited-launcher"})
    agent.stop()
    first = agent.start(1)
    agent.stop()
    second = agent.start(2)
    agent.stop()
    agent.stop()
    assert first != second and len(calls) == 2
    assert all(process.terminations == 1 and process.waits == [20] for process in processes)
    redirector = platform == "win32" and executable != base_executable
    for command, options in calls:
        assert command[0] == (base_executable if redirector else executable)
        assert command[1:5] == ["-X", "utf8", "-W", "error"]
        assert command[5] == str(service / "scripts" / "day6_observer.py")
        assert command[command.index("--database") + 1] == str(database)
        assert options["shell"] is False and options["cwd"] == service
        assert options["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
        assert options["stdout"] == subprocess.DEVNULL and options["stderr"] == subprocess.DEVNULL
        assert options["env"]["LLM_ENABLE_THINKING"] == "false"
        assert options["env"]["LLM_TIMEOUT_SECONDS"] == "60"
        assert options["env"]["LLM_PROXY_URL"] == "http://127.0.0.1:7892"
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["env"]["KEEP"] == "present"
        assert options["env"]["__PYVENV_LAUNCHER__"] == (
            executable if redirector else "inherited-launcher")
        assert Path(options["env"]["TEMP"]).is_relative_to(service / ".tmp")
        assert options["env"]["TMP"] == options["env"]["TEMP"]
    assert agent.environment["__PYVENV_LAUNCHER__"] == "inherited-launcher"


def test_observer_records_only_metadata_and_never_changes_real_response():
    module = observer()
    events = []
    seen = []
    body = {"choices": [{"message": {"role": "assistant", "content": "safe",
            "reasoning_content": AUTH + MODEL_KEY + "private-nickname"}}]}
    response = httpx.Response(200, json=body, headers={"X-Private": AUTH})

    async def original(client, request, *args, **kwargs):
        seen.append(request)
        return response

    send = module._observed_send(original, events.append,
        shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.invalid/v1/chat/completions", model_key=MODEL_KEY)

    async def execute():
        request = httpx.Request("POST", "https://model.invalid/v1/chat/completions",
            headers={"Authorization": AUTH}, json={"enable_thinking": False, "messages": [MODEL_KEY]})
        assert await send(None, request) is response
        shop = httpx.Request("GET", "http://127.0.0.1:8008/api/userinfo?nickname=private-nickname&token=" + AUTH,
                            headers={"Authorization": AUTH})
        assert await send(None, shop) is response

    asyncio.run(execute())
    assert len(seen) == 2 and response.json() == body
    assert events[0] == {
        "kind": "model_request", "method": "POST", "enable_thinking": False,
        "status": 200, "elapsed_ms": events[0]["elapsed_ms"],
    }
    assert events[1] == {"kind": "model_protocol_check", "status": "valid"}
    assert events[2]["kind"] == "shop_request" and events[2]["path"] == "/api/userinfo"
    saved = json.dumps(events)
    for secret in (AUTH, MODEL_KEY, "private-nickname", "reasoning_content", "Authorization", "messages", "model.invalid"):
        assert secret not in saved


@pytest.mark.parametrize("error,category", [
    (httpx.ReadTimeout(AUTH), "timeout"), (httpx.ConnectError(MODEL_KEY), "network_error"),
    (RuntimeError("private-nickname"), "unexpected_error"),
])
def test_observer_transport_errors_use_closed_categories_without_exception_text(error, category):
    module = observer()
    events = []

    async def original(client, request, *args, **kwargs):
        raise error

    send = module._observed_send(original, events.append,
        shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.invalid/v1/chat/completions", model_key=MODEL_KEY)
    with pytest.raises(type(error)):
        asyncio.run(send(None, httpx.Request("GET", "http://127.0.0.1:8008/api/cart/list")))
    assert events[0]["transport_error"] == category
    assert AUTH not in json.dumps(events) and MODEL_KEY not in json.dumps(events)
    assert "private-nickname" not in json.dumps(events)


def test_observer_parser_failure_is_a_closed_category_and_unknown_paths_are_not_logged(monkeypatch):
    module = observer()
    events = []
    response = httpx.Response(200, content='{"choices":[],"choices":[]}')

    async def original(client, request, *args, **kwargs):
        return response

    send = module._observed_send(original, events.append,
        shop_url="http://127.0.0.1:8008/api/",
        model_url="https://model.invalid/v1/chat/completions", model_key=MODEL_KEY)
    asyncio.run(send(None, httpx.Request("POST", "https://model.invalid/v1/chat/completions",
                                       json={"enable_thinking": False})))
    assert events[1] == {"kind": "model_protocol_check", "status": "invalid",
                         "reason": "duplicate JSON key", "phase": "envelope"}
    asyncio.run(send(None, httpx.Request("GET", "http://127.0.0.1:8008/api/" + MODEL_KEY)))
    assert events[-1]["path"] == "other_shop_path" and MODEL_KEY not in json.dumps(events)


@pytest.mark.parametrize("events_relative,database_relative", [
    ("logs/agent-day5/events.jsonl", "yshop-agent/.tmp/day6-live-safe/sessions.sqlite3"),
    ("logs/agent-day6/events.jsonl", "yshop-agent/data/day5-sessions.sqlite3"),
    ("logs/agent-day6/events.jsonl", "outside/sessions.sqlite3"),
])
def test_observer_rejects_old_evidence_or_operational_database_before_start(tmp_path, monkeypatch,
                                                                          events_relative, database_relative):
    module = observer()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SERVICE", tmp_path / "yshop-agent")
    monkeypatch.setattr(module, "_run_server", lambda *args, **kwargs: pytest.fail("unsafe observer launched"))
    events = tmp_path / events_relative
    database = tmp_path / database_relative
    assert module.main(["--port", "8016", "--events", str(events), "--database", str(database)]) != 0
    assert not events.exists()
