"""Continuous chat CLI preserves session IDs and protects existing evidence."""
import importlib
import importlib.util
import io
import json

import httpx
import pytest

from scripts import eval_policy
from tests.test_chat_cli import AUTH, MODEL_KEY, _body


def runner():
    assert importlib.util.find_spec("scripts.session_smoke_test"), "session CLI missing"
    return importlib.import_module("scripts.session_smoke_test")


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "service"
    root.mkdir()
    monkeypatch.setattr(eval_policy, "SERVICE_ROOT", root)
    return root


def run(project, handler, argv=None):
    stream = io.StringIO()
    path = project / "data" / "day5-session-test.json"
    code = runner().main(argv or ["--messages", '["查手机","预算改成2元"]'],
        environ={"YSHOP_AUTHORIZATION": AUTH, "LLM_API_KEY": MODEL_KEY},
        env_file=project / "unused.env", transport=httpx.MockTransport(handler),
        report_path=path, stdout=stream)
    return code, json.loads(stream.getvalue()), path


def test_continuous_local_chat_reuses_server_session_id_and_records_public_results(project):
    requests = []
    sid = "a" * 32
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={**_body(), "session_id": sid})
    code, summary, path = run(project, handler)
    assert code == 0 and summary["turn_count"] == 2
    assert all(r.method == "POST" and r.url.path == "/agent/chat" and r.url.host == "127.0.0.1" for r in requests)
    assert json.loads(requests[0].content) == {"message": "查手机"}
    assert json.loads(requests[1].content) == {"message": "预算改成2元", "session_id": sid}
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["mode"] == "test_double" and report["session_id"] == sid
    assert report["business_write_requests"] == 0 and len(report["turns"]) == 2
    assert AUTH not in path.read_text(encoding="utf-8") and MODEL_KEY not in path.read_text(encoding="utf-8")


def test_cli_can_resume_existing_session(project):
    sid = "b" * 32
    def handler(request):
        assert json.loads(request.content)["session_id"] == sid
        return httpx.Response(200, json={**_body(), "session_id": sid})
    assert run(project, handler, ["--messages", '["查看这个"]', "--session-id", sid])[0] == 0


@pytest.mark.parametrize("messages", ["not json", "[]", '[""]', "{}", "[true]", json.dumps(["x"] * 13)])
def test_invalid_turns_never_request_network(project, messages):
    def handler(request):
        pytest.fail("invalid arguments must not send request")
    assert run(project, handler, ["--messages", messages])[0] == 1


def test_changed_session_id_and_failed_http_stop_remaining_turns(project):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={**_body(), "session_id": ("a" if len(seen) == 1 else "b") * 32})
    code, summary, path = run(project, handler, ["--messages", '["查手机","继续","第三轮"]'])
    assert code == 1 and len(seen) == 2
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"


def test_existing_report_is_preserved_and_real_model_cannot_be_declared_with_fake_transport(project):
    path = project / "data" / "day5-session-test.json"
    path.parent.mkdir()
    path.write_text("frozen evidence", encoding="utf-8")
    def handler(request):
        pytest.fail("evidence rejection must happen before request")
    assert run(project, handler)[0] == 1
    assert path.read_text(encoding="utf-8") == "frozen evidence"
    assert run(project, handler, ["--messages", '["查手机"]', "--mode", "real_model"])[0] == 1


def test_public_credentials_rejected_without_leaking_response(project):
    def handler(request):
        return httpx.Response(200, json={**_body(answer=MODEL_KEY), "session_id": "a" * 32})
    code, summary, path = run(project, handler)
    assert code == 1 and MODEL_KEY not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("sid", [None, "x" * 32, "a" * 31, True])
def test_invalid_server_session_id_stops_before_next_request(project, sid):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={**_body(), "session_id": sid})
    code, summary, path = run(project, handler)
    assert code == 1 and len(seen) == 1
    assert summary["error_code"] == "chat_invalid_response"


@pytest.mark.parametrize("url", ["https://example.com", "http://127.0.0.1:8010/path", "http://localhost:8010\n"])
def test_nonlocal_or_malformed_origin_never_requests_network(project, url):
    def handler(request):
        pytest.fail("invalid origin must not send request")
    assert run(project, handler, ["--messages", '["查商品"]', "--api-url", url])[0] == 1


def test_http_session_error_retains_completed_turn_and_stops_remaining_messages(project):
    seen = []
    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(200, json={**_body(), "session_id": "a" * 32})
        return httpx.Response(409, json={"error": {"code": "session_busy", "message": MODEL_KEY}})
    code, summary, path = run(project, handler, ["--messages", '["查商品","继续","第三轮"]'])
    report = json.loads(path.read_text(encoding="utf-8"))
    assert code == 1 and len(seen) == 2 and len(report["turns"]) == 1
    assert summary["error_code"] == "session_busy"
    assert MODEL_KEY not in path.read_text(encoding="utf-8")


def test_existing_atomic_temporary_report_and_prior_day_name_are_protected(project):
    path = project / "data" / "day5-session-test.json.tmp"
    path.parent.mkdir()
    path.write_text("unrelated evidence", encoding="utf-8")
    def handler(request):
        pytest.fail("protected report must reject before request")
    assert run(project, handler)[0] == 1
    assert path.read_text(encoding="utf-8") == "unrelated evidence"
    assert run(project, handler, ["--messages", '["查商品"]', "--report", "data/day4-new.json"])[0] == 1


def test_temporary_evidence_created_after_report_reservation_is_preserved(project):
    path = project / "data" / "day5-session-test.json"
    temporary = path.with_name(path.name + ".tmp")
    def handler(request):
        temporary.write_text("foreign evidence", encoding="utf-8")
        return httpx.Response(200, json={**_body(), "session_id": "a" * 32})
    code, summary, report_path = run(project, handler, ["--messages", '["查商品"]'])
    assert code == 0 and summary["turn_count"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"
    assert temporary.read_text(encoding="utf-8") == "foreign evidence"


def test_credential_in_message_rejected_before_request_and_not_written(project):
    def handler(request):
        pytest.fail("credential text must reject before request")
    code, summary, path = run(project, handler, ["--messages", json.dumps([MODEL_KEY])])
    assert code == 1 and not path.exists() and MODEL_KEY not in json.dumps(summary)
