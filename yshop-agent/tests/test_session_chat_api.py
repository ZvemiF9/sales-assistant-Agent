"""Session HTTP contracts exercise authentication, storage, rollback and restart."""

import importlib
import importlib.util
import json
from contextlib import contextmanager
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.config import SERVICE_ENV_FILE, Settings
from app.errors import ShopError
from app.main import create_app
from app.tools.yshop import YShopClient
from tests.test_agent_graph import AUTH, PolicyBoundary, ScriptedModel, call


def session_settings(path):
    assert importlib.util.find_spec("app.storage.config"), "session settings missing"
    return importlib.import_module("app.storage.config").SessionSettings(_env_file=None, db_path=path)


@pytest.fixture
def database():
    return SERVICE_ENV_FILE.parent / ".tmp" / f"day5-test-{uuid4().hex}" / "sessions.sqlite3"


@contextmanager
def client_for(database, model, *, user_id=33, expired=False):
    app = create_app(Settings(_env_file=None), session_settings=session_settings(database))
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.method == "GET", "no shopping writes authorized"
        if request.url.path == "/api/userinfo":
            return httpx.Response(200, json={"status": 401 if expired else 200,
                "success": not expired, "data": None if expired else {"uid": user_id, "nickname": "secret-nickname"}})
        assert request.url.path == "/api/agent/products"
        return httpx.Response(200, json={"status": 200, "success": True, "data": {
            "products": [{"product_id": 7, "name": "手机", "description": "公开", "image": "",
                "skus": [{"name": "黑色", "unique_id": "sku7", "price": "1", "stock": 3, "image": ""}]}],
            "page": 1, "limit": 3, "total": 1}})

    with TestClient(app) as client:
        http = httpx.AsyncClient(base_url="http://shop.test/api/", transport=httpx.MockTransport(handler), trust_env=False)
        app.state.shop = YShopClient(http)
        app.state.policy = PolicyBoundary()
        app.state.chat_model = model
        try:
            yield client, app, paths
        finally:
            client.portal.call(http.aclose)


def post(client, message="查手机", session_id=None):
    body = {"message": message}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/agent/chat", json=body, headers={"Authorization": AUTH})


def replies():
    return [AIMessage(content="", tool_calls=[call("search_products", {"keyword": "手机", "max_price": 3})]),
            AIMessage(content="")]


def test_new_chat_returns_owned_session_and_reopened_application_recovers_context(database):
    with client_for(database, ScriptedModel(replies())) as (client, app, paths):
        first = post(client, "预算3元查手机")
        assert first.status_code == 200
        sid = first.json()["session_id"]
        assert len(sid) == 32 and int(sid, 16) >= 0
        assert len(first.json()["products"]) == 1
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("search_policy", {"question": "能退吗"})]),
                           AIMessage(content="")])
    with client_for(database, model) as (client, app, paths):
        response = post(client, "能退吗", sid)
        assert response.status_code == 200 and response.json()["session_id"] == sid
        state = app.state.sessions.load(sid, 33)
        assert state.conditions.keyword == "手机" and state.conditions.max_price == 3
        assert state.candidates[0].product_id == 7
        assert response.json()["model_rounds"] == 2
        assert response.json()["products"] == []
        assert "secret-nickname" not in str(model.seen)


def test_other_user_and_unknown_session_are_indistinguishable_and_model_not_called(database):
    with client_for(database, ScriptedModel(replies())) as (client, app, paths):
        sid = post(client).json()["session_id"]
    model = ScriptedModel([])
    with client_for(database, model, user_id=34) as (client, app, paths):
        foreign = post(client, session_id=sid)
        absent = post(client, session_id="f" * 32)
        assert foreign.status_code == absent.status_code == 404
        assert foreign.json() == absent.json()
        assert model.seen == []
        assert paths == ["/api/userinfo", "/api/userinfo"]


def test_model_failure_preserves_completed_state_and_releases_session(database):
    with client_for(database, ScriptedModel(replies())) as (client, app, paths):
        sid = post(client).json()["session_id"]
        before = app.state.sessions.load(sid, 33).model_dump_json()
        app.state.chat_model = ScriptedModel([ShopError("model_timeout")])
        failed = post(client, "预算改成2元", sid)
        assert failed.status_code == 504
        assert app.state.sessions.load(sid, 33).model_dump_json() == before
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.release(lease)


def test_busy_session_returns_409_before_model_and_invalid_identity_cannot_load(database):
    with client_for(database, ScriptedModel(replies())) as (client, app, paths):
        sid = post(client).json()["session_id"]
        lease = app.state.sessions.claim(sid, 33)
        app.state.chat_model = ScriptedModel([])
        blocked = post(client, session_id=sid)
        assert blocked.status_code == 409
        app.state.sessions.release(lease)
    with client_for(database, ScriptedModel([]), expired=True) as (client, app, paths):
        assert post(client, session_id=sid).status_code == 401


@pytest.mark.parametrize("sid", ["", "../x", "A" * 32, "f" * 31, 42, False])
def test_malformed_session_id_is_rejected_without_echoing_input(database, sid):
    with client_for(database, ScriptedModel([])) as (client, app, paths):
        response = post(client, session_id=sid)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_request", "message": "请求参数无效。"}}


def test_credentials_absent_from_persisted_snapshot_and_new_session_has_no_old_candidates(database):
    with client_for(database, ScriptedModel(replies())) as (client, app, paths):
        first = post(client, f"查手机 {AUTH}").json()
        app.state.chat_model = ScriptedModel([AIMessage(content='{"clarification":"product_type"}')])
        second = post(client, "推荐一下").json()
        assert second["session_id"] != first["session_id"] and second["products"] == []
        saved = app.state.sessions.load(first["session_id"], 33).model_dump_json()
        assert AUTH not in saved and AUTH.partition(" ")[2] not in saved
        assert "secret-nickname" not in saved


def test_identity_nickname_is_redacted_from_message_model_response_snapshot_and_logs(database, caplog):
    model = ScriptedModel(replies())
    with client_for(database, model) as (client, app, paths):
        caplog.set_level("INFO")
        response = post(client, "查手机 secret-nickname")
        assert response.status_code == 200
        saved = app.state.sessions.load(response.json()["session_id"], 33).model_dump_json()
        assert "secret-nickname" not in str(model.seen) + response.text + saved + caplog.text
        assert "[已隐藏]" in saved


def test_query_termination_rolls_back_completed_snapshot_and_releases_claim(database):
    with client_for(database, ScriptedModel(replies())) as (client, app, paths):
        sid = post(client).json()["session_id"]
        before = app.state.sessions.load(sid, 33).model_dump_json()
        class UnavailableShop:
            async def get_user(self, authorization):
                from app.schemas import UserIdentity
                return UserIdentity(user_id=33, nickname="secret-nickname")
            async def search_products(self, query, authorization):
                raise ShopError("shop_unavailable")
        app.state.shop = UnavailableShop()
        model = ScriptedModel(replies())
        app.state.chat_model = model
        failed = post(client, "预算改成2元", sid)
        assert failed.status_code == 502 and failed.json()["error"]["code"] == "shop_unavailable"
        assert len(model.seen) == 1 and app.state.sessions.load(sid, 33).model_dump_json() == before
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.release(lease)


def test_session_database_path_cannot_target_config_knowledge_or_outside_service(database):
    root = SERVICE_ENV_FILE.parent
    for path in [root / ".env", root / "knowledge" / "sessions.sqlite3", root.parent / "sessions.sqlite3",
                 root / "data" / "day4-existing.json", root / "data" / "chroma" / "chroma.sqlite3"]:
        with pytest.raises(ValueError):
            session_settings(path)
