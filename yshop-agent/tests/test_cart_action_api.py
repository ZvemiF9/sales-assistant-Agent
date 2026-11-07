"""Real HTTP client/parser, action store and SQLite checkpoints; no live writes."""

import asyncio
from contextlib import contextmanager
from decimal import Decimal
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from app.agent.state import ConversationState
from app.config import SERVICE_ENV_FILE, Settings
from app.errors import ShopError
from app.main import create_app
from app.storage.config import SessionSettings
from app.tools.yshop import YShopClient
from tests.test_agent_graph import AUTH, PolicyBoundary, ScriptedModel, call
from tests.test_cart_intent import frozen_action_product, frozen_action_scenario, product


class ShopTransport:
    def __init__(self):
        self.calls = []
        self.user_id = 33
        self.nickname = "secret-nickname"
        self.expired = False
        self.price = "10.00"
        self.sku_name = "Black"
        self.stock = 10
        self.is_show = 1
        self.is_integral = 0
        self.eligible = True
        self.membership = True
        self.existing = 0
        self.invalid = False
        self.duplicate = False
        self.timeout = False
        self.mismatch = False
        self.delay = 0
        self.product_name = "Phone"
        self.qualification_keyword = None
        self.image = ""

    def row(self, quantity, cart_id=41):
        return {"id": cart_id, "productId": 7, "productAttrUnique": "sku7", "cartNum": quantity,
                "truePrice": self.price, "trueStock": self.stock,
                "productInfo": {"id": 7, "storeName": self.product_name, "image": self.image,
                                "attrInfo": {"productId": 7, "unique": "sku7", "sku": self.sku_name}}}

    async def __call__(self, request):
        self.calls.append(request)
        path = request.url.path
        if path == "/api/userinfo":
            return self.ok({"uid": self.user_id, "nickname": self.nickname}, expired=self.expired)
        if path == "/api/product/detail/7":
            return self.ok({"storeInfo": {"id": 7, "storeName": self.product_name, "storeInfo": "",
                "image": self.image, "isShow": self.is_show, "isIntegral": self.is_integral},
                "productValue": {"Black": {"productId": 7, "sku": self.sku_name,
                    "unique": "sku7", "price": self.price, "stock": self.stock, "image": ""}}})
        if path == "/api/agent/products":
            expected = self.product_name[:100] if self.qualification_keyword is None else self.qualification_keyword
            assert request.url.params["keyword"] == expected
            assert request.url.params["limit"] in {"3", "20"}
            item = product().model_dump(mode="json")
            if not self.membership:
                item["skus"][0]["unique_id"] = "foreign-sku"
            return self.ok({"products": [item] if self.eligible else [], "page": 1,
                            "limit": int(request.url.params["limit"]),
                            "total": 1 if self.eligible else 0})
        if path == "/api/cart/list":
            rows = [self.row(self.existing)] if self.existing else []
            if self.duplicate and rows:
                rows.append(self.row(self.existing, 42))
            return self.ok({"valid": [] if self.invalid else rows, "invalid": rows if self.invalid else []})
        if path == "/api/cart/add":
            assert request.method == "POST"
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.timeout:
                raise httpx.ReadTimeout("uncertain", request=request)
            import json
            payload = json.loads(request.content)
            assert set(payload) == {"productId", "uniqueId", "cartNum", "new", "combinationId", "secKillId"}
            assert payload["productId"] == 7 and payload["uniqueId"] == "sku7"
            self.existing += payload["cartNum"] + (1 if self.mismatch else 0)
            return self.ok({"cartId": 41})
        raise AssertionError(f"unexpected test shop path: {path}")

    @staticmethod
    def ok(data, *, expired=False):
        return httpx.Response(200, json={"status": 401 if expired else 200,
                                       "success": not expired, "data": None if expired else data})

    @property
    def writes(self):
        return [request for request in self.calls if request.method == "POST"]


class FrozenShopTransport(ShopTransport):
    def __init__(self, scenario):
        super().__init__()
        self.item = frozen_action_product(scenario)

    async def __call__(self, request):
        self.calls.append(request)
        assert request.method == "GET", "frozen preparation must not write to the shop"
        path = request.url.path
        if path == "/api/userinfo":
            return self.ok({"uid": self.user_id, "nickname": "secret-nickname"})
        if path == f"/api/product/detail/{self.item.product_id}":
            return self.ok({
                "storeInfo": {
                    "id": self.item.product_id, "storeName": self.item.name,
                    "storeInfo": self.item.description, "image": self.item.image,
                    "isShow": 1, "isIntegral": 0,
                },
                "productValue": {
                    sku.name: {
                        "productId": self.item.product_id, "sku": sku.name, "unique": sku.unique_id,
                        "price": str(sku.price), "stock": sku.stock, "image": sku.image,
                    } for sku in self.item.skus
                },
            })
        if path == "/api/agent/products":
            assert request.url.params["keyword"] in {"华为", self.item.name}
            assert request.url.params["limit"] in {"3", "20"}
            return self.ok({
                "products": [self.item.model_dump(mode="json")], "page": 1,
                "limit": int(request.url.params["limit"]), "total": 1,
            })
        if path == "/api/cart/list":
            return self.ok({"valid": [], "invalid": []})
        raise AssertionError(f"unexpected frozen test shop path: {path}")


@pytest.fixture
def database():
    return SERVICE_ENV_FILE.parent / ".tmp" / f"day6-api-{uuid4().hex}" / "sessions.sqlite3"


@contextmanager
def client_for(database, transport=None, *, raise_server_exceptions=True):
    transport = transport or ShopTransport()
    app = create_app(Settings(_env_file=None),
                     session_settings=SessionSettings(_env_file=None, db_path=database))
    with TestClient(app, raise_server_exceptions=raise_server_exceptions) as client:
        http = httpx.AsyncClient(base_url="http://shop.test/api/", transport=httpx.MockTransport(transport),
                                 trust_env=False)
        app.state.shop = YShopClient(http)
        app.state.policy = PolicyBoundary()
        app.state.chat_model = ScriptedModel([])
        try:
            yield client, app, transport
        finally:
            client.portal.call(http.aclose)


def selected_session(app):
    sid = app.state.sessions.create(33)
    lease = app.state.sessions.claim(sid, 33)
    app.state.sessions.commit(lease, ConversationState(candidates=[product()], selected_product_id=7,
                                                     selected_sku_id="sku7"))
    return sid


def prepare(client, sid, message="这个规格加入一件购物车"):
    return client.post("/agent/chat", json={"message": message, "session_id": sid},
                       headers={"Authorization": AUTH})


def decision(client, aid, value="confirm"):
    return client.post(f"/agent/actions/{aid}/decision", json={"decision": value},
                       headers={"Authorization": AUTH})


def pending(client, app):
    response = prepare(client, selected_session(app))
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "awaiting_confirmation", response.text
    return response.json()["pending_action"]["action_id"]


def test_frozen_query_then_real_http_preparation_has_exact_pending_card_without_write(database):
    scenario = frozen_action_scenario()
    target = scenario["target"]
    with client_for(database, FrozenShopTransport(scenario)) as (client, app, shop):
        app.state.chat_model = ScriptedModel([
            AIMessage(content="", tool_calls=[call(
                "search_products", {"keyword": "华为", "max_price": 3},
            )]),
            AIMessage(content=""),
        ])
        queried = client.post("/agent/chat", json={"message": scenario["query"]},
                              headers={"Authorization": AUTH})
        assert queried.status_code == 200, queried.text
        query_body = queried.json()
        assert query_body["status"] == "answered"
        assert query_body["products"] == [shop.item.model_dump(mode="json")]
        sid = query_body["session_id"]
        assert app.state.sessions.load(sid, 33).candidates == [shop.item]
        assert len(app.state.chat_model.seen) == 2

        before = time.time()
        prepared = prepare(client, sid, scenario["prepare"])
        assert prepared.status_code == 200, prepared.text
        body = prepared.json()
        assert body["status"] == "awaiting_confirmation", prepared.text
        card = body["pending_action"]
        assert card == {
            "kind": "add_to_cart", "action_id": card["action_id"], "session_id": sid,
            "product_id": target["product_id"], "product_name": shop.item.name,
            "unique_id": target["unique_id"], "sku_name": target["sku_name"],
            "quantity": target["quantity"], "unit_price": target["unit_price"],
            "total_price": target["unit_price"], "image": shop.item.image,
            "expires_at": card["expires_at"],
        }
        assert len(card["action_id"]) == 32 and int(card["action_id"], 16) >= 0
        assert card["expires_at"] > before
        assert body["action_result"] is None and body["model_rounds"] == 0
        assert app.state.sessions.get(card["action_id"], 33).card.model_dump(mode="json") == card
        state = app.state.sessions.load(sid, 33)
        assert (state.selected_product_id, state.selected_sku_id, state.pending_action_id) == (
            target["product_id"], target["unique_id"], card["action_id"],
        )
        assert shop.writes == [] and len(app.state.chat_model.seen) == 2
        assert [request.url.path for request in shop.calls] == [
            "/api/userinfo", "/api/agent/products", "/api/userinfo",
            f"/api/product/detail/{target['product_id']}", "/api/agent/products", "/api/cart/list",
        ]


@pytest.mark.parametrize("expression", [
    "一点五件", "1.5件", "负一件", "-1件", "+1件", "1e2件", "一到两件",
    "1-2件", "1/2件", "1,234件", "1，234件", "1:2件", "一两件",
    "两三件", "十二三件", "一件两件", "1件，加入2件", "0件", "10000件",
])
def test_frozen_selection_clause_malformed_quantity_stops_before_preflight(database, expression):
    scenario = frozen_action_scenario()
    with client_for(database, FrozenShopTransport(scenario)) as (client, app, shop):
        sid = app.state.sessions.create(33)
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.commit(lease, ConversationState(candidates=[shop.item]))
        message = scenario["prepare"].replace("一件", expression, 1)
        response = prepare(client, sid, message)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "needs_input", response.text
        assert body["pending_action"] is None and "数量" in body["answer"]
        assert app.state.sessions.pending(sid, 33) is None
        assert app.state.sessions.load(sid, 33).pending_action_id is None
        assert shop.writes == [] and app.state.chat_model.seen == []
        assert [request.url.path for request in shop.calls] == ["/api/userinfo"]


def test_confirmation_is_required_and_terminal_replay_has_no_preflight(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        assert not shop.writes and app.state.chat_model.seen == []
        response = decision(client, aid)
        assert response.status_code == 200, response.text
        assert response.json()["action_result"]["status"] == "succeeded"
        assert len(shop.writes) == 1 and app.state.sessions.get(aid, 33).status == "succeeded"
        before = len(shop.calls)
        assert decision(client, aid).json()["action_result"]["status"] == "succeeded"
        assert len(shop.calls) == before + 1


def test_pending_chat_does_not_authorize_write_or_rerun_model(database):
    with client_for(database) as (client, app, shop):
        sid = selected_session(app)
        first = prepare(client, sid).json()
        again = prepare(client, sid, "确认，继续加入购物车9999件")
        assert again.json()["pending_action"]["action_id"] == first["pending_action"]["action_id"]
        assert shop.writes == [] and app.state.chat_model.seen == []


@pytest.mark.parametrize("token", ["一点五", "负一", "1e2", "一到两", "1/2", "1_2"])
def test_malformed_quantity_needs_input_without_card_or_write(database, token):
    with client_for(database) as (client, app, shop):
        sid = selected_session(app)
        response = prepare(client, sid, f"这个规格加入{token}件购物车")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "needs_input", response.text
        assert body["pending_action"] is None
        assert "数量" in body["answer"]
        assert app.state.sessions.pending(sid, 33) is None
        assert app.state.sessions.load(sid, 33).pending_action_id is None
        assert not shop.writes and app.state.chat_model.seen == []
        assert [request.url.path for request in shop.calls] == ["/api/userinfo"]


@pytest.mark.parametrize("token", ["1,234", "1，234", "1:2", "1|2", "1\u20272", "一两", "两三", "十二三"])
@pytest.mark.parametrize("label_field", [None, "product", "sku"])
def test_whole_quantity_fragment_needs_input_before_preflight(database, token, label_field):
    with client_for(database) as (client, app, shop):
        item = product()
        if label_field == "product":
            item.name = token[:2]
        elif label_field == "sku":
            item.skus[0].name = token[:2]
        sid = app.state.sessions.create(33)
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.commit(lease, ConversationState(
            candidates=[item], selected_product_id=7, selected_sku_id="sku7",
        ))
        response = prepare(client, sid, f"加入{token}件购物车")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "needs_input", response.text
        assert body["pending_action"] is None
        assert "数量" in body["answer"]
        assert app.state.sessions.pending(sid, 33) is None
        assert app.state.sessions.load(sid, 33).pending_action_id is None
        assert not shop.writes and app.state.chat_model.seen == []
        assert [request.url.path for request in shop.calls] == ["/api/userinfo"]


@pytest.mark.parametrize("current_decision", ["cancel", "confirm"])
def test_explicit_second_candidate_preparation_persists_resolved_selection(database, current_decision):
    with client_for(database) as (client, app, shop):
        sid = app.state.sessions.create(33)
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.commit(lease, ConversationState(
            candidates=[product(8, "Tablet"), product()], selected_product_id=8, selected_sku_id="sku8",
        ))
        prepared = prepare(client, sid, "把第二个 Black 加入购物车一件")
        assert prepared.status_code == 200, prepared.text
        card = prepared.json()["pending_action"]
        assert (card["product_id"], card["unique_id"]) == (7, "sku7")
        state = app.state.sessions.load(sid, 33)
        assert (state.selected_product_id, state.selected_sku_id) == (7, "sku7")
        assert [item.product_id for item in state.candidates] == [8, 7]
        assert not shop.writes
        assert decision(client, card["action_id"], current_decision).status_code == 200
        state = app.state.sessions.load(sid, 33)
        assert (state.selected_product_id, state.selected_sku_id) == (7, "sku7")
        assert state.pending_action_id is None


@pytest.mark.parametrize("body", [
    '{"decision":"confirm","decision":"confirm"}',
    '{"decision":"cancel","decision":"confirm"}',
    '{"decision":"confirm","quantity":2}', '{"decision":true}', '{"decision":"CONFIRM"}',
    '{"decision":["confirm"]}', 'null', '[]', '{}', 'not-json',
])
def test_invalid_or_duplicate_decision_body_never_executes(database, body):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        response = client.post(f"/agent/actions/{aid}/decision", content=body,
            headers={"Authorization": AUTH, "Content-Type": "application/json"})
        assert response.status_code == 422
        assert not shop.writes and app.state.sessions.get(aid, 33).status == "pending"


def test_owner_unknown_foreign_and_expired_identity(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        shop.user_id = 34
        for method in ("get", "post"):
            kwargs = {"json": {"decision": "confirm"}} if method == "post" else {}
            suffix = "/decision" if method == "post" else ""
            foreign = getattr(client, method)(f"/agent/actions/{aid}{suffix}",
                                             headers={"Authorization": AUTH}, **kwargs)
            missing = getattr(client, method)(f"/agent/actions/{'f' * 32}{suffix}",
                                             headers={"Authorization": AUTH}, **kwargs)
            assert foreign.status_code == missing.status_code == 404
            assert foreign.json() == missing.json()
        shop.expired = True
        assert decision(client, aid).status_code == 401
        assert shop.writes == []


def test_cancel_persists_and_repeated_confirm_cannot_execute(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        assert decision(client, aid, "cancel").json()["status"] == "cancelled"
        assert decision(client, aid).json()["status"] == "cancelled"
        assert shop.writes == []


@pytest.mark.parametrize("field,value,error", [
    ("stock", 0, "insufficient_stock"), ("stock", 1, "insufficient_stock"),
    ("invalid", True, "cart_item_invalid"), ("duplicate", True, "cart_state_ambiguous"),
    ("is_show", 0, "unsupported_product"), ("is_show", True, "shop_invalid_response"),
    ("is_integral", 1, "unsupported_product"), ("is_integral", False, "shop_invalid_response"),
    ("eligible", False, "unsupported_product"), ("membership", False, "sku_unavailable"),
])
def test_live_preflight_failure_never_writes(database, field, value, error):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        shop.existing = 1
        setattr(shop, field, value)
        response = decision(client, aid)
        assert response.status_code == 200, response.text
        assert response.json()["action_result"]["status"] == "failed"
        assert response.json()["action_result"]["error_code"] == error
        assert not shop.writes
        assert decision(client, aid).json()["status"] == "failed"


def test_stock_includes_existing_same_sku_quantity_and_exact_write_verification(database):
    with client_for(database) as (client, app, shop):
        shop.existing = 3
        aid = pending(client, app)
        assert decision(client, aid).json()["status"] == "succeeded"
        assert shop.existing == 4 and len(shop.writes) == 1


@pytest.mark.parametrize("field,value", [("price", "12.00"), ("sku_name", "New Black")])
def test_price_or_sku_change_cancels_old_id_and_requires_new_approval(database, field, value):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        setattr(shop, field, value)
        changed = decision(client, aid)
        assert changed.status_code == 200, changed.text
        assert changed.json()["status"] == "awaiting_confirmation"
        fresh = changed.json()["pending_action"]["action_id"]
        assert fresh != aid and app.state.sessions.get(aid, 33).status == "cancelled"
        assert not shop.writes
        assert decision(client, aid).json()["status"] == "cancelled"
        assert decision(client, fresh).json()["status"] == "succeeded"
        assert len(shop.writes) == 1


@pytest.mark.parametrize("field", ["timeout", "mismatch"])
def test_uncertain_write_never_replays(database, field):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        setattr(shop, field, True)
        response = decision(client, aid)
        assert response.json()["status"] == "unknown", response.text
        assert decision(client, aid).json()["status"] == "unknown"
        assert len(shop.writes) == 1


def test_post_write_storage_failure_keeps_executing_fence_and_recovers_unknown(database, monkeypatch):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        original = app.state.sessions.finish
        def broken(*args, **kwargs):
            raise ShopError("session_unavailable")
        monkeypatch.setattr(app.state.sessions, "finish", broken)
        response = decision(client, aid)
        assert response.status_code == 503
        assert app.state.sessions.get(aid, 33).status == "executing"
        monkeypatch.setattr(app.state.sessions, "finish", original)
        assert decision(client, aid).json()["status"] == "unknown"
        assert len(shop.writes) == 1


def test_expired_card_is_terminal_and_cannot_write(database, monkeypatch):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        expires_at = app.state.sessions.get(aid, 33).card.expires_at
        import app.storage.actions as actions
        monkeypatch.setattr(actions.time, "time", lambda: expires_at + 1)
        response = decision(client, aid)
        assert response.status_code == 409 and response.json()["error"]["code"] == "action_expired"
        assert app.state.sessions.get(aid, 33).status == "failed"
        assert not shop.writes


def test_concurrent_confirm_has_one_write(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        shop.delay = 0.05
        async def invoke():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as http:
                responses = await asyncio.gather(*[
                    http.post(f"/agent/actions/{aid}/decision", json={"decision": "confirm"},
                              headers={"Authorization": AUTH}) for _ in range(2)])
                return responses
        responses = client.portal.call(invoke)
        assert sorted(response.status_code for response in responses) == [200, 409]
        assert len(shop.writes) == 1


def test_action_response_and_checkpoint_redact_current_request_secrets(database):
    with client_for(database) as (client, app, shop):
        sid = selected_session(app)
        secret_key = "private-model-key"
        app.state.chat_model.settings = SimpleNamespace(llm_api_key=SecretStr(secret_key))
        shop.product_name = f"Phone {AUTH} secret-nickname {secret_key}"
        shop.qualification_keyword = "Phone"
        shop.sku_name = f"Black {AUTH} secret-nickname {secret_key}"
        shop.image = f"https://shop.test/{AUTH.partition(' ')[2]}/{secret_key}/secret-nickname"
        prepared = prepare(client, sid, f"这个规格加入一件购物车 {AUTH} secret-nickname {secret_key}")
        assert prepared.status_code == 200, prepared.text
        aid = prepared.json()["pending_action"]["action_id"]
        response = client.get(f"/agent/actions/{aid}", headers={"Authorization": AUTH})
        assert response.status_code == 200
        from tests.test_cart_action_graph import graph
        state = client.portal.call(graph(client, app).aget_state, {"configurable": {"thread_id": aid}})
        import json
        public = prepared.text + response.text + json.dumps(state.values) + app.state.sessions.load(sid, 33).model_dump_json()
        for secret in (AUTH, AUTH.partition(" ")[2], "secret-nickname", secret_key):
            assert secret not in public
        assert "[已隐藏]" in prepared.text


def test_missing_checkpoint_persistence_keeps_health_and_read_only_chat_usable(database):
    checkpoint = database.parent / "cart-checkpoints.sqlite3"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"foreign protected content")
    with client_for(database) as (client, app, shop):
        assert client.get("/health").status_code == 200
        app.state.chat_model = ScriptedModel([
            AIMessage(content="", tool_calls=[call("search_products", {"keyword": "Phone"})]),
            AIMessage(content=""),
        ])
        queried = client.post("/agent/chat", json={"message": "查Phone"}, headers={"Authorization": AUTH})
        assert queried.status_code == 200 and queried.json()["products"][0]["product_id"] == 7
        response = prepare(client, selected_session(app))
        assert response.status_code == 503 and not shop.writes
    assert checkpoint.read_bytes() == b"foreign protected content"


def test_preparation_stock_and_cart_failures_do_not_save_action(database):
    with client_for(database) as (client, app, shop):
        sid = selected_session(app)
        shop.existing, shop.stock = 10, 10
        response = prepare(client, sid)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "insufficient_stock"
        assert app.state.sessions.pending(sid, 33) is None and not shop.writes


def test_write_unexpected_exception_is_unknown_and_not_replayed(database, monkeypatch):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        async def uncertain(*args):
            raise RuntimeError("private upstream execution detail")
        monkeypatch.setattr(app.state.shop, "add_to_cart", uncertain)
        response = decision(client, aid)
        assert response.status_code == 200 and response.json()["status"] == "unknown"
        assert "private upstream" not in response.text
        assert decision(client, aid).json()["status"] == "unknown"


def test_missing_content_type_and_oversized_decision_never_resume(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        for body, headers in [
            ('{"decision":"confirm"}', {"Authorization": AUTH}),
            (" " * 2048 + '{"decision":"confirm"}',
             {"Authorization": AUTH, "Content-Type": "application/json"}),
        ]:
            response = client.post(f"/agent/actions/{aid}/decision", content=body, headers=headers)
            assert response.status_code == 422 and app.state.sessions.get(aid, 33).status == "pending"
        assert not shop.writes
