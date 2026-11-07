"""Public text is private; schema keys, execution IDs and typed controls are not prose."""

import json
import time
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import SecretStr

from app.actions.schemas import CART_PATH, CartActionCard, CartActionResult
from app.agent.graph import CONVERSATION_PROMPT, QueryAgent
from app.agent.state import ConversationState, MAX_SNAPSHOT_BYTES, PublicMessage
from app.api.dependencies import RequestIdentity
from app.api.routes import action_tools
from app.errors import ShopError
from app.schemas import ProductSearchInput, UserIdentity
from app.storage.sessions import SessionStore
from tests.test_agent_graph import (
    AUTH, PolicyBoundary, ScriptedModel, ShopBoundary, call, source,
)
from tests.test_cart_action_api import ShopTransport, client_for, decision, prepare, selected_session
from tests.test_cart_action_graph import service
from tests.test_cart_intent import product


NICKNAMES = ["a", "1", "action_id", "awaiting_confirmation", "add_to_cart", "user", "sku7", "10.00"]
HIDDEN = "\u005b\u5df2\u9690\u85cf\u005d"
QUERY = "\u67e5Phone"
DETAIL = "\u8fd9\u4e2a\u8be6\u60c5"
PRIVATE_KEY = "projection-private-model-key"
TOKEN = AUTH.partition(" ")[2]


def request_tools(model, shop, nickname="secret-nickname", policy=None):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        shop=shop, policy=policy or PolicyBoundary(), chat_model=model,
    )))
    identity = RequestIdentity(AUTH, UserIdentity(user_id=33, nickname=nickname))
    return action_tools(request, identity)


def public_text(text, nickname):
    return text.replace(nickname, HIDDEN)


def snapshot(client, app, aid):
    compiled = service(app).graph(33, AUTH, request_tools(
        app.state.chat_model, app.state.shop, app.state.projection_nickname, app.state.policy,
    ))
    return client.portal.call(compiled.aget_state, {"configurable": {"thread_id": aid}})


def seed_action(app, *, terminal=None, unique_id="sku7"):
    sid = app.state.sessions.create(33)
    item = product()
    item.skus[0].unique_id = unique_id
    state = ConversationState(candidates=[item], selected_product_id=7, selected_sku_id=unique_id)
    card = CartActionCard(
        action_id="a1" * 16, session_id=sid, product_id=7, product_name="Phone",
        unique_id=unique_id, sku_name="Black", quantity=1, unit_price="10.00",
        total_price="10.00", image="", expires_at=time.time() + 899,
    )
    lease = app.state.sessions.claim(sid, 33)
    app.state.sessions.prepare(lease, 33, state, card)
    if terminal is not None:
        lease = app.state.sessions.claim(sid, 33)
        if terminal in {"succeeded", "unknown"}:
            app.state.sessions.begin_execution(lease, 33, card.action_id)
        result = CartActionResult(action_id=card.action_id, session_id=sid, status=terminal,
                                  cart_id=41 if terminal == "succeeded" else None,
                                  error_code="shop_invalid_response" if terminal == "failed" else None)
        app.state.sessions.finish(lease, 33, card.action_id, result)
    return sid, card.model_dump(mode="json")


@pytest.mark.parametrize("nickname", NICKNAMES)
def test_valid_nickname_http_query_prepare_checkpoint_reopen_and_cancel_replay(tmp_path, nickname):
    database = tmp_path / "sessions.sqlite3"
    shop = ShopTransport()
    shop.nickname = nickname
    with client_for(database, shop, raise_server_exceptions=False) as (client, app, shop):
        app.state.projection_nickname = nickname
        model = ScriptedModel([
            AIMessage(content="", tool_calls=[call(
                "search_products", {"keyword": "Phone", "max_price": "10.00"}, "call_1",
            )]), AIMessage(content=""),
        ])
        app.state.chat_model = model
        queried = client.post("/agent/chat", json={"message": QUERY}, headers={"Authorization": AUTH})
        assert queried.status_code == 200, queried.text
        body = queried.json()
        sid = body["session_id"]
        assert len(sid) == 32 and int(sid, 16) >= 0
        assert body["status"] == "answered" and body["model_rounds"] == 2
        assert body["tool_calls"][0]["name"] == "search_products"
        assert body["tool_calls"][0]["status"] == "succeeded"
        assert body["products"][0]["product_id"] == 7
        sku = body["products"][0]["skus"][0]
        assert sku["unique_id"] == "sku7" and sku["price"] == "10.00"
        assert type(sku["stock"]) is int and sku["stock"] == 10
        envelope = next(message for message in model.seen[1] if isinstance(message, AIMessage) and message.tool_calls)
        assert envelope.tool_calls == [call("search_products", {"keyword": "Phone", "max_price": "10.00"}, "call_1")]
        assert model.seen[1][-1].tool_call_id == "call_1"
        assert [row["function"]["name"] for row in model.schemas[0]] == [
            "search_products", "get_product_detail", "search_policy",
        ]

        model = ScriptedModel([
            AIMessage(content="", tool_calls=[call("get_product_detail", {"product_id": 7}, "detail_1")]),
            AIMessage(content=""),
        ])
        app.state.chat_model = model
        detailed = prepare(client, sid, DETAIL)
        assert detailed.status_code == 200, detailed.text
        assert detailed.json()["session_id"] == sid and detailed.json()["status"] == "answered"
        context = json.loads(model.seen[0][1].content.removeprefix(CONVERSATION_PROMPT))
        assert context["conditions"]["max_price"] == "10.00"
        assert context["candidates"][0]["product_id"] == 7
        assert context["candidates"][0]["skus"][0]["unique_id"] == "sku7"
        state = app.state.sessions.load(sid, 33)
        assert state.selected_product_id == 7 and state.selected_sku_id == "sku7"
        assert [item.role for item in state.history] == ["user", "assistant", "user", "assistant"]
        assert state.conditions.max_price == Decimal("10.00")

        app.state.chat_model = ScriptedModel([])
        prepared = prepare(client, sid)
        assert prepared.status_code == 200, prepared.text
        body = prepared.json()
        card = body["pending_action"]
        aid = card["action_id"]
        assert body["status"] == "awaiting_confirmation" and body["model_rounds"] == 0
        assert card == {
            "kind": "add_to_cart", "action_id": aid, "session_id": sid,
            "product_id": 7, "product_name": public_text("Phone", nickname),
            "unique_id": "sku7", "sku_name": public_text("Black", nickname), "quantity": 1,
            "unit_price": "10.00", "total_price": "10.00", "image": "", "expires_at": card["expires_at"],
        }
        assert len(aid) == 32 and int(aid, 16) >= 0
        assert type(card["quantity"]) is int and type(card["expires_at"]) is float
        assert app.state.sessions.get(aid, 33).card.model_dump(mode="json") == card
        saved = snapshot(client, app, aid)
        assert saved.values == {"action_id": aid, "card": card}
        assert saved.next == ("wait",) and saved.tasks[0].interrupts[0].value == card
        assert shop.writes == []

    with client_for(database, shop, raise_server_exceptions=False) as (client, app, shop):
        app.state.projection_nickname = nickname
        loaded = client.get(f"/agent/actions/{aid}", headers={"Authorization": AUTH})
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["pending_action"] == card
        recovered = prepare(client, sid)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["pending_action"] == card
        saved = snapshot(client, app, aid)
        assert saved.values == {"action_id": aid, "card": card} and saved.next == ("wait",)
        assert saved.tasks[0].interrupts[0].value == card
        assert app.state.sessions.load(sid, 33).pending_action_id == aid
        cancelled = decision(client, aid, "cancel")
        assert cancelled.status_code == 200, cancelled.text
        outcome = cancelled.json()["action_result"]
        assert outcome["action_id"] == aid and outcome["session_id"] == sid
        assert outcome["status"] == "cancelled" and outcome["cart_path"] == CART_PATH
        before = len(shop.calls)
        replayed = decision(client, aid)
        assert replayed.status_code == 200, replayed.text
        assert replayed.json() == cancelled.json()
        assert [request.url.path for request in shop.calls[before:]] == ["/api/userinfo"]
        assert shop.writes == [] and app.state.chat_model.seen == []


@pytest.mark.parametrize("nickname", NICKNAMES)
def test_current_nickname_cannot_rewrite_previously_saved_pending_card_on_reopen(tmp_path, nickname):
    database = tmp_path / "sessions.sqlite3"
    shop = ShopTransport()
    with client_for(database, shop) as (client, app, shop):
        sid, card = seed_action(app)
    shop.nickname = nickname
    with client_for(database, shop, raise_server_exceptions=False) as (client, app, shop):
        app.state.projection_nickname = nickname
        aid = card["action_id"]
        response = client.get(f"/agent/actions/{aid}", headers={"Authorization": AUTH})
        assert response.status_code == 200, response.text
        expected = dict(card, product_name=public_text(card["product_name"], nickname),
                        sku_name=public_text(card["sku_name"], nickname))
        assert response.json()["pending_action"] == expected
        recovered = prepare(client, sid)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["pending_action"] == expected
        saved = snapshot(client, app, aid)
        assert saved.values == {"action_id": aid, "card": expected} and saved.next == ("wait",)
        assert saved.tasks[0].interrupts[0].value == expected
        assert app.state.sessions.get(aid, 33).card.model_dump(mode="json") == card
        assert shop.writes == [] and app.state.chat_model.seen == []


@pytest.mark.parametrize("nickname", NICKNAMES)
@pytest.mark.parametrize("terminal", ["succeeded", "failed", "unknown", "cancelled"])
def test_current_nickname_preserves_saved_terminal_replay_without_preflight(tmp_path, nickname, terminal):
    database = tmp_path / "sessions.sqlite3"
    shop = ShopTransport()
    with client_for(database, shop) as (client, app, shop):
        sid, card = seed_action(app, terminal=terminal)
        result = app.state.sessions.get(card["action_id"], 33).result.model_dump(mode="json")
    shop.nickname = nickname
    with client_for(database, shop, raise_server_exceptions=False) as (client, app, shop):
        aid = card["action_id"]
        for response in [
            client.get(f"/agent/actions/{aid}", headers={"Authorization": AUTH}),
            decision(client, aid),
        ]:
            assert response.status_code == 200, response.text
            assert response.json()["session_id"] == sid and response.json()["status"] == terminal
            assert response.json()["action_result"] == result
        assert [request.url.path for request in shop.calls] == ["/api/userinfo", "/api/userinfo"]
        assert shop.writes == [] and app.state.chat_model.seen == []


@pytest.mark.parametrize("nickname", ["a", "1", "search_products", "product_id", "name", "args", "id", "type"])
@pytest.mark.parametrize("name,args", [
    ("search_products", {"keyword": "\u624b\u673a", "max_price": "1000.00"}),
    ("get_product_detail", {"product_id": 7}),
    ("search_policy", {"question": "\u80fd\u9000\u5417"}),
])
async def test_query_tool_names_ids_argument_keys_and_numeric_values_survive_nickname(nickname, name, args):
    shop, policy = ShopBoundary(), PolicyBoundary()
    call_id = "call_1_product_id_name_args_id_type_search_products"
    envelope = call(name, args, call_id)
    model = ScriptedModel([AIMessage(content="", tool_calls=[envelope]), AIMessage(content="")])
    result = await QueryAgent(model, request_tools(model, shop, nickname, policy)).run("\u67e5\u8be2")
    assert result.status == "answered"
    assert result.tool_calls[0].name == name and result.tool_calls[0].status == "succeeded"
    assert model.seen[1][-2].tool_calls == [envelope]
    assert isinstance(model.seen[1][-1], ToolMessage) and model.seen[1][-1].tool_call_id == call_id
    if name == "search_policy":
        assert result.sources[0].kind == "demo" and result.sources[0].document_id == source().document_id
    else:
        assert result.products[0].product_id == 7
        assert result.products[0].skus[0].unique_id == "sku-7"
        assert result.products[0].skus[0].price == Decimal("999.50")
        assert type(result.products[0].skus[0].stock) is int


@pytest.mark.parametrize("nickname", ["a", "clarification", "budget", "product_type", "policy_question"])
@pytest.mark.parametrize("selector,answer", [
    ("budget", "\u8bf7\u63d0\u4f9b\u9884\u7b97\u6216\u4ef7\u683c\u8303\u56f4\u3002"),
    ("product_type", "\u8bf7\u63d0\u4f9b\u5e0c\u671b\u67e5\u8be2\u7684\u5546\u54c1\u7c7b\u578b\u3002"),
    ("policy_question", "\u8bf7\u63d0\u4f9b\u5e0c\u671b\u67e5\u8be2\u7684\u6f14\u793a\u89c4\u5219\u95ee\u9898\u3002"),
])
async def test_closed_clarification_json_is_a_control_not_nickname_text(nickname, selector, answer):
    shop = ShopBoundary()
    model = ScriptedModel([AIMessage(content=json.dumps({"clarification": selector}))])
    result = await QueryAgent(model, request_tools(model, shop, nickname)).run("\u5e2e\u6211\u67e5\u8be2")
    assert result.answer == answer and result.status == "needs_input"
    assert result.model_rounds == 1 and result.tool_calls == [] and shop.calls == []


def test_private_text_and_media_stay_redacted_in_response_snapshot_and_checkpoint(tmp_path, caplog):
    database = tmp_path / "sessions.sqlite3"
    shop = QualificationTransport()
    echoes = f"{AUTH} {TOKEN} {shop.nickname} {PRIVATE_KEY}"
    shop.product_name = "Phone " + echoes
    shop.sku_name = "Black " + echoes
    shop.image = "https://shop.test/" + echoes
    with client_for(database, shop, raise_server_exceptions=False) as (client, app, shop):
        caplog.set_level("INFO")
        app.state.projection_nickname = shop.nickname
        app.state.chat_model.settings = SimpleNamespace(llm_api_key=SecretStr(PRIVATE_KEY))
        sid = app.state.sessions.create(33)
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.commit(lease, ConversationState(
            candidates=[product()], selected_product_id=7, selected_sku_id="sku7",
            history=[PublicMessage(role="user", content=echoes)],
            conditions=ProductSearchInput(keyword=PRIVATE_KEY, limit=3),
        ))
        prepared = prepare(client, sid, "\u8fd9\u4e2a\u89c4\u683c\u52a0\u5165\u4e00\u4ef6\u8d2d\u7269\u8f66 " + echoes)
        assert prepared.status_code == 200, prepared.text
        card = prepared.json()["pending_action"]
        assert card["unique_id"] == "sku7" and card["unit_price"] == "10.00"
        saved = snapshot(client, app, card["action_id"])
        public = prepared.text + json.dumps(saved.values) + app.state.sessions.load(sid, 33).model_dump_json() + caplog.text
        for secret in (AUTH, TOKEN, shop.nickname, PRIVATE_KEY):
            assert secret not in public
        assert HIDDEN in prepared.text and shop.writes == []


class ContaminatedTransport(ShopTransport):
    def __init__(self, field, secret):
        super().__init__()
        self.field, self.secret = field, secret

    async def __call__(self, request):
        response = await super().__call__(request)
        if request.url.path == "/api/agent/products":
            payload = response.json()
            item = payload["data"]["products"][0]
            if self.field == "unique_id":
                item["skus"][0]["unique_id"] = "sku7-" + self.secret
            elif self.field == "price":
                item["skus"][0]["price"] = self.secret
            return self.ok(payload["data"])
        return response


@pytest.mark.parametrize("field,secret", [
    ("unique_id", AUTH), ("unique_id", TOKEN), ("unique_id", PRIVATE_KEY), ("price", "10.00"),
])
def test_credential_in_sku_identifier_or_money_fails_closed_not_published_as_a_new_id(tmp_path, field, secret):
    shop = ContaminatedTransport(field, secret)
    with client_for(tmp_path / "sessions.sqlite3", shop, raise_server_exceptions=False) as (client, app, shop):
        sid = app.state.sessions.create(33)
        before = app.state.sessions.load(sid, 33).model_dump_json()
        model = ScriptedModel([
            AIMessage(content="", tool_calls=[call("search_products", {"keyword": "Phone"})]), AIMessage(content=""),
        ])
        model.settings = SimpleNamespace(llm_api_key=SecretStr(secret if field == "price" else PRIVATE_KEY))
        app.state.chat_model = model
        response = prepare(client, sid, QUERY)
        assert response.status_code == 502, response.text
        assert response.json()["error"]["code"] == "shop_invalid_response"
        assert secret not in response.text
        assert app.state.sessions.load(sid, 33).model_dump_json() == before
        assert app.state.sessions.pending(sid, 33) is None and shop.writes == []
        assert len(model.seen) == 1


@pytest.mark.parametrize("field", ["id", "name", "argument_key", "numeric_argument"])
@pytest.mark.parametrize("credential", [TOKEN, PRIVATE_KEY])
async def test_credential_in_model_execution_controls_rejects_entire_batch_before_any_tool(field, credential):
    shop = ShopBoundary()
    bad = call("get_product_detail", {"product_id": 7}, "detail")
    if field == "id":
        bad["id"] = "detail-" + credential
    elif field == "name":
        bad["name"] = credential
    elif field == "argument_key":
        bad["args"][credential] = 1
    else:
        credential = "7"
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("search_products", {"keyword": "\u624b\u673a"}, "safe"), bad]),
        AIMessage(content=""),
    ])
    model.settings = SimpleNamespace(llm_api_key=SecretStr(credential if field == "numeric_argument" else PRIVATE_KEY))
    with pytest.raises(ShopError) as caught:
        await QueryAgent(model, request_tools(model, shop)).run("\u67e5\u8be2")
    assert caught.value.code == "model_invalid_response"
    assert shop.calls == [] and len(model.seen) == 1
    assert credential not in str(caught.value)


@pytest.mark.parametrize("credential", ["a1" * 8, "add_to_cart", "10.00"])
def test_credential_contaminated_saved_action_control_cannot_resume_or_create_public_checkpoint(tmp_path, credential):
    shop = ShopTransport()
    with client_for(tmp_path / "sessions.sqlite3", shop, raise_server_exceptions=False) as (client, app, shop):
        sid, card = seed_action(app)
        aid = card["action_id"]
        before = app.state.sessions.load(sid, 33).model_dump_json()
        app.state.chat_model.settings = SimpleNamespace(llm_api_key=SecretStr(credential))
        app.state.projection_nickname = shop.nickname
        for response in [
            client.get(f"/agent/actions/{aid}", headers={"Authorization": AUTH}),
            decision(client, aid),
        ]:
            assert response.status_code in {422, 502, 503}, response.text
            assert set(response.json()) == {"error"}
            assert credential not in response.text
        assert app.state.sessions.get(aid, 33).card.model_dump(mode="json") == card
        assert app.state.sessions.get(aid, 33).status == "pending"
        assert app.state.sessions.load(sid, 33).model_dump_json() == before
        saved = snapshot(client, app, aid)
        assert saved.values == {} and saved.next == ()
        assert shop.writes == [] and app.state.chat_model.seen == []


@pytest.mark.parametrize("context", [
    {"unexpected": "public"},
    {"conditions": {"keyword": "Phone", "authorization": AUTH}, "candidates": [],
     "selected_product_id": None, "selected_sku_id": None},
    {"conditions": {}, "candidates": [], "selected_product_id": None, "selected_sku_id": object()},
    {"conditions": {}, "candidates": [], "selected_product_id": None, "selected_sku_id": "x" * 129},
])
async def test_unknown_or_unbounded_structured_context_fails_closed_before_model(context):
    shop = ShopBoundary()
    model = ScriptedModel([AIMessage(content='{"clarification":"product_type"}')])
    with pytest.raises(ShopError) as caught:
        await QueryAgent(model, request_tools(model, shop)).run("\u67e5\u8be2", context=context)
    assert caught.value.code == "invalid_request" and model.seen == [] and shop.calls == []


class QualificationTransport(ShopTransport):
    async def __call__(self, request):
        if request.url.path != "/api/agent/products":
            return await super().__call__(request)
        self.calls.append(request)
        assert request.method == "GET"
        item = product(name=self.product_name)
        return self.ok({"products": [item.model_dump(mode="json")], "page": 1,
                        "limit": int(request.url.params["limit"]), "total": 1})


@pytest.mark.parametrize("name,nickname", [
    pytest.param("Phone " + AUTH, "secret-nickname", id="authorization-only"),
    pytest.param("Phone " + TOKEN, "secret-nickname", id="token-only"),
    pytest.param("Phone secret-nickname", "secret-nickname", id="nickname-only"),
    pytest.param("Phone " + PRIVATE_KEY, "secret-nickname", id="model-key-only"),
    pytest.param("Phone secret-nickname " + PRIVATE_KEY + " " + TOKEN + " " + AUTH,
                 "secret-nickname", id="nickname-key-token-authorization"),
    pytest.param("Phone " + PRIVATE_KEY + " " + AUTH + " secret-nickname",
                 "secret-nickname", id="key-authorization-nickname"),
    pytest.param("Phone " + TOKEN + " secret-nickname " + PRIVATE_KEY + " " + AUTH,
                 "secret-nickname", id="token-nickname-key-authorization"),
    pytest.param(AUTH + " Phone", "secret-nickname", id="authorization-before-safe-name"),
    pytest.param(PRIVATE_KEY + " Phone", "secret-nickname", id="model-key-before-safe-name"),
    pytest.param("Phone" + PRIVATE_KEY + "Edition", "secret-nickname", id="embedded-key-no-token-separator"),
    pytest.param("Phone " + "P" * 85 + " " + AUTH, "secret-nickname", id="secret-crosses-query-length-bound"),
    pytest.param("Phone a", "a", id="short-nickname"),
])
def test_qualification_query_uses_safe_genuine_name_substring_without_private_fragments(tmp_path, caplog, name, nickname):
    shop = QualificationTransport()
    shop.product_name, shop.nickname = name, nickname
    with client_for(tmp_path / "sessions.sqlite3", shop, raise_server_exceptions=False) as (client, app, shop):
        caplog.set_level("INFO")
        app.state.chat_model.settings = SimpleNamespace(llm_api_key=SecretStr(PRIVATE_KEY))
        sid = selected_session(app)
        response = prepare(client, sid)
        assert response.status_code == 200, response.text
        catalog = [request for request in shop.calls if request.url.path == "/api/agent/products"]
        assert len(catalog) == 1
        query = catalog[0].url.params
        keyword = query["keyword"]
        assert keyword and len(keyword) <= 100 and keyword in name
        assert HIDDEN not in keyword
        assert query["limit"] == "20" and query["page"] == "1"
        assert catalog[0].headers["Authorization"] == AUTH
        for secret in (AUTH, TOKEN, nickname, PRIVATE_KEY):
            assert secret not in keyword
        for fragment in ("Bearer", "synthetic-shop", "private-token", "projection-pr", "model-key", "secret-n"):
            assert fragment not in keyword
        public = response.text + app.state.sessions.load(sid, 33).model_dump_json() + caplog.text
        for secret in (AUTH, TOKEN, PRIVATE_KEY):
            assert secret not in public
        if len(nickname) > 1:
            assert nickname not in public
        card = response.json()["pending_action"]
        assert card["product_id"] == 7 and card["unique_id"] == "sku7"
        assert card["quantity"] == 1 and card["unit_price"] == "10.00"
        assert shop.writes == [] and app.state.chat_model.seen == []


@pytest.mark.parametrize("name", [AUTH, TOKEN, "secret-nickname", PRIVATE_KEY])
def test_qualification_query_with_no_safe_name_fails_before_catalog_or_action(tmp_path, caplog, name):
    shop = QualificationTransport()
    shop.product_name = name
    with client_for(tmp_path / "sessions.sqlite3", shop, raise_server_exceptions=False) as (client, app, shop):
        caplog.set_level("INFO")
        app.state.chat_model.settings = SimpleNamespace(llm_api_key=SecretStr(PRIVATE_KEY))
        sid = selected_session(app)
        before = app.state.sessions.load(sid, 33).model_dump_json()
        response = prepare(client, sid)
        assert response.status_code == 502, response.text
        assert response.json()["error"]["code"] == "shop_invalid_response"
        assert [request.url.path for request in shop.calls] == ["/api/userinfo", "/api/product/detail/7"]
        assert app.state.sessions.load(sid, 33).model_dump_json() == before
        assert app.state.sessions.pending(sid, 33) is None
        for secret in (AUTH, TOKEN, shop.nickname, PRIVATE_KEY):
            assert secret not in response.text + caplog.text
        assert shop.writes == [] and app.state.chat_model.seen == []


def capacity_snapshot(kind, character="x"):
    if kind == "legacy":
        initial = ConversationState(candidates=[{
            "product_id": 7, "name": "phone", "description": "", "image": "", "skus": [],
        }])
    else:
        initial = ConversationState(
            history=[PublicMessage(role="user" if index % 2 == 0 else "assistant", content="h" * 4000)
                     for index in range(12)],
            conditions=ProductSearchInput(keyword="Phone", max_price="20.00", limit=3),
            candidates=[product(), product(8, "Tablet")],
            selected_product_id=7, selected_sku_id="sku7",
        )
    values = initial.model_dump(mode="json", exclude={"pending_action_id"})
    compact = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    available = MAX_SNAPSHOT_BYTES - len(compact.encode("utf-8"))
    quotient, remainder = divmod(available, len(values["candidates"]))
    width = len(character.encode("utf-8"))
    for index, candidate in enumerate(values["candidates"]):
        budget = quotient + (1 if index < remainder else 0)
        count, padding = divmod(budget, width)
        candidate["description"] = character * count + "x" * padding
    compact = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    assert len(compact.encode("utf-8")) == MAX_SNAPSHOT_BYTES == 65536
    recovered = SessionStore._decode(compact)
    assert len(SessionStore._encode(recovered).encode("utf-8")) == MAX_SNAPSHOT_BYTES
    if kind == "legacy" and character == "x":
        assert len(recovered.candidates[0].description) == 65286
    if kind == "full_history":
        assert all(len(candidate.description) < 16384 for candidate in recovered.candidates)
        assert len(json.dumps(recovered.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")) > MAX_SNAPSHOT_BYTES
    return recovered


@pytest.mark.parametrize("kind", ["legacy", "full_history"])
@pytest.mark.parametrize("character", ["x", "\u4e2d"], ids=["ascii", "utf8"])
def test_known_schema_capacity_exact_compact_snapshot_preserves_all_data_and_input(kind, character):
    state = capacity_snapshot(kind, character)
    before = state.model_dump_json()
    tools = request_tools(ScriptedModel([]), ShopBoundary())
    projected = tools.public_model(state, error_code="invalid_request")
    assert projected is not state and projected.candidates[0] is not state.candidates[0]
    assert projected.model_dump_json() == before == state.model_dump_json()
    assert SessionStore._encode(projected) == SessionStore._encode(state)
    assert len(projected.model_dump_json().encode("utf-8")) == MAX_SNAPSHOT_BYTES
    if character != "x":
        assert len(projected.model_dump_json()) < MAX_SNAPSHOT_BYTES
    if kind == "full_history":
        assert len(projected.history) == 12 and all(len(item.content) == 4000 for item in projected.history)
        assert projected.selected_product_id == 7 and projected.selected_sku_id == "sku7"
        assert projected.conditions.max_price == Decimal("20.00")
        assert projected.candidates[0].skus[0].price == Decimal("10.00")


@pytest.mark.parametrize("kind", ["legacy", "full_history"])
@pytest.mark.parametrize("character", ["x", "\u4e2d"], ids=["ascii", "utf8"])
def test_known_schema_capacity_exact_legacy_store_reaches_authenticated_normal_http_query(tmp_path, kind, character):
    state = capacity_snapshot(kind, character)
    before = state.model_dump_json()
    with client_for(tmp_path / "sessions.sqlite3", raise_server_exceptions=False) as (client, app, shop):
        assert app.state.checkpoints is not None
        sid = app.state.sessions.create(33)
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.commit(lease, state)
        assert app.state.sessions.load(sid, 33).model_dump_json() == before
        model = ScriptedModel([
            AIMessage(content="", tool_calls=[call("search_products", {"keyword": "Phone"})]),
            AIMessage(content=""),
        ])
        app.state.chat_model = model
        response = prepare(client, sid, "\u6362\u6210Phone")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] == sid and body["status"] == "answered"
        assert body["model_rounds"] == 2 and body["products"] == [product().model_dump(mode="json")]
        assert body["tool_calls"][0]["name"] == "search_products"
        assert body["tool_calls"][0]["status"] == "succeeded"
        completed = app.state.sessions.load(sid, 33)
        assert completed.conditions.keyword == "Phone" and completed.candidates == [product()]
        assert len(completed.model_dump_json().encode("utf-8")) <= MAX_SNAPSHOT_BYTES
        assert completed.pending_action_id is None and app.state.sessions.pending(sid, 33) is None
        assert [request.url.path for request in shop.calls] == ["/api/userinfo", "/api/agent/products"]
        assert all(request.headers["Authorization"] == AUTH for request in shop.calls)
        assert len(model.seen) == 2 and shop.writes == []
        assert state.model_dump_json() == before


@pytest.mark.parametrize("kind", ["legacy", "full_history"])
@pytest.mark.parametrize("error_code", ["invalid_request", "session_unavailable"])
def test_known_schema_capacity_above_compact_snapshot_limit_retains_sanitized_caller_error(kind, error_code):
    state = capacity_snapshot(kind)
    state.candidates[0].description += "x"
    before = state.model_dump_json()
    assert len(before.encode("utf-8")) == MAX_SNAPSHOT_BYTES + 1
    with pytest.raises(ShopError) as caught:
        request_tools(ScriptedModel([]), ShopBoundary()).public_model(state, error_code=error_code)
    assert caught.value.code == error_code and state.model_dump_json() == before
    assert "x" * 100 not in str(caught.value)


@pytest.mark.parametrize("error_code", ["invalid_request", "session_unavailable"])
def test_known_schema_capacity_redaction_expansion_cannot_exceed_compact_output_limit(error_code):
    state = capacity_snapshot("full_history")
    state.candidates[0].description = "a" + state.candidates[0].description[1:]
    before = state.model_dump_json()
    assert len(before.encode("utf-8")) == MAX_SNAPSHOT_BYTES
    tools = request_tools(ScriptedModel([]), ShopBoundary(), nickname="a")
    with pytest.raises(ShopError) as caught:
        tools.public_model(state, error_code=error_code)
    assert caught.value.code == error_code and state.model_dump_json() == before


def test_known_schema_capacity_long_private_display_text_is_redacted_without_truncation():
    state = capacity_snapshot("legacy")
    original = state.candidates[0].description
    state.candidates[0].description = PRIVATE_KEY + original[len(PRIVATE_KEY):]
    before = state.model_dump_json()
    model = ScriptedModel([])
    model.settings = SimpleNamespace(llm_api_key=SecretStr(PRIVATE_KEY))
    projected = request_tools(model, ShopBoundary()).public_model(state)
    assert projected.candidates[0].description == HIDDEN + original[len(PRIVATE_KEY):]
    assert PRIVATE_KEY not in projected.model_dump_json()
    assert len(projected.model_dump_json().encode("utf-8")) <= MAX_SNAPSHOT_BYTES
    assert state.model_dump_json() == before


@pytest.mark.parametrize("field", ["identifier", "sku_list"])
def test_known_schema_capacity_keeps_declared_product_field_capacities(field):
    item = product()
    if field == "identifier":
        item.skus[0].unique_id = "s" * 16385
    else:
        item.skus = [item.skus[0].model_copy(update={"unique_id": f"sku-{index}"}) for index in range(65)]
    before = item.model_dump_json()
    projected = request_tools(ScriptedModel([]), ShopBoundary()).public_model(item)
    assert projected.model_dump_json() == before == item.model_dump_json()
    assert projected.skus[0].price == Decimal("10.00")


def test_known_schema_capacity_long_identifier_still_rejects_credentials():
    item = product()
    item.skus[0].unique_id = PRIVATE_KEY + "s" * 16385
    model = ScriptedModel([])
    model.settings = SimpleNamespace(llm_api_key=SecretStr(PRIVATE_KEY))
    before = item.model_dump_json()
    with pytest.raises(ShopError) as caught:
        request_tools(model, ShopBoundary()).public_model(item, error_code="shop_invalid_response")
    assert caught.value.code == "shop_invalid_response"
    assert PRIVATE_KEY not in str(caught.value) and item.model_dump_json() == before


def capacity_nested_json():
    value = "public"
    for _ in range(14):
        value = {"child": value}
    return value


@pytest.mark.parametrize("value", [
    "x" * 16385, {"note": "x" * 16385}, list(range(65)),
    {str(index): index for index in range(65)}, capacity_nested_json(),
    object(), b"not-json", float("nan"), float("inf"), ["\u4e2d" * 8000] * 3,
], ids=["long-string", "nested-long-string", "wide-list", "wide-object", "deep",
        "foreign-object", "bytes", "nan", "infinity", "utf8-byte-limit"])
def test_unknown_control_capacity_guards_remain_strict(value):
    with pytest.raises(ValueError):
        request_tools(ScriptedModel([]), ShopBoundary()).public_control(value)


def test_unknown_free_text_capacity_guard_is_not_globally_relaxed():
    with pytest.raises(ValueError):
        request_tools(ScriptedModel([]), ShopBoundary()).public_data("x" * 16385)


def capacity_context_state(field):
    item = product()
    if field == "identifier":
        item.skus[0].unique_id = "s" * 16385
    else:
        item.skus = [item.skus[0].model_copy(update={"unique_id": f"sku-{index}"}) for index in range(65)]
    return ConversationState(
        conditions=ProductSearchInput(keyword="Phone", limit=3),
        candidates=[item], selected_product_id=7,
    )


def capacity_context(state=None):
    state = state if state is not None else ConversationState(candidates=[product()])
    return {
        "conditions": state.conditions.model_dump(mode="json"),
        "candidates": [
            {"product_id": item.product_id, "name": item.name,
             "skus": [{"unique_id": sku.unique_id, "name": sku.name} for sku in item.skus]}
            for item in state.candidates
        ],
        "selected_product_id": state.selected_product_id,
        "selected_sku_id": state.selected_sku_id,
    }


def capacity_context_boundary(character="x", field="unique_id"):
    context = capacity_context()
    sku = context["candidates"][0]["skus"][0]
    sku[field] = ""
    compact = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    available = MAX_SNAPSHOT_BYTES - len(compact.encode("utf-8"))
    count, padding = divmod(available, len(character.encode("utf-8")))
    sku[field] = character * count + "x" * padding
    compact = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    assert len(compact.encode("utf-8")) == MAX_SNAPSHOT_BYTES
    assert len(json.dumps(context, ensure_ascii=False).encode("utf-8")) > MAX_SNAPSHOT_BYTES
    return context


@pytest.mark.parametrize("field", ["identifier", "sku_list"])
def test_capacity_context_known_candidate_retains_declared_ids_and_sku_list(field):
    state = capacity_context_state(field)
    context = capacity_context(state)
    before = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    assert len(state.model_dump_json().encode("utf-8")) < MAX_SNAPSHOT_BYTES
    tools = request_tools(ScriptedModel([]), ShopBoundary(), nickname="s")
    assert tools.public_model(state).model_dump_json() == state.model_dump_json()
    projected = tools.public_context(context)
    assert projected == context
    assert projected is not context and projected["candidates"] is not context["candidates"]
    assert projected["candidates"][0]["skus"] is not context["candidates"][0]["skus"]
    assert json.dumps(context, ensure_ascii=False, separators=(",", ":")) == before
    assert projected["selected_product_id"] == 7 and projected["selected_sku_id"] is None
    if field == "identifier":
        assert projected["candidates"][0]["skus"][0]["unique_id"] == "s" * 16385
    else:
        assert [sku["unique_id"] for sku in projected["candidates"][0]["skus"]] == [
            f"sku-{index}" for index in range(65)
        ]


@pytest.mark.parametrize("field", ["identifier", "sku_list"])
def test_capacity_context_saved_candidate_reaches_authenticated_http_without_type_switch(tmp_path, field):
    state = capacity_context_state(field)
    before = state.model_dump_json()
    expected = capacity_context(state)
    with client_for(tmp_path / "sessions.sqlite3", raise_server_exceptions=False) as (client, app, shop):
        assert app.state.checkpoints is not None
        sid = app.state.sessions.create(33)
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.commit(lease, state)
        assert app.state.sessions.load(sid, 33).model_dump_json() == before
        model = ScriptedModel([
            AIMessage(content="", tool_calls=[call("search_products", {"keyword": "Phone"})]),
            AIMessage(content=""),
        ])
        app.state.chat_model = model
        response = prepare(client, sid, QUERY)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] == sid and body["status"] == "answered"
        assert body["model_rounds"] == 2 and body["products"] == [product().model_dump(mode="json")]
        assert len(model.seen) == 2
        seen_context = json.loads(model.seen[0][1].content.removeprefix(CONVERSATION_PROMPT))
        assert seen_context == expected
        assert seen_context["candidates"][0]["skus"] == expected["candidates"][0]["skus"]
        assert body["tool_calls"][0]["name"] == "search_products"
        assert body["tool_calls"][0]["status"] == "succeeded"
        assert [row["function"]["name"] for row in model.schemas[0]] == [
            "search_products", "get_product_detail", "search_policy",
        ]
        completed = app.state.sessions.load(sid, 33)
        assert completed.candidates == [product()] and completed.conditions.keyword == "Phone"
        assert completed.pending_action_id is None and app.state.sessions.pending(sid, 33) is None
        assert body["pending_action"] is None and body["action_result"] is None
        assert len(completed.model_dump_json().encode("utf-8")) <= MAX_SNAPSHOT_BYTES
        assert [request.url.path for request in shop.calls] == ["/api/userinfo", "/api/agent/products"]
        assert all(request.headers["Authorization"] == AUTH for request in shop.calls)
        assert shop.writes == [] and state.model_dump_json() == before


@pytest.mark.parametrize("character", ["x", "\u4e2d"], ids=["ascii", "utf8"])
def test_capacity_context_exact_compact_utf8_limit_preserves_input_and_full_identifier(character):
    context = capacity_context_boundary(character)
    before = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    projected = request_tools(ScriptedModel([]), ShopBoundary()).public_context(context)
    assert projected == context
    assert projected["candidates"][0]["skus"][0]["unique_id"] == context["candidates"][0]["skus"][0]["unique_id"]
    assert len(json.dumps(projected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) == MAX_SNAPSHOT_BYTES
    assert json.dumps(context, ensure_ascii=False, separators=(",", ":")) == before


def test_capacity_context_long_display_redacts_private_text_without_truncating_or_mutating():
    context = capacity_context()
    original = "x" * 20000 + f" {AUTH} {TOKEN} secret-nickname {PRIVATE_KEY}"
    context["candidates"][0]["skus"][0]["name"] = original
    before = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    model = ScriptedModel([])
    model.settings = SimpleNamespace(llm_api_key=SecretStr(PRIVATE_KEY))
    projected = request_tools(model, ShopBoundary()).public_context(context)
    assert projected["candidates"][0]["skus"][0]["name"] == "x" * 20000 + " " + " ".join([HIDDEN] * 4)
    assert projected["candidates"][0]["skus"][0]["unique_id"] == "sku7"
    for secret in (AUTH, TOKEN, "secret-nickname", PRIVATE_KEY):
        assert secret not in json.dumps(projected, ensure_ascii=False)
    assert json.dumps(context, ensure_ascii=False, separators=(",", ":")) == before


@pytest.mark.parametrize("field", ["name", "unique_id"])
async def test_capacity_context_redaction_expansion_still_rejects_oversized_output_before_model(field):
    context = capacity_context_boundary(field=field)
    context["candidates"][0]["skus"][0]["name"] = (
        "a" + context["candidates"][0]["skus"][0]["name"][1:]
    )
    before = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    assert len(before.encode("utf-8")) == MAX_SNAPSHOT_BYTES
    shop = ShopBoundary()
    model = ScriptedModel([])
    with pytest.raises(ShopError) as caught:
        await QueryAgent(model, request_tools(model, shop, nickname="a")).run(QUERY, context=context)
    assert caught.value.code == "invalid_request"
    assert json.dumps(context, ensure_ascii=False, separators=(",", ":")) == before
    assert model.seen == [] and shop.calls == []


@pytest.mark.parametrize("kind", ["one-byte-over", "utf8-over", "normalization-would-shrink"])
async def test_capacity_context_raw_compact_size_limit_rejects_before_schema_normalization_or_model(kind):
    context = capacity_context()
    if kind == "one-byte-over":
        context = capacity_context_boundary()
        context["candidates"][0]["skus"][0]["unique_id"] += "x"
    elif kind == "utf8-over":
        context["candidates"][0]["name"] = "\u4e2d" * 22000
    else:
        context["candidates"][0]["name"] = " " * MAX_SNAPSHOT_BYTES + "Phone"
    before = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    assert len(before.encode("utf-8")) > MAX_SNAPSHOT_BYTES
    shop = ShopBoundary()
    model = ScriptedModel([])
    with pytest.raises(ShopError) as caught:
        await QueryAgent(model, request_tools(model, shop)).run(QUERY, context=context)
    assert caught.value.code == "invalid_request"
    assert json.dumps(context, ensure_ascii=False, separators=(",", ":")) == before
    assert model.seen == [] and shop.calls == []


class ContextDictionary(dict):
    pass


class ContextList(list):
    pass


class ContextString(str):
    pass


@pytest.mark.parametrize("kind", [
    "root-bytes", "root-tuple", "root-object", "keyword-bytes", "candidates-tuple",
    "skus-tuple", "root-dict-subclass", "candidate-list-subclass", "name-str-subclass",
    "non-string-key", "deep", "nan", "positive-infinity", "negative-infinity",
    "unknown-root", "unknown-condition", "unknown-candidate", "unknown-sku",
    "invalid-product-id", "too-many-candidates", "long-selection", "reversed-budget",
])
async def test_capacity_context_invalid_raw_json_and_schema_fail_before_model(kind):
    context = capacity_context()
    candidate = context["candidates"][0]
    if kind == "root-bytes":
        context = b"raw-private-context"
    elif kind == "root-tuple":
        context = (context,)
    elif kind == "root-object":
        context = SimpleNamespace(**context)
    elif kind == "keyword-bytes":
        context["conditions"]["keyword"] = b"Phone"
    elif kind == "candidates-tuple":
        context["candidates"] = tuple(context["candidates"])
    elif kind == "skus-tuple":
        candidate["skus"] = tuple(candidate["skus"])
    elif kind == "root-dict-subclass":
        context = ContextDictionary(context)
    elif kind == "candidate-list-subclass":
        context["candidates"] = ContextList(context["candidates"])
    elif kind == "name-str-subclass":
        candidate["skus"][0]["name"] = ContextString("Black")
    elif kind == "non-string-key":
        context[1] = "private-unknown-field"
    elif kind == "deep":
        context["conditions"]["keyword"] = capacity_nested_json()
    elif kind in {"nan", "positive-infinity", "negative-infinity"}:
        context["conditions"]["max_price"] = {
            "nan": float("nan"), "positive-infinity": float("inf"), "negative-infinity": -float("inf"),
        }[kind]
    elif kind == "unknown-root":
        context["authorization"] = AUTH
    elif kind == "unknown-condition":
        context["conditions"]["authorization"] = AUTH
    elif kind == "unknown-candidate":
        candidate["authorization"] = AUTH
    elif kind == "unknown-sku":
        candidate["skus"][0]["authorization"] = AUTH
    elif kind == "invalid-product-id":
        candidate["product_id"] = True
    elif kind == "too-many-candidates":
        context["candidates"] *= 4
    elif kind == "long-selection":
        context["selected_sku_id"] = "s" * 129
    else:
        context["conditions"].update(min_price="20.00", max_price="10.00")
    before = repr(context)
    shop = ShopBoundary()
    model = ScriptedModel([])
    with pytest.raises(ShopError) as caught:
        await QueryAgent(model, request_tools(model, shop)).run(QUERY, context=context)
    assert caught.value.code == "invalid_request"
    assert AUTH not in str(caught.value) and "raw-private-context" not in str(caught.value)
    assert repr(context) == before and model.seen == [] and shop.calls == []


@pytest.mark.parametrize("credential", [AUTH, TOKEN, PRIVATE_KEY])
@pytest.mark.parametrize("field", ["candidate-id", "selected-id"])
async def test_capacity_context_credential_contaminated_controls_reject_before_model(field, credential):
    context = capacity_context(capacity_context_state("identifier"))
    if field == "candidate-id":
        context["candidates"][0]["skus"][0]["unique_id"] += credential
    else:
        context["selected_sku_id"] = "sku7-" + credential
    before = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    shop = ShopBoundary()
    model = ScriptedModel([])
    model.settings = SimpleNamespace(llm_api_key=SecretStr(PRIVATE_KEY))
    with pytest.raises(ShopError) as caught:
        await QueryAgent(model, request_tools(model, shop)).run(QUERY, context=context)
    assert caught.value.code == "invalid_request" and credential not in str(caught.value)
    assert json.dumps(context, ensure_ascii=False, separators=(",", ":")) == before
    assert model.seen == [] and shop.calls == []
