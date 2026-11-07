"""Authenticated chat HTTP contracts and isolation of concurrent graph states."""

import asyncio
from contextlib import contextmanager
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.config import AgentSettings
from app.config import Settings
from app.storage.config import SessionSettings
from app.errors import ShopError
from app.main import create_app
from app.tools.yshop import YShopClient
from tests.test_agent_graph import AUTH, PolicyBoundary, ScriptedModel, call


@contextmanager
def chat_client(*, model=None, expired=False, cors=False):
    settings = Settings(_env_file=None, cors_origins=["http://localhost:5173"] if cors else [])
    application = create_app(settings)
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "GET", "chat may not perform shop writes"
        if request.url.path == "/api/userinfo":
            return httpx.Response(200, json={"status": 401 if expired else 200, "success": not expired,
                "data": None if expired else {"uid": 33, "nickname": "private-nickname", "password": "private-password"}})
        assert request.url.path == "/api/agent/products"
        assert request.url.params["limit"] == "3"
        return httpx.Response(200, json={"status": 200, "success": True, "data": {
            "products": [{"product_id": 7, "name": "手机", "description": "公开", "image": "phone.jpg",
                "costPrice": "private-cost", "skus": [{"name": "黑色", "unique_id": "sku-7", "price": "99.50",
                    "stock": 3, "image": "phone.jpg", "costPrice": "private-cost"}]}],
            "page": 1, "limit": 3, "total": 1}})

    with TestClient(application) as client:
        http = httpx.AsyncClient(base_url="http://shop.test/api/", transport=httpx.MockTransport(handler), trust_env=False)
        application.state.shop = YShopClient(http)
        application.state.policy = PolicyBoundary()
        if model is not None:
            application.state.chat_model = model
        try:
            yield client, application, calls
        finally:
            client.portal.call(http.aclose)


def test_chat_checks_identity_then_runs_read_only_query_and_returns_allowlisted_cards():
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("search_products", {"keyword": "手机"})]),
                           AIMessage(content="价格库存见商品卡。")])
    with chat_client(model=model) as (client, application, calls):
        response = client.post("/agent/chat", json={"message": "查手机"}, headers={"Authorization": AUTH})
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "answered" and result["model_rounds"] == 2
        assert result["products"][0]["product_id"] == 7 and result["products"][0]["skus"][0]["price"] == "99.50"
        assert [request.url.path for request in calls] == ["/api/userinfo", "/api/agent/products"]
        assert all(request.headers["Authorization"] == AUTH for request in calls)
        assert all("private" not in message.content for batch in model.seen for message in batch)
        assert "private" not in response.text and "user_id" not in response.text


@pytest.mark.parametrize("authorization", [None, "Basic bad", "Bearer", "Bearer a b"])
def test_chat_rejects_missing_or_invalid_identity_before_model_and_tools(authorization):
    model = ScriptedModel([])
    with chat_client(model=model) as (client, application, calls):
        headers = {} if authorization is None else {"Authorization": authorization}
        response = client.post("/agent/chat", json={"message": "查手机"}, headers=headers)
        assert response.status_code == 401 and response.json()["error"]["code"] == "login_required"
        assert response.headers["WWW-Authenticate"] == "Bearer" and calls == [] and model.seen == []


def test_chat_expired_identity_stops_before_model_or_retrieval():
    model = ScriptedModel([])
    with chat_client(model=model, expired=True) as (client, application, calls):
        response = client.post("/agent/chat", json={"message": "能退吗？"}, headers={"Authorization": AUTH})
        assert response.status_code == 401 and len(calls) == 1
        assert model.seen == [] and application.state.policy.calls == [] and AUTH not in response.text


@pytest.mark.parametrize("payload", [{}, {"message": " "}, {"message": "x" * 1001}, {"message": 42},
    *[{"message": "查商品", name: "private-input"} for name in ["session_id", "user_id", "token", "authorization", "tool", "tools"]]])
def test_chat_rejects_nonmessage_fields_and_sanitizes_validation_errors(payload):
    model = ScriptedModel([])
    with chat_client(model=model) as (client, application, calls):
        response = client.post("/agent/chat", json=payload, headers={"Authorization": AUTH})
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_request", "message": "请求参数无效。"}}
        assert "private-input" not in response.text and model.seen == []


def test_chat_unconfigured_model_returns_actionable_503_while_health_remains_local():
    application = create_app(Settings(_env_file=None))
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"status": 200, "success": True, "data": {"uid": 33}})

    with TestClient(application) as client:
        http = httpx.AsyncClient(base_url="http://shop.test/api/", transport=httpx.MockTransport(handler), trust_env=False)
        application.state.shop = YShopClient(http)
        # Use explicit empty configuration even when the developer has a local .env.
        from app.agent.model import OpenAIChatModel
        application.state.chat_model = OpenAIChatModel(AgentSettings(_env_file=None, llm_base_url=None,
            llm_model=None, llm_api_key=None), application.state.chat_model.http) if hasattr(application.state, "chat_model") else None
        try:
            response = client.post("/agent/chat", json={"message": "查商品"}, headers={"Authorization": AUTH})
            assert response.status_code == 503 and response.json()["error"]["code"] == "model_not_configured"
            assert all(name in response.text for name in ["LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"])
            assert client.get("/health").status_code == 200 and paths == ["/api/userinfo"]
        finally:
            client.portal.call(http.aclose)


def test_chat_model_timeout_is_safe_504_and_not_retried():
    model = ScriptedModel([ShopError("model_timeout")])
    with chat_client(model=model) as (client, application, calls):
        response = client.post("/agent/chat", json={"message": "查商品"}, headers={"Authorization": AUTH})
        assert response.status_code == 504 and response.json()["error"]["code"] == "model_timeout"
        assert len(model.seen) == len(calls) == 1 and AUTH not in response.text


def test_chat_cors_allows_post_with_authorization_and_content_type_without_cart_write():
    with chat_client(cors=True) as (client, application, calls):
        response = client.options("/agent/chat", headers={"Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Authorization, Content-Type"})
        assert response.status_code == 200
        assert "POST" in response.headers["Access-Control-Allow-Methods"]
        assert "GET" in response.headers["Access-Control-Allow-Methods"]
        assert "authorization" in response.headers["Access-Control-Allow-Headers"].lower()
        assert "content-type" in response.headers["Access-Control-Allow-Headers"].lower()
        assert client.post("/agent/cart", json={"product_id": 7}).status_code == 405 and calls == []


async def test_concurrent_chats_share_only_model_client_not_messages_candidates_or_tokens():
    from app.config import SERVICE_ENV_FILE
    application = create_app(Settings(_env_file=None), session_settings=SessionSettings(_env_file=None,
        db_path=SERVICE_ENV_FILE.parent / ".tmp" / f"day5-concurrent-{uuid4().hex}" / "sessions.sqlite3"))
    batches, authorizations = [], []
    first_rounds = 0
    both_started = asyncio.Event()

    class ConcurrentModel:
        async def ainvoke(self, messages, tools):
            nonlocal first_rounds
            batches.append(list(messages))
            human = next(message.content for message in messages if isinstance(message, HumanMessage))
            if not any(isinstance(message, ToolMessage) for message in messages):
                first_rounds += 1
                if first_rounds == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=2)
                return AIMessage(content="", tool_calls=[call("search_products", {"keyword": human})])
            return AIMessage(content="商品见卡片。")

    def handler(request):
        auth = request.headers["Authorization"]
        authorizations.append((request.url.path, auth))
        number = 7 if auth == "Bearer token-a" else 8
        data = {"uid": number, "nickname": "private-user"} if request.url.path.endswith("userinfo") else {
            "products": [{"product_id": number, "name": request.url.params["keyword"], "description": "公开",
                          "image": "", "skus": [{"name": "标准", "unique_id": f"sku-{number}",
                          "price": "10", "stock": 3, "image": ""}]}], "page": 1, "limit": 3, "total": 1}
        return httpx.Response(200, json={"status": 200, "success": True, "data": data})

    async with application.router.lifespan_context(application), httpx.AsyncClient(
            base_url="http://shop.test/api/", transport=httpx.MockTransport(handler), trust_env=False) as shop_http:
        application.state.shop = YShopClient(shop_http)
        application.state.chat_model = ConcurrentModel()
        application.state.policy = PolicyBoundary()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://agent.test") as client:
            responses = await asyncio.gather(*[client.post("/agent/chat", json={"message": message},
                headers={"Authorization": token}) for message, token in [("手机A", "Bearer token-a"), ("杯子B", "Bearer token-b")]])
    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["products"][0]["product_id"] for response in responses] == [7, 8]
    assert len(batches) == len(authorizations) == 4
    for messages in batches:
        content = str(messages)
        assert ("手机A" in content) != ("杯子B" in content)
        assert "token-a" not in content and "token-b" not in content and "private-user" not in content
    assert not hasattr(application.state, "agent") and not hasattr(application.state, "authorization")
