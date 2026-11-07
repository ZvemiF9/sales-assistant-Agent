"""Day-four settings, public schema, and real HTTP wire contracts."""

import importlib.util
import inspect
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import SecretStr, ValidationError

from app.errors import ShopError


SECRET = "synthetic-model-key-do-not-log"
TOOL = {"type": "function", "function": {
    "name": "search_products", "description": "查询商品",
    "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}},
                   "additionalProperties": False},
}}


def test_optional_generation_proxy_defaults_to_none(monkeypatch):
    from app.agent.config import AgentSettings
    monkeypatch.delenv("LLM_PROXY_URL", raising=False)
    config = AgentSettings(_env_file=None)
    assert config.llm_proxy_url is None


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("   ", None),
    ("http://127.0.0.1:7892", "http://127.0.0.1:7892"),
    ("http://localhost:7892/", "http://localhost:7892"),
    ("http://[::1]:7892", "http://[::1]:7892"),
    ("http://localhost", "http://localhost"),
])
def test_proxy_accepts_only_optional_loopback_origins(value, expected):
    from app.agent.config import AgentSettings
    assert AgentSettings(_env_file=None, llm_proxy_url=value).llm_proxy_url == expected


@pytest.mark.parametrize("value", [
    1, True, "https://localhost:7892", "socks5://127.0.0.1:7892",
    "http://proxy.test:7892", "http://127.0.0.2:7892", "http://0.0.0.0:7892",
    "http://localhost.evil.test:7892", f"http://user:{SECRET}@localhost:7892",
    "http://localhost:7892?key=private", "http://localhost:7892#private",
    "http://localhost:7892/path", "http://localhost:0", "http://localhost:65536",
    "http://localhost:bad", "http://localhost:7892\\private",
    "http://localhost:7892/\x00private", " http://localhost:7892",
])
def test_proxy_rejects_invalid_origins_without_echoing_inputs(value):
    from app.agent.config import AgentSettings
    with pytest.raises(ValidationError) as caught:
        AgentSettings(_env_file=None, llm_proxy_url=value)
    assert SECRET not in str(caught.value) and "private" not in str(caught.value)


def test_proxy_explicit_environment_setting_is_separate_from_ambient_proxies(tmp_path, monkeypatch):
    from app.agent.config import AgentSettings
    monkeypatch.delenv("LLM_PROXY_URL", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://remote.invalid:7892")
    monkeypatch.setenv("HTTPS_PROXY", "http://remote.invalid:7892")
    env = tmp_path / "proxy.env"
    env.write_text("LLM_PROXY_URL=http://127.0.0.1:7892\n", encoding="utf-8")
    assert AgentSettings(_env_file=None).llm_proxy_url is None
    assert AgentSettings(_env_file=env).llm_proxy_url == "http://127.0.0.1:7892"


def test_proxy_lifespan_uses_separate_clients_and_closes_both(monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app
    assert "agent_settings" in inspect.signature(create_app).parameters, "model lifespan configuration missing"
    real_client = httpx.AsyncClient
    clients = []

    def client_factory(*args, **kwargs):
        client = real_client(*args, **kwargs)
        clients.append((client, kwargs))
        return client

    monkeypatch.setattr("app.main.httpx.AsyncClient", client_factory)
    application = create_app(Settings(_env_file=None), agent_settings=settings(llm_proxy_url="http://127.0.0.1:7892"))
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert len(clients) == 2
        shop_client, shop_kwargs = clients[0]
        model_client, model_kwargs = clients[1]
        assert shop_kwargs.get("proxy") is None
        assert model_kwargs["proxy"] == "http://127.0.0.1:7892"
        assert all(kwargs["trust_env"] is False and kwargs["follow_redirects"] is False for _, kwargs in clients)
        assert application.state.shop.http is shop_client
        assert application.state.chat_model.http is model_client
        assert model_kwargs["timeout"] == 30
    assert all(client.is_closed for client, _ in clients)


def settings(**overrides):
    from app.agent.config import AgentSettings
    return AgentSettings(_env_file=None, llm_base_url="https://model.test/v1",
                         llm_model="demo/model-v1", llm_api_key=SECRET, **overrides)


def completion(message=None, usage=None):
    result = {"id": "upstream-secret-id", "choices": [
        {"index": 0, "message": message or {"role": "assistant", "content": "找到商品。"},
         "finish_reason": "stop"}], "vendor_secret": SECRET}
    if usage is not None:
        result["usage"] = usage
    return result


async def invoke(handler, *, messages=None, tools=None, config=None):
    from app.agent.model import OpenAIChatModel
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                trust_env=False, follow_redirects=False) as http:
        return await OpenAIChatModel(config or settings(), http).ainvoke(
            messages or [HumanMessage(content="推荐手机")], [TOOL] if tools is None else tools)


def test_settings_can_be_constructed_without_model_configuration(monkeypatch):
    assert importlib.util.find_spec("app.agent") is not None, "day-four agent package is missing"
    from app.agent.config import AgentSettings
    for name in ["LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_TIMEOUT_SECONDS"]:
        monkeypatch.delenv(name, raising=False)
    config = AgentSettings(_env_file=None)
    assert config.llm_base_url is config.llm_model is config.llm_api_key is None
    assert not config.configured
    assert config.llm_timeout_seconds == 30


def test_settings_secret_is_masked_in_repr_and_serialization():
    config = settings()
    assert isinstance(config.llm_api_key, SecretStr)
    assert SECRET not in repr(config)
    assert SECRET not in config.model_dump_json()
    assert config.llm_base_url == "https://model.test/v1/"
    assert config.configured


def test_settings_read_the_explicit_service_environment_file(tmp_path, monkeypatch):
    from app.agent.config import AgentSettings
    for name in ["LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_TIMEOUT_SECONDS"]:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "synthetic.env"
    env.write_text("LLM_BASE_URL=https://model.test/v1\nLLM_MODEL=demo\n"
                   f"LLM_API_KEY={SECRET}\nLLM_TIMEOUT_SECONDS=12\nYSHOP_AUTHORIZATION=ignored\n",
                   encoding="utf-8")
    config = AgentSettings(_env_file=env)
    assert config.llm_timeout_seconds == 12
    assert config.configured
    assert not hasattr(config, "yshop_authorization")


def test_blank_optional_model_configuration_is_unconfigured(monkeypatch):
    from app.agent.config import AgentSettings
    for name in ["LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    config = AgentSettings(_env_file=None, llm_base_url="", llm_model="   ", llm_api_key="")
    assert config.llm_base_url is config.llm_model is config.llm_api_key is None
    assert not config.configured


@pytest.mark.parametrize("field,value", [
    ("llm_base_url", "ftp://model.test/v1"),
    ("llm_base_url", f"https://user:{SECRET}@model.test/v1"),
    ("llm_base_url", "https://model.test/v1?key=secret"),
    ("llm_base_url", "https://model.test/v1#fragment"),
    ("llm_base_url", "https://model.test:0/v1"),
    ("llm_base_url", "https://model.test:99999/v1"),
    ("llm_base_url", "https://model.test\\evil/v1"),
    ("llm_base_url", "https://model.test/white space"),
    ("llm_base_url", "https://model.test/\x00hidden"),
    ("llm_model", "model\r\ninjected"), ("llm_model", "x" * 257),
    ("llm_model", "model name"),
    ("llm_api_key", "secret\r\ninjected"),
    ("llm_api_key", "secret\x00injected"),
    ("llm_timeout_seconds", 0), ("llm_timeout_seconds", -1),
    ("llm_timeout_seconds", True),
    ("llm_timeout_seconds", float("nan")), ("llm_timeout_seconds", float("inf")),
])
def test_settings_reject_invalid_configuration_without_echoing_inputs(field, value):
    from app.agent.config import AgentSettings
    with pytest.raises(ValidationError) as caught:
        AgentSettings(_env_file=None, **{field: value})
    assert SECRET not in str(caught.value)
    assert "injected" not in str(caught.value)


@pytest.mark.parametrize("value", ["", "   ", 42, True, None, "x" * 1001])
def test_chat_request_rejects_blank_nonstring_and_oversized_message(value):
    from app.agent.schemas import ChatRequest
    with pytest.raises(ValidationError):
        ChatRequest(message=value)


@pytest.mark.parametrize("extra", ["session_id", "user_id", "authorization", "tools"])
def test_chat_request_only_accepts_trimmed_message(extra):
    from app.agent.schemas import ChatRequest
    assert ChatRequest(message="  查一下商品  ").message == "查一下商品"
    with pytest.raises(ValidationError):
        ChatRequest(message="查询", **{extra: SECRET})


def test_public_response_shape_has_no_session_identity_or_raw_model_fields():
    from app.agent.schemas import ChatResponse, TokenUsage, ToolTrace
    response = ChatResponse(answer="请提供预算。", status="needs_input", model_rounds=1)
    assert response.model_dump() == {
        "answer": "请提供预算。", "status": "needs_input", "products": [], "sources": [],
        "notice": None, "tool_calls": [], "model_rounds": 1,
        "usage": {"input_tokens": None, "output_tokens": None},
    }
    assert ToolTrace(name="search_products", status="succeeded", elapsed_ms=3).model_dump() == {
        "name": "search_products", "status": "succeeded", "elapsed_ms": 3}
    for data in [{"input_tokens": -1}, {"output_tokens": True}]:
        with pytest.raises(ValidationError):
            TokenUsage(**data)
    for rounds in [0, 6, True]:
        with pytest.raises(ValidationError):
            ChatResponse(answer="解释", status="answered", model_rounds=rounds)


async def test_unconfigured_model_returns_actionable_503_without_network(monkeypatch):
    from app.agent.config import AgentSettings
    for name in ["LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion())
    with pytest.raises(ShopError) as caught:
        await invoke(handler, config=AgentSettings(_env_file=None))
    assert caught.value.code == "model_not_configured"
    assert caught.value.http_status == 503
    assert requests == []


async def test_chat_completions_wire_contains_only_explicit_message_and_tool_fields():
    messages = [
        SystemMessage(content="只查询公共数据。", additional_kwargs={"secret": SECRET}),
        HumanMessage(content="预算3元", additional_kwargs={"user_id": 33}),
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "search_products",
                                          "args": {"keyword": "手机"}, "type": "tool_call"}],
                  response_metadata={"secret": SECRET}),
        ToolMessage(content='{"products": []}', tool_call_id="call_1", name="search_products",
                    additional_kwargs={"authorization": SECRET}),
    ]
    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "https://model.test/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        assert request.extensions["timeout"] == {
            "connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}
        payload = json.loads(request.content)
        assert set(payload) == {"model", "messages", "tools", "tool_choice"}
        assert payload["model"] == "demo/model-v1"
        assert payload["tools"] == [TOOL]
        assert payload["tool_choice"] == "auto"
        assert payload["messages"] == [
            {"role": "system", "content": "只查询公共数据。"},
            {"role": "user", "content": "预算3元"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "search_products", "arguments": '{"keyword":"手机"}'}}]},
            {"role": "tool", "content": '{"products": []}', "tool_call_id": "call_1"},
        ]
        assert SECRET not in request.content.decode()
        return httpx.Response(200, json=completion(usage={
            "prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20,
            "vendor_secret": SECRET}))
    reply = await invoke(handler, messages=messages)
    assert isinstance(reply, AIMessage)
    assert reply.content == "找到商品。"
    assert reply.usage_metadata == {"input_tokens": 13, "output_tokens": 7, "total_tokens": 20}
    assert reply.response_metadata == reply.additional_kwargs == {}
    assert SECRET not in repr(reply)


async def test_response_function_call_json_object_is_converted_to_ai_tool_call():
    reply = await invoke(lambda request: httpx.Response(200, json=completion({
        "role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "search_products", "arguments": '{"keyword":"华为","max_price":3}'},
             "vendor_secret": SECRET}]})))
    assert reply.content == ""
    assert reply.tool_calls == [{"id": "call_1", "name": "search_products",
                                 "args": {"keyword": "华为", "max_price": 3},
                                 "type": "tool_call"}]
    assert reply.usage_metadata is None
    assert SECRET not in repr(reply)


async def test_unknown_function_name_is_retained_for_safe_executor_rejection():
    reply = await invoke(lambda request: httpx.Response(200, json=completion({
        "role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "add_to_cart", "arguments": "{}"}}]})))
    assert reply.tool_calls[0]["name"] == "add_to_cart"


@pytest.mark.parametrize("message", [
    {"role": "assistant", "content": f"upstream echoed {SECRET}"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "search_products", "arguments": json.dumps({"keyword": SECRET})}}]},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "search_products", "arguments": json.dumps({"nested": [{SECRET: "value"}]})}}]},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": SECRET, "type": "function", "function": {
            "name": "search_products", "arguments": "{}"}}]},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": SECRET, "arguments": "{}"}}]},
])
async def test_upstream_credential_echo_never_enters_accepted_message_fields(message):
    with pytest.raises(ShopError) as caught:
        await invoke(lambda request: httpx.Response(200, json=completion(message)))
    assert caught.value.code == "model_invalid_response"
    assert SECRET not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("field", ["messages", "tools"])
async def test_outgoing_credential_echo_is_rejected_before_network(field):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion())
    arguments = {"messages": [HumanMessage(content=SECRET)]} if field == "messages" else {
        "tools": [{"type": "function", "function": {"name": "search_products", "description": SECRET}}]}
    with pytest.raises(ShopError) as caught:
        await invoke(handler, **arguments)
    assert caught.value.code == "invalid_request"
    assert requests == []
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("credential", ['synthetic-"model-key', r"synthetic-\model-key"],
                         ids=["quote", "backslash"])
@pytest.mark.parametrize("location", ["nested_value", "nested_key"])
async def test_outgoing_escaped_credential_in_structured_tool_arguments_is_rejected(credential, location):
    from app.agent.config import AgentSettings
    config = AgentSettings(_env_file=None, llm_base_url="https://model.test/v1",
                           llm_model="demo/model-v1", llm_api_key=credential)
    arguments = {"filters": [{"keyword": credential}]} if location == "nested_value" else {
        "filters": [{credential: "phone"}]}
    messages = [AIMessage(content="", tool_calls=[{
        "id": "call_1", "name": "search_products", "args": arguments, "type": "tool_call"}])]
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion())
    with pytest.raises(ShopError) as caught:
        await invoke(handler, messages=messages, config=config)
    assert caught.value.code == "invalid_request"
    assert credential not in str(caught.value)
    assert caught.value.__suppress_context__
    assert requests == []


async def test_valid_text_reply_with_nullable_tool_calls_has_no_calls():
    reply = await invoke(lambda request: httpx.Response(200, json=completion({
        "role": "assistant", "content": "请提供商品预算。", "tool_calls": None})))
    assert reply.content == "请提供商品预算。"
    assert reply.tool_calls == []
    assert reply.additional_kwargs == reply.response_metadata == {}


async def test_omits_tool_fields_when_no_tools_available():
    def handler(request):
        assert set(json.loads(request.content)) == {"model", "messages"}
        return httpx.Response(200, json=completion())
    await invoke(handler, tools=[])


async def test_redirect_response_is_rejected_even_when_injected_client_follows_redirects():
    from app.agent.model import OpenAIChatModel
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(307, headers={"Location": "https://evil.test/steal"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                trust_env=False, follow_redirects=True) as http:
        with pytest.raises(ShopError) as caught:
            await OpenAIChatModel(settings(), http).ainvoke([HumanMessage(content="查询")], [])
    assert caught.value.code == "model_rejected"
    assert len(requests) == 1


async def test_absent_usage_total_is_computed_from_known_input_and_output():
    reply = await invoke(lambda request: httpx.Response(200, json=completion(
        usage={"prompt_tokens": 4, "completion_tokens": 2})))
    assert reply.usage_metadata == {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}


@pytest.mark.parametrize("message", [
    {"role": "user", "content": "bad"}, {"role": "assistant", "content": 1},
    {"role": "assistant", "content": None},
    {"role": "assistant", "content": "", "tool_calls": "invalid"},
    {"role": "assistant", "content": "", "tool_calls": {}},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "", "type": "function", "function": {"name": "search_products", "arguments": "{}"}}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "bad\nheader", "type": "function", "function": {"name": "search_products", "arguments": "{}"}}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "x" * 129, "type": "function", "function": {"name": "search_products", "arguments": "{}"}}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "", "arguments": "{}"}}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "add to cart", "arguments": "{}"}}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "x" * 65, "arguments": "{}"}}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "other", "function": {"name": "search_products", "arguments": "{}"}}]},
    *[{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "search_products", "arguments": args}}]}
      for args in ["[]", "null", "NaN", "{", {"keyword": "手机"},
                   '{"budget":NaN}', '{"budget":Infinity}', '{"x":1,"x":2}',
                   '{"keyword":"' + "x" * 8193 + '"}']],
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "search_products", "arguments": "{}"}}] * 2},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": "search_products", "arguments": "{}"}}
        for i in range(4)]},
])
async def test_invalid_model_message_and_calls_are_safely_rejected(message):
    with pytest.raises(ShopError) as caught:
        await invoke(lambda request: httpx.Response(200, json=completion(message)))
    assert caught.value.code == "model_invalid_response"
    assert SECRET not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("payload", [[], {}, {"choices": []}, {"choices": "bad"},
    {"choices": [{"message": {"role": "assistant", "content": "a"}}] * 2},
    completion(usage={"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0}),
    completion(usage={"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2}),
    completion(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 1}),
    completion(usage={"completion_tokens": 2}), completion(usage=[]),
    completion(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": True}),
])
async def test_invalid_envelope_and_token_usage_are_safely_rejected(payload):
    with pytest.raises(ShopError) as caught:
        await invoke(lambda request: httpx.Response(200, json=payload))
    assert caught.value.code == "model_invalid_response"
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("failure,code", [("timeout", "model_timeout"),
    ("connect", "model_unavailable"), ("401", "model_rejected"),
    ("429", "model_rejected"), ("500", "model_rejected"),
    ("redirect", "model_rejected"), ("json", "model_invalid_response")])
async def test_upstream_failures_are_sanitized_and_never_retried(failure, code):
    requests = []
    def handler(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout(SECRET, request=request)
        if failure == "connect":
            raise httpx.ConnectError(SECRET, request=request)
        if failure == "redirect":
            return httpx.Response(302, headers={"Location": "https://evil.test/steal"})
        if failure == "json":
            return httpx.Response(200, content=SECRET)
        return httpx.Response(int(failure), json={"error": SECRET})
    with pytest.raises(ShopError) as caught:
        await invoke(handler)
    assert caught.value.code == code
    assert SECRET not in str(caught.value)
    assert len(requests) == 1
    assert caught.value.__suppress_context__
