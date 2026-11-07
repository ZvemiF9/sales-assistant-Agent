"""Request-scoped model tool schemas stay aligned with conversation execution policy."""

import json
from decimal import Decimal

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agent.state import ConversationState
from app.agent.tools import QueryTools
from tests.test_agent_graph import AUTH, PolicyBoundary, ScriptedModel, ShopBoundary, call, product
from tests.test_agent_model import completion, settings
from tests.test_conversation import conversation, prior


ALL_QUERY_TOOLS = {"search_products", "get_product_detail", "search_policy"}


def _names(schemas):
    return {schema["function"]["name"] for schema in schemas}


def _tool_reply(name, args, call_id="call_1"):
    return completion({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    })


async def test_lower_bound_http_round_stops_advertising_forbidden_detail_after_search():
    from app.agent.model import OpenAIChatModel

    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        names = _names(payload["tools"])
        if len(requests) == 1:
            assert names == {"search_products", "search_policy"}
            return httpx.Response(200, json=_tool_reply(
                "search_products", {"keyword": "华为", "min_price": 3, "limit": 3}))
        if "get_product_detail" in names:
            return httpx.Response(200, json=_tool_reply(
                "get_product_detail", {"product_id": 7}, f"call_{len(requests)}"))
        return httpx.Response(200, json=completion({"role": "assistant", "content": ""}))

    shop = ShopBoundary()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as http:
        model = OpenAIChatModel(settings(llm_enable_thinking=False), http)
        response, updated = await conversation(model, shop).run(
            "预算至少3元查华为", ConversationState())

    assert len(requests) == response.model_rounds == 2
    assert [_names(payload["tools"]) for payload in requests] == [
        {"search_products", "search_policy"}, {"search_products", "search_policy"}]
    assert [(trace.name, trace.status) for trace in response.tool_calls] == [
        ("search_products", "succeeded")]
    assert [entry[0] for entry in shop.calls] == ["search_products"]
    assert shop.calls[0][1].keyword == updated.conditions.keyword == "华为"
    assert shop.calls[0][1].min_price == updated.conditions.min_price == Decimal("3")
    assert response.notice is None and "部分查询未能完成" not in response.answer


@pytest.mark.parametrize(("message", "state", "tool_name", "args", "expected"), [
    ("预算至少3元查华为", ConversationState(), "search_products",
     {"keyword": "华为", "min_price": 3}, {"search_products", "search_policy"}),
    ("预算改成1000元", prior(), "search_products",
     {"keyword": "手机", "max_price": 1000}, {"search_products", "search_policy"}),
    ("换成杯子", prior(), "search_products",
     {"keyword": "杯子"}, {"search_products", "search_policy"}),
    ("看看第二个详情", prior(), "get_product_detail",
     {"product_id": 999}, {"get_product_detail", "search_policy"}),
    ("退货规则是什么？", prior(), "search_policy",
     {"question": "退货规则是什么？"}, {"search_policy"}),
    ("查询华为商品，优惠券怎么用？", ConversationState(), "search_products",
     {"keyword": "华为"}, ALL_QUERY_TOOLS),
    ("查询华为商品", ConversationState(), "search_products",
     {"keyword": "华为"}, ALL_QUERY_TOOLS),
])
async def test_every_conversation_model_round_offers_only_eligible_tools(
        message, state, tool_name, args, expected):
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call(tool_name, args)]),
        AIMessage(content=""),
    ])

    response, _ = await conversation(model).run(message, state)

    assert len(model.schemas) == response.model_rounds == 2
    assert [_names(schemas) for schemas in model.schemas] == [expected, expected]
    assert model.replies == []


async def test_ordinary_query_agent_and_query_tools_keep_all_public_schemas():
    from app.agent.graph import QueryAgent

    tools = QueryTools(ShopBoundary(), PolicyBoundary(), AUTH)
    model = ScriptedModel([AIMessage(content="")])

    response = await QueryAgent(model, tools).run("查询华为商品")

    assert response.model_rounds == 1 and _names(model.schemas[0]) == ALL_QUERY_TOOLS
    assert _names(tools.function_schemas()) == ALL_QUERY_TOOLS


async def test_forbidden_detail_call_during_budget_turn_still_fails_without_shop_detail():
    shop = ShopBoundary()
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("get_product_detail", {"product_id": 7})]),
        AIMessage(content=""),
    ])

    response, updated = await conversation(model, shop).run(
        "预算至少3元查华为", ConversationState())

    assert [(trace.name, trace.status) for trace in response.tool_calls] == [
        ("get_product_detail", "failed")]
    assert shop.calls == [] and updated.conditions.min_price == Decimal("3")
    assert "部分查询未能完成" in response.answer
    assert len(model.seen) == response.model_rounds == 2


async def test_model_schema_mutation_cannot_contaminate_rounds_or_conversation_requests():
    class MutatingModel:
        def __init__(self):
            self.names = []
            self.calls = 0

        async def ainvoke(self, messages, tools):
            self.names.append(_names(tools))
            tools[0]["function"]["name"] = "mutated_by_model"
            tools.clear()
            self.calls += 1
            if self.calls in {1, 3}:
                return AIMessage(content="", tool_calls=[call("search_products", {"keyword": "华为"})])
            return AIMessage(content="")

    model = MutatingModel()
    runner = conversation(model)

    first, _ = await runner.run("查询华为商品", ConversationState())
    second, _ = await runner.run("查询华为商品", ConversationState())

    assert model.names == [ALL_QUERY_TOOLS] * 4
    assert model.calls == first.model_rounds + second.model_rounds == 4
    assert [(trace.name, trace.status) for trace in first.tool_calls] == [
        ("search_products", "succeeded")]
    assert [(trace.name, trace.status) for trace in second.tool_calls] == [
        ("search_products", "succeeded")]
