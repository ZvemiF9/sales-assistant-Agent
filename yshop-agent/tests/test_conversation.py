"""Conversation behavior exercised through the real graph and query executor."""

import importlib.util
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import SecretStr

from app.agent.state import ConversationState, PublicMessage
from app.agent.tools import QueryTools
from app.errors import ShopError
from app.schemas import ProductSearchInput
from tests.test_agent_graph import AUTH, PolicyBoundary, ScriptedModel, ShopBoundary, call, product, source


def conversation(model, shop=None, policy=None):
    assert importlib.util.find_spec("app.agent.conversation"), "conversation engine missing"
    from app.agent.conversation import ConversationAgent
    return ConversationAgent(model, QueryTools(shop or ShopBoundary(), policy or PolicyBoundary(), AUTH))


def prior(**changes):
    values = dict(conditions=ProductSearchInput(keyword="手机", category_id=4, max_price="1500", limit=3),
                  candidates=[product(7), product(8), product(9)], selected_product_id=8,
                  selected_sku_id="sku-8")
    values.update(changes)
    return ConversationState(**values)


def tool_then_finish(name="search_products", args=None):
    return ScriptedModel([AIMessage(content="", tool_calls=[call(name, args)]),
                          AIMessage(content="已经下单，并保证永久包退。")])


@pytest.mark.parametrize("message,bounds", [
    ("预算改成1000元", (None, Decimal("1000"))),
    ("预算改成两元", (None, Decimal("2"))),
    ("预算改成二十到五十元", (Decimal("20"), Decimal("50"))),
    ("预算调整到1000-1200元", (Decimal("1000"), Decimal("1200"))),
    ("预算至少800元", (Decimal("800"), None)),
    ("取消预算限制", (None, None)),
])
async def test_budget_patch_controls_actual_query_and_preserves_product_type(message, bounds):
    shop, state = ShopBoundary(), prior()
    snapshot = state.model_dump_json()
    # Even a model which supplies stale bounds must execute the explicit new budget.
    model = tool_then_finish(args={"keyword": "", "max_price": 1500})
    response, updated = await conversation(model, shop).run(message, state)
    query = shop.calls[0][1]
    assert query.keyword == updated.conditions.keyword == "手机"
    assert query.category_id == updated.conditions.category_id == 4
    assert (query.min_price, query.max_price) == bounds
    assert (updated.conditions.min_price, updated.conditions.max_price) == bounds
    assert query.limit == 3 and updated.candidates == response.products == [product(7)]
    assert updated.selected_product_id is None and updated.selected_sku_id is None
    assert state.model_dump_json() == snapshot
    assert "已经下单" not in response.answer and updated.history[-1].content == response.answer


async def test_explicit_budget_still_queries_when_model_only_clarifies():
    shop = ShopBoundary()
    model = ScriptedModel([AIMessage(content='{"clarification":"budget"}'), AIMessage(content="")])
    result, state = await conversation(model, shop).run("预算改成200元", prior())
    assert shop.calls[0][1].max_price == state.conditions.max_price == 200
    assert result.model_rounds == 2 and result.tool_calls[0].name == "search_products"


@pytest.mark.parametrize("message,minimum,maximum", [
    ("预算改成2元", None, "2"),
    ("预算改成1到2元", "1", "2"),
])
async def test_forced_budget_search_can_be_serialized_for_the_next_real_model_request(message, minimum, maximum):
    import json
    import httpx
    from app.agent.model import OpenAIChatModel
    from tests.test_agent_model import completion, settings

    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(200, json=completion({"role": "assistant", "content": '{"clarification":"budget"}'}))
        assistant = next(item for item in payload["messages"] if item.get("tool_calls"))
        forced = assistant["tool_calls"][0]["function"]
        arguments = json.loads(forced["arguments"])
        assert forced["name"] == "search_products"
        assert arguments["min_price"] == minimum and arguments["max_price"] == maximum
        assert payload["messages"][-1]["role"] == "tool"
        return httpx.Response(200, json=completion({"role": "assistant", "content": ""}))

    shop = ShopBoundary()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as http:
        model = OpenAIChatModel(settings(llm_enable_thinking=False), http)
        response, updated = await conversation(model, shop).run(message, prior())
    assert len(requests) == response.model_rounds == 2
    assert response.status == "answered" and response.products == updated.candidates
    assert shop.calls[0][1].max_price == updated.conditions.max_price == Decimal(maximum)
    assert updated.conditions.min_price == (None if minimum is None else Decimal(minimum))


async def test_explicit_type_switch_resets_incompatible_filters():
    shop = ShopBoundary()
    model = tool_then_finish(args={"keyword": "手机", "category_id": 4, "max_price": 1500})
    _, state = await conversation(model, shop).run("换成杯子", prior())
    query = shop.calls[0][1]
    assert query.keyword == state.conditions.keyword == "杯子"
    assert query.category_id is None and query.min_price is None and query.max_price is None
    assert state.selected_product_id is None and state.selected_sku_id is None


@pytest.mark.parametrize("message", [
    "预算200元，换成杯子",
    "换成杯子，预算200元",
    "预算改成两百元，换成杯子",
    "预算200元，改成杯子",
])
async def test_budget_and_type_switch_apply_as_independent_clauses_in_either_order(message):
    shop = ShopBoundary()
    model = tool_then_finish(args={"keyword": "杯子", "max_price": 200})
    _, state = await conversation(model, shop).run(message, prior())
    query = shop.calls[0][1]
    assert query.keyword == state.conditions.keyword == "杯子"
    assert query.category_id is None and query.min_price is None
    assert query.max_price == state.conditions.max_price == 200


@pytest.mark.parametrize("message", ["看看第二个详情", "第2个库存是多少"])
async def test_ordinal_reference_reads_fresh_detail_and_replaces_displayed_order(message):
    shop = ShopBoundary()
    model = tool_then_finish("get_product_detail", {"product_id": 999})
    response, state = await conversation(model, shop).run(message, prior())
    assert shop.calls == [("get_product_detail", 8, AUTH)]
    assert response.products == state.candidates == [product(8)]
    assert state.selected_product_id == 8 and state.selected_sku_id == "sku-8"


async def test_this_reference_uses_retained_selection_and_refreshes_detail():
    shop = ShopBoundary()
    result, state = await conversation(tool_then_finish("get_product_detail", {"product_id": 7}), shop).run(
        "这个现在多少钱", prior())
    assert shop.calls[0][1] == 8 and result.products == [product(8)] and state.selected_product_id == 8


@pytest.mark.parametrize("message,state", [
    ("第二个详情", ConversationState()),
    ("第四个详情", prior()),
    ("这个的详情", prior(selected_product_id=None, selected_sku_id=None)),
    ("第二个和第三个的详情", prior()),
])
async def test_missing_or_ambiguous_reference_clarifies_without_guessing(message, state):
    shop = ShopBoundary()
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("get_product_detail", {"product_id": 7})])])
    response, updated = await conversation(model, shop).run(message, state)
    assert response.status == "needs_input" and "选择" in response.answer
    assert shop.calls == [] and response.products == [] and response.tool_calls == []
    assert updated.candidates == state.candidates and updated.selected_product_id == state.selected_product_id


async def test_rule_interlude_exposes_only_current_evidence_and_preserves_selection():
    policy, shop, state = PolicyBoundary(), ShopBoundary(), prior()
    response, updated = await conversation(tool_then_finish("search_policy", {"question": "退货规则"}),
                                          shop, policy).run("退货规则是什么？", state)
    assert response.products == [] and response.sources == [source()]
    assert updated.candidates == state.candidates and updated.selected_product_id == 8
    no_evidence = PolicyBoundary(evidence=False)
    response2, _ = await conversation(tool_then_finish("search_policy", {"question": "运费"}),
                                      shop, no_evidence).run("运费规则是什么？", updated)
    assert response2.products == [] and response2.sources == [] and response2.status == "no_evidence"
    assert source().text not in response2.answer and shop.calls == []


@pytest.mark.parametrize("question", [
    "会员生日礼券每年发多少元？",
    "会员积分兑换礼券的比例是多少？",
])
async def test_policy_only_query_preserves_requested_fact_when_model_broadens_question(question):
    class SpecificPolicy(PolicyBoundary):
        def answer(self, received):
            # The exact requested fact is absent; a broad coupon query has evidence.
            self.evidence = received != question
            return super().answer(received)

    policy, shop, state = SpecificPolicy(), ShopBoundary(), prior()
    model = tool_then_finish("search_policy", {"question": "优惠券使用条件"})
    response, updated = await conversation(model, shop, policy).run(question, state)
    assert policy.calls == [question]
    assert response.status == "no_evidence" and response.sources == []
    assert response.products == [] and shop.calls == []
    assert updated.candidates == state.candidates
    assert updated.selected_product_id == 8 and updated.selected_sku_id == "sku-8"


@pytest.mark.parametrize("first_reply", ["", '{"clarification":"policy_question"}'])
async def test_explicit_policy_interlude_still_retrieves_when_model_omits_tool(first_reply):
    policy, shop, state = PolicyBoundary(), ShopBoundary(), prior()
    model = ScriptedModel([
        AIMessage(content=first_reply), AIMessage(content=""),
        AIMessage(content="", tool_calls=[call("get_product_detail", {"product_id": 999})]),
        AIMessage(content=""),
    ])
    runner = conversation(model, shop, policy)
    question = "七天无理由退货有什么条件？"
    response, updated = await runner.run(question, state)
    assert policy.calls == [question] and shop.calls == []
    assert response.status == "answered" and response.sources == [source()]
    assert response.products == [] and response.model_rounds == 2
    assert [(t.name, t.status) for t in response.tool_calls] == [("search_policy", "succeeded")]
    assert updated.candidates == state.candidates and updated.selected_product_id == 8
    detail, completed = await runner.run("选中的商品现在库存和价格是什么？", updated)
    assert shop.calls == [("get_product_detail", 8, AUTH)]
    assert detail.products == completed.candidates == [product(8)]


@pytest.mark.parametrize("args", [
    {}, {"question": " "}, {"question": 7},
    {"question": "优惠券规则", "unexpected": "extra"},
])
async def test_policy_grounding_does_not_repair_invalid_model_arguments(args):
    policy, shop, state = PolicyBoundary(), ShopBoundary(), prior()
    response, updated = await conversation(
        tool_then_finish("search_policy", args), shop, policy
    ).run("会员生日礼券每年发多少元？", state)
    assert policy.calls == [] and shop.calls == [] and response.status == "needs_input"
    assert [(t.name, t.status) for t in response.tool_calls] == [("search_policy", "failed")]
    assert response.sources == [] and updated.candidates == state.candidates
    assert updated.selected_product_id == 8 and updated.selected_sku_id == "sku-8"


async def test_unspecified_policy_topic_keeps_model_clarification_without_forced_retrieval():
    policy, shop = PolicyBoundary(), ShopBoundary()
    model = ScriptedModel([AIMessage(content='{"clarification":"policy_question"}')])
    response, _ = await conversation(model, shop, policy).run("规则", prior())
    assert response.status == "needs_input" and response.model_rounds == 1
    assert response.tool_calls == [] and policy.calls == [] and shop.calls == []


@pytest.mark.parametrize("search_clause,policy_question", [
    ("查询华为商品", "优惠券怎么用？"),
    ("帮我找华为", "会员礼券有什么规则？"),
    ("帮我查一下华为", "优惠券怎么用？"),
])
async def test_product_search_and_voucher_question_preserve_both_requested_tools(search_clause, policy_question):
    policy, shop = PolicyBoundary(), ShopBoundary()
    model = ScriptedModel([AIMessage(content="", tool_calls=[
        call("search_products", {"keyword": "华为"}, "products"),
        call("search_policy", {"question": policy_question}, "policy"),
    ]), AIMessage(content="")])
    response, updated = await conversation(model, shop, policy).run(
        search_clause + "，" + policy_question, ConversationState()
    )
    assert len(shop.calls) == 1 and shop.calls[0][1].keyword == "华为"
    assert policy.calls == [policy_question]
    assert response.status == "answered" and response.products == updated.candidates == [product(7)]
    assert response.sources == [source()]
    assert [(t.name, t.status) for t in response.tool_calls] == [
        ("search_products", "succeeded"), ("search_policy", "succeeded")]


@pytest.mark.parametrize("preface", ["帮我查一下", "麻烦查询下吧", "请查一查"])
async def test_targetless_search_preface_keeps_explicit_policy_question(preface):
    policy, shop, state = PolicyBoundary(), ShopBoundary(), prior()
    model = ScriptedModel([AIMessage(content='{"clarification":"policy_question"}'), AIMessage(content="")])
    question = preface + "，七天无理由退货有什么条件？"
    response, updated = await conversation(model, shop, policy).run(question, state)
    assert policy.calls == [question] and shop.calls == []
    assert response.status == "answered" and response.sources == [source()]
    assert response.model_rounds == 2 and response.products == []
    assert updated.candidates == state.candidates and updated.selected_product_id == 8


async def test_short_pronoun_policy_question_uses_policy_and_preserves_selection():
    policy, shop, state = PolicyBoundary(), ShopBoundary(), prior()
    response, updated = await conversation(tool_then_finish("search_policy", {"question": "这个能退吗"}),
                                           shop, policy).run("这个能退吗", state)
    assert policy.calls == ["这个能退吗"] and shop.calls == []
    assert response.sources == [source()] and response.products == []
    assert updated.candidates == state.candidates
    assert updated.selected_product_id == 8 and updated.selected_sku_id == "sku-8"


async def test_referenced_return_shipping_cost_is_policy_only_and_preserves_selection():
    policy, shop, state = PolicyBoundary(), ShopBoundary(), prior()
    response, updated = await conversation(
        tool_then_finish("search_policy", {"question": "这个退货运费多少钱？"}), shop, policy
    ).run("这个退货运费多少钱？", state)
    assert policy.calls == ["这个退货运费多少钱？"] and shop.calls == []
    assert response.sources == [source()] and response.products == []
    assert updated.candidates == state.candidates
    assert updated.selected_product_id == 8 and updated.selected_sku_id == "sku-8"


@pytest.mark.parametrize("message", [
    "这个商品现在多少钱，退货运费怎么收？",
    "这个退货运费多少钱还有商品多少钱？",
])
async def test_referenced_product_price_and_return_shipping_cost_runs_detail_and_policy(message):
    policy, shop, state = PolicyBoundary(), ShopBoundary(), prior()
    model = ScriptedModel([AIMessage(content="", tool_calls=[
        call("get_product_detail", {"product_id": 999}, "detail"),
        call("search_policy", {"question": "这个退货运费怎么收？"}, "policy"),
    ]), AIMessage(content="")])
    response, updated = await conversation(model, shop, policy).run(message, state)
    assert shop.calls == [("get_product_detail", 8, AUTH)]
    assert policy.calls == ["这个退货运费怎么收？"]
    assert response.products == [product(8)] and response.sources == [source()]
    assert updated.candidates == [product(8)] and updated.selected_product_id == 8


async def test_short_pronoun_policy_question_rejects_unrequested_product_call_without_losing_selection():
    shop, state = ShopBoundary(), prior()
    response, updated = await conversation(tool_then_finish("search_products", {"keyword": "杯子"}), shop).run(
        "这个能退吗", state)
    assert shop.calls == [] and response.status == "needs_input"
    assert updated.candidates == state.candidates
    assert updated.selected_product_id == 8 and updated.selected_sku_id == "sku-8"


async def test_empty_new_query_invalidates_obsolete_candidates():
    shop = ShopBoundary()
    shop.search_ids = [[]]
    response, updated = await conversation(tool_then_finish(args={}), shop).run("预算改成1元", prior())
    assert response.products == updated.candidates == []
    assert updated.selected_product_id is None and updated.selected_sku_id is None
    assert updated.conditions.max_price == 1


async def test_each_message_has_fresh_counters_traces_and_no_unfinished_tool_history():
    shop = ShopBoundary()
    replies = [AIMessage(content="", tool_calls=[call("search_products", {"keyword": "手机"})]),
               AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
               AIMessage(content="", tool_calls=[call("search_products", {})]),
               AIMessage(content="", usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4})]
    model = ScriptedModel(replies)
    runner = conversation(model, shop)
    first, state = await runner.run("查手机", ConversationState())
    second, state2 = await runner.run("预算改成100元", state)
    assert first.model_rounds == second.model_rounds == 2
    assert len(second.tool_calls) == 1 and second.usage.input_tokens == 3
    assert len(state2.history) == 4
    assert not any(isinstance(item, ToolMessage) for item in model.seen[2])
    assert not any(isinstance(item, AIMessage) and item.tool_calls for item in model.seen[2])


async def test_model_failure_keeps_input_snapshot_unchanged():
    model, state = ScriptedModel([ShopError("model_timeout")]), prior()
    snapshot = state.model_dump_json()
    with pytest.raises(ShopError, match="超时"):
        await conversation(model).run("预算改成2元", state)
    assert state.model_dump_json() == snapshot


async def test_current_auth_and_model_key_are_removed_from_all_public_context_and_snapshot():
    key = "synthetic-current-model-secret"
    model = ScriptedModel([AIMessage(content="业务已经完成。")])
    model.settings = type("Settings", (), {"llm_api_key": SecretStr(key)})()
    state = prior(history=[PublicMessage(role="user", content=AUTH + " " + key)],
                  conditions=ProductSearchInput(keyword=key, limit=3))
    result, updated = await conversation(model).run("查询 " + AUTH + " " + key, state)
    public = str(model.seen) + updated.model_dump_json() + result.model_dump_json()
    assert key not in public and AUTH not in public and AUTH.partition(" ")[2] not in public


async def test_new_session_does_not_inherit_other_sessions_conditions_or_selection():
    model = tool_then_finish(args={"keyword": "杯子"})
    shop = ShopBoundary()
    _, state = await conversation(model, shop).run("查杯子", ConversationState())
    assert state.conditions.keyword == "杯子" and state.conditions.max_price is None
    assert state.selected_product_id is None


async def test_history_remains_bounded_to_completed_public_turns():
    state = ConversationState(history=[PublicMessage(role="user" if index % 2 == 0 else "assistant",
                                                     content=f"公开{index}") for index in range(12)])
    _, updated = await conversation(ScriptedModel([AIMessage(content="自由业务声明")])).run("继续", state)
    assert len(updated.history) == 12 and updated.history[0].content == "公开2"
    assert all(isinstance(item, PublicMessage) for item in updated.history)


@pytest.mark.parametrize("message", ["第二个价格是多少", "第2个的价格", "这个价格是1500元吗？"])
async def test_reference_price_question_does_not_turn_ordinal_or_quoted_price_into_budget(message):
    shop = ShopBoundary()
    _, state = await conversation(tool_then_finish("get_product_detail", {"product_id": 8}), shop).run(message, prior())
    assert shop.calls == [("get_product_detail", 8, AUTH)]
    assert state.conditions.max_price == 1500 and state.selected_product_id == 8


@pytest.mark.parametrize("message", ["预算改成-10元", "预算改成50到20元", "预算改成10000000000元"])
async def test_invalid_or_unsupported_explicit_budget_clarifies_without_weakening_bounds(message):
    shop = ShopBoundary()
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("search_products", {})])])
    response, updated = await conversation(model, shop).run(message, prior())
    assert shop.calls == [] and response.status == "needs_input" and "预算" in response.answer
    assert updated.conditions.max_price == 1500 and updated.candidates == prior().candidates


async def test_ambiguous_sku_request_clarifies_without_choosing_a_variant():
    item = product(8)
    item.skus.append(item.skus[0].model_copy(update={"name": "白色", "unique_id": "white-8"}))
    state = prior(candidates=[item], selected_sku_id=None)
    shop = ShopBoundary()
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("get_product_detail", {"product_id": 8})])])
    response, updated = await conversation(model, shop).run("这个规格的库存", state)
    assert response.status == "needs_input" and "规格" in response.answer and shop.calls == []
    assert updated.selected_sku_id is None and updated.candidates == state.candidates


async def test_rule_interlude_cannot_replace_selection_via_unrequested_product_call():
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("search_products", {"keyword": "杯子"}, "a"),
                            call("search_policy", {"question": "退货规则"}, "b")]), AIMessage(content="")])
    shop, state = ShopBoundary(), prior()
    response, updated = await conversation(model, shop).run("退货规则是什么？", state)
    assert response.products == [] and shop.calls == []
    assert updated.candidates == state.candidates and updated.selected_product_id == state.selected_product_id


async def test_same_turn_failed_refresh_rolls_back_preceding_product_results():
    class FlakyShop(ShopBoundary):
        async def search_products(self, query, authorization):
            if self.calls:
                self.failure = ShopError("shop_unavailable")
            return await super().search_products(query, authorization)
    shop = FlakyShop()
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("search_products", {"keyword": "手机"})]),
                           AIMessage(content="", tool_calls=[call("search_products", {}, "b")]), AIMessage(content="")])
    state = prior()
    before = state.model_dump_json()
    with pytest.raises(ShopError) as failure:
        await conversation(model, shop).run("查手机", state)
    assert failure.value.code == "shop_unavailable"
    assert state.model_dump_json() == before and len(model.seen) == 2


async def test_new_query_can_combine_explicit_budget_with_model_parsed_product_type():
    shop = ShopBoundary()
    response, updated = await conversation(tool_then_finish(args={"keyword": "手机", "max_price": 99}), shop).run(
        "预算3元查手机", ConversationState())
    assert response.products and updated.conditions.keyword == "手机"
    assert shop.calls[0][1].max_price == updated.conditions.max_price == 3


async def test_budget_without_product_type_does_not_accept_a_guessed_type():
    shop = ShopBoundary()
    response, state = await conversation(tool_then_finish(args={"keyword": "手机"}), shop).run(
        "预算改成3元", ConversationState())
    assert shop.calls == [] and response.status == "needs_input"
    assert state.conditions.keyword == "" and state.conditions.max_price == 3


@pytest.mark.parametrize("name,error", [("search_products", "shop_unavailable"),
    ("search_products", "shop_invalid_response"), ("search_policy", "rag_unavailable")])
async def test_infrastructure_query_failure_terminates_without_completed_state(name, error):
    shop, policy, state = ShopBoundary(), PolicyBoundary(), prior()
    if name == "search_policy":
        policy.failure = ShopError(error)
    else:
        shop.failure = ShopError(error)
    model = tool_then_finish(name, {"question": "退货规则"} if name == "search_policy" else {})
    before = state.model_dump_json()
    with pytest.raises(ShopError) as failure:
        await conversation(model, shop, policy).run("退货规则" if name == "search_policy" else "预算改成3元", state)
    assert failure.value.code == error and state.model_dump_json() == before
    assert len(model.seen) == 1


async def test_conversation_never_exceeds_five_model_calls_or_three_tools_per_round():
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("search_products", {"keyword": "手机"})])
                           for _ in range(5)])
    shop = ShopBoundary()
    response, _ = await conversation(model, shop).run("预算改成20元", prior())
    assert response.status == "limit_reached" and response.model_rounds == len(model.seen) == 5
    assert len(shop.calls) == len(response.tool_calls) == 4


async def test_detail_preserves_valid_selected_sku_after_live_refresh_with_multiple_variants():
    item = product(8)
    item.skus.append(item.skus[0].model_copy(update={"name": "白色", "unique_id": "white-8"}))
    class VariantShop(ShopBoundary):
        async def get_product_detail(self, product_id, authorization):
            self.calls.append(("get_product_detail", product_id, authorization))
            return item
    state = prior(candidates=[item])
    _, updated = await conversation(tool_then_finish("get_product_detail", {"product_id": 8}), VariantShop()).run(
        "这个详情", state)
    assert updated.selected_product_id == 8 and updated.selected_sku_id == "sku-8"


async def test_removed_selected_sku_with_multiple_live_replacements_asks_for_sku():
    live = product(8)
    live.skus[0] = live.skus[0].model_copy(update={"name": "蓝色", "unique_id": "blue-8"})
    live.skus.append(live.skus[0].model_copy(update={"name": "白色", "unique_id": "white-8"}))

    class VariantShop(ShopBoundary):
        async def get_product_detail(self, product_id, authorization):
            self.calls.append(("get_product_detail", product_id, authorization))
            return live

    shop = VariantShop()
    response, updated = await conversation(tool_then_finish("get_product_detail", {"product_id": 8}), shop).run(
        "这个规格的库存", prior())
    assert shop.calls == [("get_product_detail", 8, AUTH)]
    assert response.status == "needs_input" and "规格" in response.answer
    assert response.products == updated.candidates == [live]
    assert updated.selected_product_id == 8 and updated.selected_sku_id is None


async def test_rejected_product_call_clarification_preserves_previous_display_and_selection():
    state = prior()
    response, updated = await conversation(tool_then_finish("search_products", {"limit": 99})).run(
        "再看看商品", state)
    assert response.status == "needs_input"
    assert response.tool_calls[-1].status == "failed" and response.products == []
    assert updated.candidates == state.candidates
    assert updated.selected_product_id == 8 and updated.selected_sku_id == "sku-8"


async def test_explicit_budget_overrides_stale_inverted_model_interval_before_merged_validation():
    shop = ShopBoundary()
    model = tool_then_finish(args={"keyword": "手机", "min_price": 2000, "max_price": 1000})
    _, updated = await conversation(model, shop).run("预算改成100到200元", prior())
    assert shop.calls[0][1].min_price == updated.conditions.min_price == 100
    assert shop.calls[0][1].max_price == updated.conditions.max_price == 200


async def test_budget_from_old_amount_to_new_amount_uses_new_amount():
    shop = ShopBoundary()
    _, updated = await conversation(tool_then_finish(args={}), shop).run("预算从1500元改成200元", prior())
    assert shop.calls[0][1].max_price == updated.conditions.max_price == 200


async def test_long_public_answers_trim_completed_history_to_snapshot_byte_limit():
    class LongPolicy(PolicyBoundary):
        def answer(self, question):
            result = super().answer(question)
            result.answer = "演示" * 3000
            return result
    state = prior(history=[PublicMessage(role="user" if index % 2 == 0 else "assistant", content="公开" * 1000)
                           for index in range(10)])
    response, updated = await conversation(tool_then_finish("search_policy", {"question": "退货规则"}),
                                           policy=LongPolicy()).run("退货规则是什么？", state)
    assert len(response.answer) > 4000 and len(updated.history[-1].content) == 4000
    assert len(updated.model_dump_json().encode("utf-8")) <= 65536
    assert updated.candidates == state.candidates and updated.history[-2].content == "退货规则是什么？"
