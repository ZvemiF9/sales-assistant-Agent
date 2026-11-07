"""Completed-turn conversation orchestration over the existing bounded query graph."""

import re
from decimal import Decimal

from pydantic import ValidationError

from app.agent.graph import QueryAgent
from app.agent.model import ChatModel
from app.agent.schemas import ChatRequest, ChatResponse, ToolTrace
from app.agent.state import ConversationState, MAX_HISTORY_MESSAGES, MAX_MESSAGE_CHARACTERS, MAX_SNAPSHOT_BYTES, PublicMessage
from app.agent.tools import DetailQuery, ProductQuery, QueryTools, ToolResult
from app.errors import ShopError
from app.rag.schemas import PolicyQuery
from app.schemas import ProductDetail, ProductSearchInput


_AMOUNT = r"(?:[0-9]{1,9}(?:\.[0-9]{1,2})?|[零〇一二两三四五六七八九十百千万]{1,12}(?:点[零〇一二两三四五六七八九]{1,2})?)"
_DIGITS = {character: number for number, character in enumerate("零一二三四五六七八九")}
_DIGITS.update({"〇": 0, "两": 2})
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_ORDINAL = re.compile(r"第(" + _AMOUNT + r")[个件款台]?")
_POLICY = re.compile(r"规则|退货|退款|(?:能|可以|可)退|售后|运费|配送|包邮|保修|支付|发货|优惠券|礼券|会员|积分")
_POLICY_QUESTION = re.compile(r"[?？]|什么|哪些|多少|怎么|如何|吗|是否|能否|能不能|会不会")
_SEARCH_REQUEST = re.compile(r"查询|搜索|查找|推荐|看看|想买|购买|查|找")
_SEARCH_COURTESY = re.compile(r"(?:一下|下|一查|一找|看看|看)?(?:吧|呀|呢|啊|哦|谢谢)*")
_LIVE_DETAIL = re.compile(r"详情|库存|多少钱|价格|规格|型号")
_POLICY_COST = re.compile(
    r"(?:退货|退款|售后|运费|配送|包邮|发货)[^,，。；;！？!?]{0,20}?(?:多少钱|价格|费用|收费|怎么收)"
)
_SKU_REFERENCE = re.compile(r"这个规格|这款规格|选.*规格")
_SKU_SELECTION_PROMPT = "请明确选择商品规格名称；有多个规格时需要说明希望查询哪一个。"
_CLAUSE_SEPARATOR = re.compile(r"[,，。；;！？!?]")


def _amount(text: str) -> Decimal:
    if text[0].isascii():
        return Decimal(text)
    integer, _, fraction = text.partition("点")
    if all(character in _DIGITS for character in integer):
        value = int("".join(str(_DIGITS[character]) for character in integer))
    else:
        total = section = number = 0
        for character in integer:
            if character in _DIGITS:
                number = _DIGITS[character]
            elif character == "万":
                total += (section + number) * 10000
                section = number = 0
            else:
                section += (number or 1) * _UNITS[character]
                number = 0
        value = total + section + number
    decimal = "".join(str(_DIGITS[character]) for character in fraction)
    return Decimal(str(value) + ("." + decimal if decimal else ""))


def _budget_patch(message: str) -> dict | None:
    """Recognize explicit bounded money syntax; leave broader language to the model."""
    if re.search(r"(?:不限|不限制|取消|去掉|清除|不设).*?(?:预算|价格)|(?:预算|价格).*?(?:不限|不限制|取消|去掉|清除)", message):
        return {"min_price": None, "max_price": None}
    if not re.search(r"预算|价格|[元块]|不超过|不低于", message):
        return None
    # Product ordinals and a quoted live price are not changes to search bounds.
    if (_ORDINAL.search(message) or re.search(r"这个|这件|这款|它|刚才那个", message)) and "预算" not in message:
        return None
    clauses = _CLAUSE_SEPARATOR.split(message)
    clause = next((item for item in clauses if re.search(r"预算|价格", item)), message)
    budget_context = re.search(r"预算|价格", clause)
    text = clause[budget_context.end():] if budget_context else clause
    adjustments = list(re.finditer(r"改成|改为|调整到|设为|提高到|降低到", text))
    if adjustments:
        text = text[adjustments[-1].end():]
    if "负" in text:
        raise ValueError("negative explicit budget")
    # Reject unsupported numeric tokens as a whole, rather than matching a prefix.
    for token in re.findall(r"(?<![0-9])[-+]?[0-9]+(?:\.[0-9]+)?", text):
        if not re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,2})?", token):
            raise ValueError("unsupported explicit budget")
    interval = re.search("(" + _AMOUNT + r")(?:元|块)?\s*(?:到|至|[-~～—])\s*(" + _AMOUNT + r")(?:元|块)?", text)
    if interval:
        return {"min_price": _amount(interval[1]), "max_price": _amount(interval[2])}
    amount = re.search(_AMOUNT, text)
    if amount is None:
        return None
    value = _amount(amount[0])
    minimum = bool(re.search(r"至少|不低于|以上|起步", message))
    return {"min_price": value if minimum else None, "max_price": None if minimum else value}


def _type_switch(message: str) -> str | None:
    selections = re.finditer(
        r"(?:换成|改为|改成)\s*(?:查询|查|看看|看|买)?([A-Za-z\u4e00-\u9fff]{1,30})", message)
    for selection in selections:
        clause_prefix = _CLAUSE_SEPARATOR.split(message[:selection.start()])[-1]
        if re.search(r"预算|价格|规格|颜色", clause_prefix):
            continue
        keyword = re.split(r"预算|价格|以内|以下|以上", selection[1])[0].rstrip("吧呢了")
        if keyword and not re.search(r"元|块|不限|限制", keyword):
            return keyword
    return None


def _reference(message: str, state: ConversationState) -> tuple[bool, ProductDetail | None]:
    ordinals = _ORDINAL.findall(message)
    if ordinals:
        if len(ordinals) != 1:
            return True, None
        value = _amount(ordinals[0])
        index = int(value) - 1
        if value != int(value) or not 0 <= index < len(state.candidates):
            return True, None
        return True, state.candidates[index]
    if re.search(r"这个|这件|这款|它|刚才那个|选中的|所选", message):
        selected = next((item for item in state.candidates if item.product_id == state.selected_product_id), None)
        if selected is not None:
            return True, selected
        return True, state.candidates[0] if len(state.candidates) == 1 else None
    return False, None


def _product_detail_requested(message: str) -> bool:
    """Exclude policy-scoped costs while retaining genuine product-price requests."""
    without_policy_cost = _POLICY_COST.sub("", message)
    return bool(_LIVE_DETAIL.search(without_policy_cost))


def _separate_search_requested(message: str) -> bool:
    for clause in _CLAUSE_SEPARATOR.split(message):
        intent = _SEARCH_REQUEST.search(clause)
        if intent and not _POLICY.search(clause):
            target = clause[intent.end():].strip()
            if target and not _SEARCH_COURTESY.fullmatch(target):
                return True
    return False


class _ConversationTools:
    """Request-local argument validation and condition inheritance, never persisted."""

    def __init__(self, tools: QueryTools, conditions: ProductSearchInput, *, budget: dict | None,
                 switch: str | None, reference: ProductDetail | None, policy_only: bool, message: str):
        self._tools = tools
        self.conditions = conditions
        self._budget = budget
        self._switch = switch
        self._reference = reference
        self._policy_only = policy_only
        self._message = message

    def public_data(self, value):
        return self._tools.public_data(value)

    def public_model(self, value, **kwargs):
        return self._tools.public_model(value, **kwargs)

    def public_context(self, value):
        return self._tools.public_context(value)

    def public_control(self, value):
        return self._tools.public_control(value)

    def public_calls(self, value):
        return self._tools.public_calls(value)

    def _allows_tool(self, name: str) -> bool:
        if self._policy_only and name in {"search_products", "get_product_detail"}:
            return False
        if name == "search_products" and self._reference is not None:
            return False
        if name == "get_product_detail" and (self._budget is not None or self._switch is not None):
            return False
        return True

    def function_schemas(self) -> list[dict]:
        return [schema for schema in self._tools.function_schemas()
                if self._allows_tool(schema["function"]["name"])]

    async def execute(self, name: str, args: dict) -> ToolResult:
        try:
            if not self._allows_tool(name):
                raise ValueError
            if name == "search_products":
                # Explicit server values replace stale model conditions before the
                # merged interval is validated. Unknown fields and tool limits still fail.
                constrained = dict(args)
                if self._switch is not None:
                    constrained.update(keyword=self._switch, category_id=None, min_price=None, max_price=None, page=1)
                if self._budget is not None:
                    constrained.update(self._budget)
                query = ProductQuery.model_validate(constrained)
                supplied = query.model_dump(exclude_unset=True)
                merged = {**self.conditions.model_dump(), **supplied, "limit": 3}
                if not supplied.get("keyword") and self.conditions.keyword:
                    merged["keyword"] = self.conditions.keyword
                if self._switch is not None:
                    merged.update(keyword=self._switch, category_id=None, min_price=None, max_price=None, page=1)
                elif self._budget is not None and self.conditions.keyword:
                    merged.update(keyword=self.conditions.keyword, category_id=self.conditions.category_id, page=1)
                if self._budget is not None:
                    merged.update(self._budget)
                query = ProductQuery.model_validate(merged)
                if (self._budget is not None and not (self.conditions.keyword or self.conditions.category_id)
                        and (not query.keyword or query.keyword not in self._message)):
                    # A first-message budget does not license a guessed product type.
                    raise ValueError
                self.conditions = ProductSearchInput(**query.model_dump())
                args = query.model_dump()
            elif name == "get_product_detail":
                query = DetailQuery.model_validate(args)
                if self._reference is not None:
                    query = DetailQuery(product_id=self._reference.product_id)
                args = query.model_dump()
            elif name == "search_policy" and self._policy_only:
                # Validate the proposal before grounding it; malformed arguments
                # must not become valid merely because the user asked a question.
                PolicyQuery.model_validate(args)
                args = PolicyQuery(question=self._message).model_dump()
        except (ValidationError, ValueError, TypeError):
            error = ShopError("invalid_request")
            return ToolResult(payload={"error": {"code": error.code, "message": error.message}},
                              trace=ToolTrace(name=name, status="failed", elapsed_ms=0))
        result = await self._tools.execute(name, args)
        if result.trace.status == "failed":
            error = ShopError(result.payload["error"]["code"])
            if error.http_status >= 500:
                # Infrastructure failures terminate a conversation turn; the API
                # releases the claim without committing any intermediate query.
                raise error from None
        return result


class ConversationAgent:
    def __init__(self, model: ChatModel, tools: QueryTools):
        self._model = model
        key = getattr(getattr(model, "settings", None), "llm_api_key", None)
        secrets = (key.get_secret_value(),) if key is not None else ()
        self._tools = tools.with_public_secrets(secrets)

    async def run(self, message: str, state: ConversationState) -> tuple[ChatResponse, ConversationState]:
        try:
            message = ChatRequest(message=message).message
            # Revalidate a detached copy; never mutate the caller's claimed snapshot.
            state = self._tools.public_model(state, error_code="invalid_request")
        except (ValidationError, ValueError, TypeError, AttributeError):
            raise ShopError("invalid_request") from None
        message = self._tools.public_data(message)
        switch = _type_switch(message)
        invalid_budget = False
        try:
            budget = _budget_patch(message)
        except ValueError:
            budget, switch, invalid_budget = None, None, True
        has_reference, reference = _reference(message, state)
        policy_only = (bool(_POLICY.search(message)) and not _product_detail_requested(message)
                       and not _separate_search_requested(message)
                       and budget is None and switch is None)
        clarification = "budget" if invalid_budget else None
        required_call = None
        conditions = state.conditions.model_dump()
        if switch is not None:
            conditions.update(keyword=switch, category_id=None, min_price=None, max_price=None, page=1, limit=3)
        if budget is not None:
            conditions.update(budget)
            conditions.update(page=1, limit=3)
        try:
            parsed_conditions = ProductSearchInput.model_validate(conditions)
        except ValidationError:
            parsed_conditions, budget, switch = state.conditions, None, None
            clarification = "budget"
        if clarification is None and (budget is not None or switch is not None):
            if parsed_conditions.keyword or parsed_conditions.category_id:
                required_call = {"name": "search_products", "args": parsed_conditions.model_dump(mode="json"),
                                 "id": "conversation_required", "type": "tool_call"}
            # With no prior type, allow the model to extract one from this message.
            # The executor accepts only a keyword grounded in the user's text.
            # A new search invalidates the previous choice immediately in local state.
            state.candidates = []
            state.selected_product_id = state.selected_sku_id = None
            reference = None
        elif clarification is None and has_reference and not policy_only:
            if reference is None:
                clarification = "product_selection"
            else:
                if _SKU_REFERENCE.search(message) and len(reference.skus) > 1:
                    matches = [sku for sku in reference.skus if sku.name and sku.name in message]
                    if len(matches) != 1 and state.selected_sku_id not in {sku.unique_id for sku in reference.skus}:
                        clarification = "sku_selection"
                if clarification is None:
                    required_call = {"name": "get_product_detail", "args": {"product_id": reference.product_id},
                                     "id": "conversation_required", "type": "tool_call"}
        if clarification is None and policy_only and _POLICY_QUESTION.search(message):
            required_call = {"name": "search_policy", "args": {"question": message},
                             "id": "conversation_required", "type": "tool_call"}
        executor = _ConversationTools(self._tools, parsed_conditions, budget=budget, switch=switch,
                                      reference=reference if has_reference and not policy_only else None,
                                      policy_only=policy_only, message=message)
        # Historical fields needed for reference resolution exclude old live prices/stocks.
        context = {"conditions": parsed_conditions.model_dump(mode="json"),
                   "candidates": [{"product_id": item.product_id, "name": item.name,
                                   "skus": [{"unique_id": sku.unique_id, "name": sku.name} for sku in item.skus]}
                                  for item in state.candidates],
                   "selected_product_id": state.selected_product_id,
                   "selected_sku_id": state.selected_sku_id}
        response = await QueryAgent(self._model, executor).run(message, history=state.history, context=context,
                            required_call=required_call, clarification=clarification)
        state.conditions = executor.conditions
        product_traces = [trace for trace in response.tool_calls
                          if trace.status == "succeeded" and trace.name in {"search_products", "get_product_detail"}]
        if product_traces and not policy_only:
            previous_product_id, previous_sku_id = state.selected_product_id, state.selected_sku_id
            state.candidates = response.products
            state.selected_product_id = state.selected_sku_id = None
            if product_traces[-1].name == "get_product_detail" and response.products:
                selected = response.products[0]
                state.selected_product_id = selected.product_id
                if len(selected.skus) == 1:
                    state.selected_sku_id = selected.skus[0].unique_id
                else:
                    matches = [sku for sku in selected.skus if sku.name and sku.name in message]
                    if len(matches) == 1:
                        state.selected_sku_id = matches[0].unique_id
                    elif selected.product_id == previous_product_id and previous_sku_id in {sku.unique_id for sku in selected.skus}:
                        state.selected_sku_id = previous_sku_id
                    elif _SKU_REFERENCE.search(message):
                        response = response.model_copy(update={"answer": _SKU_SELECTION_PROMPT, "status": "needs_input"})
        state.history = [*state.history, PublicMessage(role="user", content=message[:MAX_MESSAGE_CHARACTERS]),
                         PublicMessage(role="assistant", content=response.answer[:MAX_MESSAGE_CHARACTERS])][-MAX_HISTORY_MESSAGES:]
        # UTF-8 byte size can bind before the message-count limit for Chinese text.
        # Evict older completed turns while keeping this turn and displayed candidates.
        while len(state.history) > 2 and len(state.model_dump_json().encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            state.history = state.history[2:]
        try:
            completed = self._tools.public_model(state, error_code="session_unavailable")
        except (ValidationError, ValueError, TypeError):
            raise ShopError("session_unavailable") from None
        return response, completed
