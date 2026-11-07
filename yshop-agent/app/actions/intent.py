"""Conservative cart intent from the current user message, never retrieved text."""

import re
from dataclasses import dataclass

from app.agent.conversation import _amount
from app.agent.state import ConversationState


_NUMBER = r"[0-9零〇一二两三四五六七八九十百千万]+"
_ORDINAL = re.compile(r"第(" + _NUMBER + r")[个件款台]?")
_CHINESE_DIGIT = r"[一二三四五六七八九]"
_CHINESE_HIGH_DIGIT = r"[一二两三四五六七八九]"
_CHINESE_TENS = rf"(?:{_CHINESE_DIGIT}?十{_CHINESE_DIGIT}?)"
_CHINESE_HUNDREDS = rf"(?:{_CHINESE_HIGH_DIGIT}百(?:{_CHINESE_DIGIT}十{_CHINESE_DIGIT}?|[零〇]{_CHINESE_DIGIT})?)"
_CHINESE_THOUSANDS = rf"(?:{_CHINESE_HIGH_DIGIT}千(?:{_CHINESE_HUNDREDS}|[零〇](?:{_CHINESE_TENS}|{_CHINESE_DIGIT}))?)"
_INTEGER = re.compile(
    rf"(?:[0-9]+|[零〇]*{_CHINESE_DIGIT}|两|{_CHINESE_TENS}|{_CHINESE_HUNDREDS}|{_CHINESE_THOUSANDS})"
)
_NUMERIC_LABEL_LETTERS = frozenset("零〇一二两三四五六七八九十百千万点负正eE到至件个台份只")
_CART_VERB = r"(?:加入|加到|添加到|添加进|放入|放到|加进|加)"
# Consume a specification-clause comma only at the cursor and before a cart verb.
_QUANTITY_PREFIX = re.compile(
    r"(?:(?:" + _CART_VERB
    + r"|规格\s*[,，](?=\s*" + _CART_VERB + r")"
    + r"|购物车|这个规格|这款规格|这个|这件|这款|它|刚才那个|选中的|所选|选择|规格|商品|请|帮我|给我|把|将|的|继续)|\s)*"
)
_QUANTITY_UNIT = re.compile(r"[件个台份只]")
_ADD = re.compile(_CART_VERB + r"\s*(?:.{0,30}?)购物车")
_UNTRUSTED = re.compile(r"规则|政策|描述|指令|提示词|系统|如果|假如|若|不要|别|不加|取消|[?？]|吗|是否|能否|能不能|可以.*购物车")
_QUOTES = re.compile(r"""["'`“”‘’「」『』]""")


@dataclass(frozen=True)
class CartIntent:
    product_id: int | None = None
    unique_id: str | None = None
    quantity: int | None = None
    reason: str | None = None


def resolve_cart_intent(message: str, state: ConversationState) -> CartIntent | None:
    if not _ADD.search(message) or _UNTRUSTED.search(message) or _QUOTES.search(message):
        return None
    candidates = state.candidates
    ordinals = list(_ORDINAL.finditer(message))
    selected = None
    if ordinals:
        if len(ordinals) == 1:
            index = int(_amount(ordinals[0][1])) - 1
            if 0 <= index < len(candidates):
                selected = candidates[index]
    else:
        names = [item for item in candidates if item.name in message]
        if len(names) == 1:
            selected = names[0]
        elif len(names) == 0:
            selected = next((item for item in candidates if item.product_id == state.selected_product_id), None)
            if selected is None and len(candidates) == 1:
                selected = candidates[0]
    if selected is None:
        return CartIntent(reason="product_selection")
    matches = [sku for sku in selected.skus if sku.name and sku.name in message]
    if len(matches) == 1:
        sku = matches[0]
    elif len(matches) > 1:
        return CartIntent(reason="sku_selection")
    elif selected.product_id == state.selected_product_id and state.selected_sku_id is not None:
        sku = next((item for item in selected.skus if item.unique_id == state.selected_sku_id), None)
    else:
        sku = selected.skus[0] if len(selected.skus) == 1 else None
    if sku is None:
        return CartIntent(reason="sku_selection")
    text = _ORDINAL.sub(" ", message)
    for name in (selected.name, sku.name):
        if name and any(letter.isalpha() and letter not in _NUMERIC_LABEL_LETTERS for letter in name):
            text = text.replace(name, " ")
    # Strip context only at the cursor, then retain every raw character up to a unit.
    quantities = []
    cursor = 0
    while cursor < len(text):
        start = _QUANTITY_PREFIX.match(text, cursor).end()
        unit = _QUANTITY_UNIT.search(text, start)
        if unit is None:
            break
        quantities.append(text[start:unit.start()].strip())
        cursor = unit.end()
    if len(quantities) != 1:
        return CartIntent(reason="quantity")
    token = quantities[0]
    if _INTEGER.fullmatch(token) is None:
        return CartIntent(reason="quantity")
    quantity = int(_amount(token))
    if not 1 <= quantity <= 9999:
        return CartIntent(reason="quantity")
    return CartIntent(selected.product_id, sku.unique_id, quantity)
