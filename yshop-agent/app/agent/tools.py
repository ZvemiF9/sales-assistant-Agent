"""Read-only request executor; credentials never enter tool messages or graph state."""

import json
import logging
import math
from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from app.actions.schemas import ActionResponse, CartActionCard, CartActionResult
from app.agent.schemas import ChatResponse, SessionChatResponse, TokenUsage, ToolTrace
from app.agent.state import ConversationState, MAX_SNAPSHOT_BYTES, PublicMessage
from app.errors import ShopError
from app.rag.schemas import PolicyAnswer, PolicyQuery, PolicySource
from app.rag.service import PolicyService
from app.schemas import (
    CartSnapshot, CartItem, NonBlankString, PositiveId, ProductDetail, ProductSearchInput,
    ProductSearchResult, ProductSKU, PublicModel,
)
from app.tools.yshop import YShopClient
from app.telemetry import metadata_log


logger = logging.getLogger(__name__)


class ProductQuery(ProductSearchInput):
    keyword: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, max_length=100)] = ""
    limit: Literal[3] = 3

    @field_validator("limit", mode="before")
    @classmethod
    def fixed_limit(cls, value):
        if type(value) is not int or value != 3:
            raise ValueError("product queries use a limit of three")
        return value


class DetailQuery(PublicModel):
    product_id: PositiveId


class _ContextSKU(PublicModel):
    unique_id: NonBlankString
    name: str


class _ContextProduct(PublicModel):
    product_id: PositiveId
    name: NonBlankString
    skus: list[_ContextSKU]


class _ConversationContext(PublicModel):
    conditions: ProductSearchInput
    candidates: list[_ContextProduct] = Field(max_length=3)
    selected_product_id: PositiveId | None
    selected_sku_id: Annotated[str, StringConstraints(strict=True, strip_whitespace=True,
                                                    min_length=1, max_length=128)] | None


# Only these schema fields are prose/media; everything else keeps its typed value.
_TEXT_FIELDS = {
    ProductSKU: {"name", "image"},
    ProductDetail: {"name", "description", "image"},
    ProductSearchInput: {"keyword"},
    ProductQuery: {"keyword"},
    DetailQuery: set(),
    ProductSearchResult: set(),
    CartItem: {"name", "sku_name", "image"},
    CartSnapshot: set(),
    PolicyQuery: {"question"},
    PolicySource: {"title", "section", "source", "text"},
    PolicyAnswer: {"question", "answer", "notice"},
    PublicMessage: {"content"},
    ConversationState: set(),
    CartActionCard: {"product_name", "sku_name", "image"},
    CartActionResult: set(),
    ActionResponse: {"answer"},
    ToolTrace: set(),
    TokenUsage: set(),
    ChatResponse: {"answer", "notice"},
    SessionChatResponse: {"answer", "notice"},
    _ContextSKU: {"name"},
    _ContextProduct: {"name"},
    _ConversationContext: set(),
}


def _bounded_json(value, depth=0):
    if depth > 12:
        raise ValueError("public JSON depth")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str and len(value) <= 16384:
        return
    if type(value) is list and len(value) <= 64:
        for item in value:
            _bounded_json(item, depth + 1)
        return
    if type(value) is dict and len(value) <= 64 and all(type(key) is str for key in value):
        for key, item in value.items():
            _bounded_json(key, depth + 1)
            _bounded_json(item, depth + 1)
        return
    raise ValueError("non-public JSON")


_INPUTS = {
    "search_products": ProductQuery,
    "get_product_detail": DetailQuery,
    "search_policy": PolicyQuery,
}
_DESCRIPTIONS = {
    "search_products": "只读查询商城商品，支持关键词、分类和预算，每次最多返回三个商品。",
    "get_product_detail": "只读查询正整数商品 ID 的详情、规格、价格和库存。",
    "search_policy": "检索商城演示规则原文；没有充分依据时明确拒答。",
}


def function_schemas() -> list[dict]:
    # Fresh dictionaries keep a shared model/client from mutating request schemas.
    return [{"type": "function", "function": {"name": name, "description": _DESCRIPTIONS[name],
             "parameters": query.model_json_schema()}} for name, query in _INPUTS.items()]


@dataclass(frozen=True)
class ToolResult:
    payload: dict
    trace: ToolTrace
    products: list[ProductDetail] | None = None
    policy: PolicyAnswer | None = None


class QueryTools:
    def __init__(self, shop: YShopClient, policy: PolicyService, authorization: str,
                 *, public_secrets: tuple[str, ...] = (), public_text_secrets: tuple[str, ...] = ()):
        self._shop = shop
        self._policy = policy
        self._authorization = authorization
        self._public_secrets = tuple(secret for secret in public_secrets if secret)
        self._public_text_secrets = tuple(secret for secret in public_text_secrets if secret)

    def function_schemas(self) -> list[dict]:
        return function_schemas()

    def with_public_secrets(self, secrets: tuple[str, ...]) -> "QueryTools":
        """Keep private transport settings in a fresh executor, outside public state."""
        return QueryTools(self._shop, self._policy, self._authorization,
                          public_secrets=(*self._public_secrets, *secrets),
                          public_text_secrets=self._public_text_secrets)

    async def get_cart_product(self, product_id, unique_id, authorization):
        return await self._shop.get_cart_product(product_id, unique_id, authorization,
            public_secrets=(*self._public_secrets, *self._public_text_secrets))

    def public_data(self, value):
        """Redact free text only; structured data must use an explicit public schema."""
        if type(value) is not str or len(value) > 16384:
            raise ValueError("invalid public text")
        return self._redact_text(value)

    def _redact_text(self, value):
        for secret in (self._authorization, self._authorization.partition(" ")[2],
                       *self._public_secrets, *self._public_text_secrets):
            if secret:
                value = value.replace(secret, "[已隐藏]")
        return value

    def public_control(self, value, *, reject_private_text=False):
        """Identifiers and controls are immutable: credential contamination rejects them."""
        _bounded_json(value)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("public JSON size")
        return self._check_control(value, reject_private_text=reject_private_text)

    def _check_control(self, value, *, reject_private_text=False):
        def check(item):
            if type(item) is dict:
                for key, child in item.items():
                    check(key)
                    check(child)
            elif type(item) is list:
                for child in item:
                    check(child)
            elif item is not None:
                text = item if type(item) is str else json.dumps(item, allow_nan=False)
                if any(secret and secret in text for secret in (
                        self._authorization, self._authorization.partition(" ")[2], *self._public_secrets,
                        *(self._public_text_secrets if reject_private_text else ()))):
                    raise ValueError("credential in public control")
        check(value)
        return value

    def public_model(self, value, *, error_code="shop_invalid_response"):
        try:
            model = type(value)
            if model not in _TEXT_FIELDS:
                raise ValueError("unknown public schema")
            raw = value.model_dump(mode="json", warnings="error")
            validated = model.model_validate(raw)
            # Known fields use their schemas, and snapshot sizing uses the store's compact serializer.
            if len(validated.model_dump_json(warnings="error").encode("utf-8")) > MAX_SNAPSHOT_BYTES:
                raise ValueError("public model size")

            def project(item):
                if type(item) in _TEXT_FIELDS:
                    fields = _TEXT_FIELDS[type(item)]
                    result = {}
                    for name in item.model_dump():
                        child = getattr(item, name)
                        result[name] = self._redact_text(child) if name in fields and child is not None else project(child)
                    return result
                if type(item) is list:
                    return [project(child) for child in item]
                if item is None or type(item) in {str, bool, int, float, Decimal}:
                    return self._check_control(str(item) if type(item) is Decimal else item)
                raise ValueError("unknown public field")

            projected = model.model_validate(project(validated))
            if len(projected.model_dump_json(warnings="error").encode("utf-8")) > MAX_SNAPSHOT_BYTES:
                raise ValueError("public model size")
            return projected
        except (ValueError, TypeError, AttributeError, RecursionError, OverflowError):
            raise ShopError(error_code) from None

    def public_context(self, value):
        try:
            # The known schema owns field capacities; raw JSON still cannot rely on coercion.
            def check(item, depth=0):
                if depth > 12:
                    raise ValueError("public JSON depth")
                if item is None or type(item) in {str, bool, int}:
                    return
                if type(item) is float and math.isfinite(item):
                    return
                if type(item) is list:
                    for child in item:
                        check(child, depth + 1)
                    return
                if type(item) is dict and all(type(key) is str for key in item):
                    for key, child in item.items():
                        check(key, depth + 1)
                        check(child, depth + 1)
                    return
                raise ValueError("non-public JSON")

            check(value)
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
                raise ValueError("public context size")
            context = _ConversationContext.model_validate(value)
            return self.public_model(context, error_code="invalid_request").model_dump(mode="json")
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise ShopError("invalid_request") from None

    def public_calls(self, calls):
        _bounded_json(calls)
        result = []
        for call in calls:
            if type(call) is not dict or set(call) != {"name", "id", "args", "type"}:
                raise ValueError("invalid tool envelope")
            controls = {key: call[key] for key in ("name", "id", "type")}
            self.public_control(controls)
            if type(call["args"]) is not dict:
                raise ValueError("invalid tool arguments")
            text_fields = {"keyword"} if call["name"] == "search_products" else (
                {"question"} if call["name"] == "search_policy" else set())
            known_fields = _INPUTS[call["name"]].model_fields if call["name"] in _INPUTS else {}
            args = {}
            for key, value in call["args"].items():
                self.public_control(key, reject_private_text=key not in known_fields)
                if key in text_fields and type(value) is str:
                    args[key] = self.public_data(value)
                else:
                    args[key] = self.public_control(value, reject_private_text=key not in known_fields)
            result.append({**controls, "args": args})
        return result

    async def execute(self, name: str, args: dict) -> ToolResult:
        started = perf_counter()
        products, policy = None, None
        safe_name = name if name in _INPUTS else "unknown_tool"
        try:
            if name not in _INPUTS or not isinstance(args, dict):
                raise ShopError("invalid_request")
            try:
                query = self.public_model(_INPUTS[name].model_validate(args), error_code="invalid_request")
            except (ValidationError, ValueError, TypeError):
                raise ShopError("invalid_request") from None
            if name == "search_products":
                result = await self._shop.search_products(
                    ProductSearchInput(**query.model_dump()), self._authorization)
                if not isinstance(result, ProductSearchResult):
                    raise ShopError("shop_invalid_response")
                result = self.public_model(result)
                products = result.products[:3]
                payload = result.model_dump(mode="json")
                payload["products"] = [item.model_dump(mode="json") for item in products]
            elif name == "get_product_detail":
                result = await self._shop.get_product_detail(query.product_id, self._authorization)
                if not isinstance(result, ProductDetail):
                    raise ShopError("shop_invalid_response")
                result = self.public_model(result)
                products = [result]
                payload = result.model_dump(mode="json")
            else:
                result = await run_in_threadpool(self._policy.answer, query.question)
                if not isinstance(result, PolicyAnswer):
                    raise ShopError("rag_unavailable")
                policy = self.public_model(result, error_code="rag_unavailable")
                payload = policy.model_dump(mode="json")
            status = "succeeded"
        except ShopError as error:
            if error.code == "login_required":
                # No remaining tool or model is called after identity expiration.
                metadata_log(logger, "agent tool name=%s status=login_required elapsed_ms=%d", safe_name,
                            max(0, int((perf_counter() - started) * 1000)))
                raise ShopError("login_required") from None
            payload = {"error": {"code": error.code, "message": error.message}}
            status = "failed"
        except Exception:
            error = ShopError("rag_unavailable" if name == "search_policy" else "shop_invalid_response")
            payload = {"error": {"code": error.code, "message": error.message}}
            status = "failed"
        elapsed_ms = max(0, int((perf_counter() - started) * 1000))
        metadata_log(logger, "agent tool name=%s status=%s elapsed_ms=%d", safe_name, status, elapsed_ms)
        return ToolResult(payload=payload, products=products, policy=policy,
                          trace=ToolTrace(name=safe_name, status=status, elapsed_ms=elapsed_ms))
