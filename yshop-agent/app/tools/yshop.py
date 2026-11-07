import json
from decimal import Decimal
from typing import Any

import httpx
from pydantic import ValidationError

from app.errors import ShopError
from app.schemas import (
    CartAddInput, CartAddResult, CartItem, CartSnapshot,
    ProductDetail, ProductSearchInput, ProductSearchResult, ProductSKU, UserIdentity,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _qualification_keyword(name: str, secrets: tuple[str, ...]) -> str:
    if type(name) is not str or not name or len(name) > 16384:
        raise ShopError("shop_invalid_response")
    spans = []
    for secret in secrets:
        if not secret:
            continue
        start = name.find(secret)
        while start >= 0:
            spans.append((start, start + len(secret)))
            start = name.find(secret, start + 1)
    # Exclude complete echo spans before truncation; never join across a removed secret.
    parts, cursor = [], 0
    for start, end in sorted(spans):
        if start > cursor:
            parts.append(name[cursor:start])
        cursor = max(cursor, end)
    parts.append(name[cursor:])
    candidates = [part.strip()[:100].strip() for part in parts]
    keyword = max(candidates, key=len, default="")
    if not keyword:
        raise ShopError("shop_invalid_response")
    return keyword


class YShopClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self.http = http

    async def _get_data(
        self, path: str, authorization: str, *, params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(2):
            try:
                response = await self.http.get(path, headers={"Authorization": authorization}, params=params)
                break
            except httpx.DecodingError:
                raise ShopError("shop_invalid_response") from None
            except httpx.RequestError:
                if attempt == 1:
                    raise ShopError("shop_unavailable") from None

        if response.status_code == 401:
            raise ShopError("login_required")
        if not response.is_success:
            raise ShopError("shop_unavailable" if response.status_code >= 500 else "shop_rejected")
        try:
            payload = json.loads(response.content, parse_float=Decimal)
        except (ValueError, UnicodeError):
            raise ShopError("shop_invalid_response") from None
        if not isinstance(payload, dict):
            raise ShopError("shop_invalid_response")
        status = payload.get("status")
        if type(status) is int and status in {401, 5104, 5107}:
            raise ShopError("login_required")
        success = payload.get("success")
        if type(status) is not int or type(success) is not bool:
            raise ShopError("shop_invalid_response")
        if status != 200 or success is not True:
            raise ShopError("shop_rejected")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ShopError("shop_invalid_response")
        return data

    async def get_user(self, authorization: str) -> UserIdentity:
        data = await self._get_data("userinfo", authorization)
        try:
            return UserIdentity(user_id=data.get("uid"), nickname=data.get("nickname") or "")
        except ValidationError:
            raise ShopError("shop_invalid_response") from None

    async def search_products(
        self, query: ProductSearchInput, authorization: str,
    ) -> ProductSearchResult:
        if not isinstance(query, ProductSearchInput):
            raise ShopError("invalid_request")
        try:
            query = ProductSearchInput.model_validate(query.model_dump(warnings=False))
        except ValidationError:
            raise ShopError("invalid_request") from None
        params = {key: str(value) for key, value in query.model_dump().items()
                  if value is not None and value != ""}
        data = await self._get_data("agent/products", authorization, params=params)
        if not isinstance(data.get("products"), list):
            raise ShopError("shop_invalid_response")
        products = []
        product_ids = set()
        try:
            for row in data["products"]:
                if not isinstance(row, dict) or not isinstance(row.get("skus"), list):
                    raise ShopError("shop_invalid_response")
                skus = []
                unique_ids = set()
                for value in row["skus"]:
                    if not isinstance(value, dict):
                        raise ShopError("shop_invalid_response")
                    sku = ProductSKU(**{key: value.get(key) for key in ProductSKU.model_fields})
                    if (sku.stock <= 0 or sku.unique_id in unique_ids
                            or (query.min_price is not None and sku.price < query.min_price)
                            or (query.max_price is not None and sku.price > query.max_price)):
                        raise ShopError("shop_invalid_response")
                    unique_ids.add(sku.unique_id)
                    skus.append(sku)
                product = ProductDetail(
                    **{key: row.get(key) for key in ProductDetail.model_fields if key != "skus"},
                    skus=skus,
                )
                if not skus or product.product_id in product_ids:
                    raise ShopError("shop_invalid_response")
                product_ids.add(product.product_id)
                products.append(product)
            result = ProductSearchResult(products=products, page=data.get("page"),
                                         limit=data.get("limit"), total=data.get("total"))
        except ValidationError:
            raise ShopError("shop_invalid_response") from None
        if (result.page != query.page or result.limit != query.limit
                or len(products) > query.limit or result.total < len(products)
                or (products and (query.page - 1) * query.limit + len(products) > result.total)):
            raise ShopError("shop_invalid_response")
        return result

    async def get_product_detail(self, product_id: int, authorization: str) -> ProductDetail:
        if type(product_id) is not int or product_id <= 0:
            raise ShopError("invalid_request")
        data = await self._get_data(f"product/detail/{product_id}", authorization)
        return self._detail(data, product_id)

    @staticmethod
    def _detail(data: dict[str, Any], product_id: int) -> ProductDetail:
        store = data.get("storeInfo")
        values = data.get("productValue")
        if not isinstance(store, dict) or not isinstance(values, dict):
            raise ShopError("shop_invalid_response")
        if type(store.get("id")) is not int or store["id"] != product_id:
            raise ShopError("shop_invalid_response")
        if store.get("isIntegral") == 1:
            raise ShopError("unsupported_product")
        if type(store.get("isIntegral")) is not int or store["isIntegral"] != 0:
            raise ShopError("shop_invalid_response")

        skus = []
        unique_ids = set()
        try:
            for name, value in values.items():
                if not isinstance(value, dict):
                    raise ShopError("shop_invalid_response")
                if type(value.get("productId")) is not int or value["productId"] != product_id:
                    raise ShopError("shop_invalid_response")
                sku = ProductSKU(
                    name=value.get("sku") or name,
                    unique_id=value.get("unique"),
                    price=value.get("price"),
                    stock=value.get("stock"),
                    image=value.get("image") or store.get("image") or "",
                )
                if sku.unique_id in unique_ids:
                    raise ShopError("shop_invalid_response")
                unique_ids.add(sku.unique_id)
                skus.append(sku)
            return ProductDetail(
                product_id=store["id"],
                name=store.get("storeName"),
                description=store.get("storeInfo") or "",
                image=store.get("image") or "",
                skus=skus,
            )
        except ValidationError:
            raise ShopError("shop_invalid_response") from None

    async def get_cart_product(self, product_id: int, unique_id: str,
                               authorization: str, *, public_secrets: tuple[str, ...] = ()) -> ProductDetail:
        if type(product_id) is not int or product_id <= 0 or not isinstance(unique_id, str) or not unique_id:
            raise ShopError("invalid_request")
        data = await self._get_data(f"product/detail/{product_id}", authorization)
        product = self._detail(data, product_id)
        store = data["storeInfo"]
        if type(store.get("isShow")) is not int:
            raise ShopError("shop_invalid_response")
        if store["isShow"] != 1:
            raise ShopError("unsupported_product")
        if not any(sku.unique_id == unique_id for sku in product.skus):
            raise ShopError("sku_unavailable")
        # The detail VO has no deletion flag. The ordinary-product query is
        # authoritative for ordinary, undeleted, in-sale product/SKU membership.
        keyword = _qualification_keyword(product.name, (
            authorization, authorization.partition(" ")[2], *public_secrets,
        ))
        eligible = await self.search_products(
            ProductSearchInput(keyword=keyword, limit=20), authorization)
        matches = [item for item in eligible.products if item.product_id == product_id]
        if len(matches) != 1:
            raise ShopError("unsupported_product")
        if not any(sku.unique_id == unique_id for sku in matches[0].skus):
            raise ShopError("sku_unavailable")
        return product

    async def get_cart(self, authorization: str) -> CartSnapshot:
        data = await self._get_data("cart/list", authorization)
        if not isinstance(data.get("valid"), list) or not isinstance(data.get("invalid"), list):
            raise ShopError("shop_invalid_response")
        groups: dict[str, list[CartItem]] = {"valid": [], "invalid": []}
        cart_ids = set()
        try:
            for group, items in groups.items():
                for row in data[group]:
                    item = self._cart_item(row, valid=group == "valid")
                    if item.cart_id in cart_ids:
                        raise ShopError("shop_invalid_response")
                    cart_ids.add(item.cart_id)
                    items.append(item)
            return CartSnapshot(**groups)
        except ValidationError:
            raise ShopError("shop_invalid_response") from None

    @staticmethod
    def _cart_item(row: Any, *, valid: bool) -> CartItem:
        if not isinstance(row, dict) or not isinstance(row.get("productInfo"), dict):
            raise ShopError("shop_invalid_response")
        product = row["productInfo"]
        attr = product.get("attrInfo")
        if attr is not None and not isinstance(attr, dict):
            raise ShopError("shop_invalid_response")
        attr = attr if attr is not None else {}
        for info in (product, attr):
            key = "id" if info is product else "productId"
            if key in info and (type(info[key]) is not int or info[key] != row.get("productId")):
                raise ShopError("shop_invalid_response")

        unique_ids = []
        for unique in (row.get("productAttrUnique"), attr.get("unique")):
            if unique is not None and not isinstance(unique, str):
                raise ShopError("shop_invalid_response")
            if unique is not None and unique.strip():
                unique_ids.append(unique.strip())
        if len(set(unique_ids)) > 1:
            raise ShopError("shop_invalid_response")
        if valid and (row.get("truePrice") is None or row.get("trueStock") is None):
            raise ShopError("shop_invalid_response")
        return CartItem(
            cart_id=row.get("id"), product_id=row.get("productId"),
            unique_id=unique_ids[0] if unique_ids else None,
            quantity=row.get("cartNum"), name=product.get("storeName"),
            sku_name=attr.get("sku") or "", image=product.get("image") or "",
            price=row.get("truePrice"), stock=row.get("trueStock"),
        )

    async def add_to_cart(self, item: CartAddInput, authorization: str) -> CartAddResult:
        try:
            if not isinstance(item, CartAddInput):
                raise ShopError("invalid_request")
            item = CartAddInput.model_validate(item.model_dump(warnings=False))
        except ValidationError:
            raise ShopError("invalid_request") from None
        # A write has exactly one transport attempt, including uncertain results.
        try:
            response = await self.http.post("cart/add", headers={"Authorization": authorization}, json={
                "productId": item.product_id, "uniqueId": item.unique_id, "cartNum": item.quantity,
                "new": 0, "combinationId": 0, "secKillId": 0,
            })
        except httpx.RequestError:
            raise ShopError("cart_result_unknown", outcome="unknown") from None
        if response.status_code == 401:
            raise ShopError("login_required", outcome="failed")
        if response.status_code >= 500 or response.is_redirect:
            raise ShopError("cart_result_unknown", outcome="unknown")
        try:
            payload = json.loads(response.content, parse_float=Decimal,
                                 object_pairs_hook=_unique_json_object)
        except (ValueError, UnicodeError):
            raise ShopError("cart_result_unknown", outcome="unknown") from None
        if not isinstance(payload, dict):
            raise ShopError("cart_result_unknown", outcome="unknown")
        status, success = payload.get("status"), payload.get("success")
        if type(status) is int and status in {401, 5104, 5107}:
            raise ShopError("login_required", outcome="failed")
        if type(status) is not int or type(success) is not bool:
            raise ShopError("cart_result_unknown", outcome="unknown")
        # Java 5100 is a system exception; 5101 is a confirmed business rejection.
        confirmed_rejections = {403, 404, 5001, 5002, 5003, 5101, 5105, 5106, 5108, 5109}
        if status in confirmed_rejections and success is False:
            raise ShopError("shop_rejected", outcome="failed")
        if not response.is_success or status != 200 or success is not True:
            raise ShopError("cart_result_unknown", outcome="unknown")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ShopError("cart_result_unknown", outcome="unknown")
        try:
            return CartAddResult(cart_id=data.get("cartId"))
        except ValidationError:
            raise ShopError("cart_result_unknown", outcome="unknown") from None
