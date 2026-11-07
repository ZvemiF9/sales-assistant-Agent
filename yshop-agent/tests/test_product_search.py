"""Run the real query validation, adapter and API with only HTTP replaced."""

from copy import deepcopy
from decimal import Decimal

import httpx
import pytest
from pydantic import ValidationError

from app.errors import ShopError
from app import schemas
from app.tools.yshop import YShopClient
from tests.test_api import make_client


AUTH = "Bearer search-test-token"


def query(**values):
    assert hasattr(schemas, "ProductSearchInput"), "budget query validation is missing"
    return schemas.ProductSearchInput(**values)


def payload():
    return {"products": [{
        "product_id": 7, "name": "保温杯", "description": "实用礼物", "image": "cup.png",
        "cost": "secret-cost", "skus": [{"name": "红色", "unique_id": "sku-red",
            "price": "29.99", "stock": 8, "image": "red.png", "cost": "secret-cost"}],
    }], "page": 1, "limit": 10, "total": 1, "internal": "secret-value"}


def ok(data):
    return httpx.Response(200, json={"status": 200, "success": True, "data": data})


async def call(handler, search):
    async with httpx.AsyncClient(base_url="http://shop.test/api/",
                                transport=httpx.MockTransport(handler)) as http:
        shop = YShopClient(http)
        assert hasattr(shop, "search_products"), "budget query adapter is missing"
        return await shop.search_products(search, AUTH)


def test_query_defaults_equal_bounds_and_normalized_keyword():
    search = query(keyword="  保温杯  ", min_price="29.99", max_price="29.99")
    assert search.keyword == "保温杯"
    assert search.min_price == search.max_price == Decimal("29.99")
    assert (search.page, search.limit, search.category_id) == (1, 10, None)


@pytest.mark.parametrize("values", [
    {"min_price": "-0.01"}, {"max_price": "-1"}, {"min_price": "NaN"},
    {"max_price": "Infinity"}, {"min_price": "30", "max_price": "29.99"},
    {"page": 0}, {"page": -1}, {"page": True}, {"page": 1.5},
    {"limit": 0}, {"limit": 21}, {"limit": True}, {"category_id": 0},
    {"category_id": True}, {"category_id": "1,11"}, {"category_id": 9223372036854775808},
    {"page": 2147483648}, {"keyword": "x" * 101},
])
def test_query_rejects_invalid_business_parameters(values):
    assert hasattr(schemas, "ProductSearchInput"), "budget query validation is missing"
    with pytest.raises(ValidationError):
        schemas.ProductSearchInput(**values)


async def test_adapter_sends_exact_query_and_only_returns_public_fields():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/agent/products"
        assert request.headers["Authorization"] == AUTH
        assert dict(request.url.params) == {
            "keyword": "礼物%_&", "category_id": "11", "min_price": "29.9900",
            "max_price": "300.01", "page": "1", "limit": "10",
        }
        return ok(payload())

    result = await call(handler, query(keyword="礼物%_&", category_id=11,
                                     min_price="29.9900", max_price="300.01"))
    assert result.products[0].skus[0].price == Decimal("29.99")
    public = result.model_dump_json()
    assert "secret" not in public and "cost" not in public and "internal" not in public
    assert result.total == 1


async def test_missing_bounds_are_omitted_and_empty_result_is_success():
    def handler(request):
        assert dict(request.url.params) == {"page": "2", "limit": "3"}
        return ok({"products": [], "page": 2, "limit": 3, "total": 0})

    result = await call(handler, query(page=2, limit=3))
    assert result.products == [] and result.total == 0


@pytest.mark.parametrize("defect", [
    "out_of_stock", "over_budget", "under_budget", "duplicate_product", "duplicate_sku",
    "empty_skus", "blank_unique", "negative_price", "wrong_page", "wrong_limit",
    "negative_total", "boolean_total", "too_many_products", "invalid_products",
])
async def test_adapter_rejects_untrustworthy_search_results(defect):
    data = payload()
    product = data["products"][0]
    sku = product["skus"][0]
    if defect == "out_of_stock":
        sku["stock"] = 0
    elif defect == "over_budget":
        sku["price"] = "30.01"
    elif defect == "under_budget":
        sku["price"] = "29.98"
    elif defect == "duplicate_product":
        data["products"].append(deepcopy(product))
        data["total"] = 2
    elif defect == "duplicate_sku":
        product["skus"].append(deepcopy(sku))
    elif defect == "empty_skus":
        product["skus"] = []
    elif defect == "blank_unique":
        sku["unique_id"] = " "
    elif defect == "negative_price":
        sku["price"] = "-1"
    elif defect == "wrong_page":
        data["page"] = 2
    elif defect == "wrong_limit":
        data["limit"] = 20
    elif defect == "negative_total":
        data["total"] = -1
    elif defect == "boolean_total":
        data["total"] = True
    elif defect == "too_many_products":
        data["products"] = [dict(deepcopy(product), product_id=i) for i in range(1, 12)]
        data["total"] = 11
    else:
        data["products"] = "secret-invalid"
    with pytest.raises(ShopError) as caught:
        await call(lambda request: ok(data), query(min_price="29.99", max_price="30"))
    assert caught.value.code == "shop_invalid_response"
    assert "secret" not in str(caught.value)


async def test_read_retries_once_for_transient_failure():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("secret", request=request)
        return ok(payload())

    result = await call(handler, query())
    assert result.total == 1 and len(calls) == 2


async def test_invalid_response_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return ok({"products": []})

    with pytest.raises(ShopError):
        await call(handler, query())
    assert len(calls) == 1


async def test_adapter_rejects_page_that_extends_past_reported_total():
    data = payload()
    product = data["products"][0]
    data.update(page=2, total=11,
                products=[dict(deepcopy(product), product_id=i) for i in range(1, 11)])
    with pytest.raises(ShopError) as caught:
        await call(lambda request: ok(data), query(page=2))
    assert caught.value.code == "shop_invalid_response"


def test_api_checks_login_before_search_and_preserves_response_contract():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.headers["Authorization"] == AUTH
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        assert request.url.path == "/api/agent/products"
        assert request.url.params["max_price"] == "29.99"
        return ok(payload())

    client = make_client(handler)
    try:
        assert client.get("/agent/products").status_code == 401
        assert paths == []
        response = client.get("/agent/products?max_price=29.99", headers={"Authorization": AUTH})
        assert response.status_code == 200
        assert paths == ["/api/userinfo", "/api/agent/products"]
        assert response.json()["products"][0]["skus"][0]["price"] == "29.99"
        assert "secret" not in response.text
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)


@pytest.mark.parametrize("params", ["min_price=-1", "min_price=30&max_price=29.99",
    "max_price=NaN", "max_price=Infinity", "category_id=0", "limit=21", "page=0",
    "page=1.5", "keyword=" + "x" * 101])
def test_api_invalid_query_is_sanitized_and_does_not_search(params):
    def handler(request):
        assert request.url.path == "/api/userinfo"
        return ok({"uid": 33})

    client = make_client(handler)
    try:
        response = client.get("/agent/products?" + params, headers={"Authorization": AUTH})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert AUTH not in response.text
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)


def test_api_expired_identity_stops_search():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"status": 5107, "success": False,
                                        "data": None, "msg": AUTH})

    client = make_client(handler)
    try:
        response = client.get("/agent/products", headers={"Authorization": AUTH})
        assert response.status_code == 401
        assert paths == ["/api/userinfo"]
        assert AUTH not in response.text
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)
