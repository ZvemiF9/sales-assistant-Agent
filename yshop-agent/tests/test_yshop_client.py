import asyncio
from decimal import Decimal

import httpx
import pytest

from app.errors import ShopError
from app.schemas import CartAddInput
from app.tools.yshop import YShopClient


AUTHORIZATION = "Bearer test-token-not-for-logging"


def ok(data):
    return httpx.Response(200, json={
        "status": 200, "success": True, "msg": "操作成功",
        "total": None, "totalPage": None, "data": data,
        "time": "2026-10-23 10:00:00",
    })


def detail():
    return {
        "storeInfo": {
            "id": 7, "storeName": "保温杯", "storeInfo": "日常使用",
            "image": "https://example.com/cup.jpg", "price": "19.90",
            "stock": 8, "isShow": 1, "isIntegral": 0,
            "cost": "2.00", "description": "内部字段",
        },
        "productAttr": [{"attrName": "颜色", "attrValues": "红,蓝"}],
        "productValue": {
            "红色": {"id": 101, "productId": 7, "sku": "红色", "unique": "sku-red",
                     "price": "29.99", "stock": 8, "cost": "3.50",
                     "image": "https://example.com/red.jpg"},
            "蓝色": {"id": 102, "productId": 7, "sku": "蓝色", "unique": "sku-blue",
                     "price": "35.10", "stock": 0, "cost": "4.00", "image": ""},
        },
        "mapKey": "private-map-key", "uid": 33, "goodList": [],
        "reply": {"nickname": "另一个用户"},
    }


async def with_shop(handler, operation):
    async with httpx.AsyncClient(
        base_url="http://shop.test/api/", transport=httpx.MockTransport(handler),
    ) as http:
        return await operation(YShopClient(http))


async def test_user_identity_comes_from_shop_and_excludes_personal_fields():
    def handler(request):
        assert request.url.path == "/api/userinfo"
        assert request.headers["Authorization"] == AUTHORIZATION
        return ok({"uid": 33, "nickname": "测试用户", "password": "password-hash",
                   "phone": "13800000000", "nowMoney": "100.00"})

    user = await with_shop(handler, lambda shop: shop.get_user(AUTHORIZATION))
    assert user.model_dump() == {"user_id": 33, "nickname": "测试用户"}


@pytest.mark.parametrize("status", [401, 5104, 5107])
async def test_http_200_with_expired_identity_is_not_treated_as_success(status):
    def handler(request):
        return httpx.Response(200, json={"status": status, "success": False,
                                        "msg": AUTHORIZATION, "data": None})

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.get_user(AUTHORIZATION))
    assert caught.value.http_status == 401
    assert AUTHORIZATION not in str(caught.value)


async def test_false_success_flag_never_authenticates_user():
    def handler(request):
        return httpx.Response(200, json={"status": 200, "success": False,
                                        "data": {"uid": 33}})

    with pytest.raises(ShopError):
        await with_shop(handler, lambda shop: shop.get_user(AUTHORIZATION))


async def test_product_uses_current_sku_prices_and_outputs_only_allowed_fields():
    product = await with_shop(lambda request: ok(detail()),
                              lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    result = product.model_dump(mode="json")
    assert result == {
        "product_id": 7, "name": "保温杯", "description": "日常使用",
        "image": "https://example.com/cup.jpg",
        "skus": [
            {"name": "红色", "unique_id": "sku-red", "price": "29.99",
             "stock": 8, "image": "https://example.com/red.jpg"},
            {"name": "蓝色", "unique_id": "sku-blue", "price": "35.10",
             "stock": 0, "image": "https://example.com/cup.jpg"},
        ],
    }
    assert product.skus[0].price == Decimal("29.99")


async def test_missing_sku_id_is_rejected_instead_of_using_database_row_id():
    data = detail()
    data["productValue"]["红色"]["unique"] = ""
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data),
                        lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"


async def test_sku_from_another_product_is_rejected():
    data = detail()
    data["productValue"]["红色"]["productId"] = 99
    with pytest.raises(ShopError):
        await with_shop(lambda request: ok(data),
                        lambda shop: shop.get_product_detail(7, AUTHORIZATION))


async def test_integral_product_cannot_be_used_as_ordinary_product():
    data = detail()
    data["storeInfo"]["isIntegral"] = 1
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data),
                        lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    assert caught.value.code == "unsupported_product"


async def test_read_retries_once_after_temporary_network_failure():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("temporary", request=request)
        return ok({"uid": 33, "nickname": "测试用户"})

    user = await with_shop(handler, lambda shop: shop.get_user(AUTHORIZATION))
    assert user.user_id == 33
    assert len(requests) == 2


async def test_read_stops_after_second_network_failure():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ConnectError("temporary", request=request)

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.get_user(AUTHORIZATION))
    assert len(requests) == 2
    assert caught.value.code == "shop_unavailable"


async def test_corrupted_gzip_read_is_invalid_response_without_retry_or_credentials():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, headers={"Content-Encoding": "gzip"},
                              stream=httpx.ByteStream(b"invalid-gzip" + AUTHORIZATION.encode()))

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.get_user(AUTHORIZATION))
    assert requests == [("GET", "/api/userinfo")]
    assert caught.value.code == "shop_invalid_response"
    assert AUTHORIZATION not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


async def test_concurrent_requests_do_not_share_authorization_headers():
    seen = []

    async def handler(request):
        token = request.headers["Authorization"]
        await asyncio.sleep(0)
        seen.append(token)
        return ok({"uid": 1 if token == "Bearer user-a" else 2, "nickname": ""})

    async def operation(shop):
        return await asyncio.gather(shop.get_user("Bearer user-a"),
                                    shop.get_user("Bearer user-b"))

    users = await with_shop(handler, operation)
    assert [user.user_id for user in users] == [1, 2]
    assert sorted(seen) == ["Bearer user-a", "Bearer user-b"]


@pytest.mark.parametrize("data", [{"id": 33}, {"uid": True}, {"uid": 0}, {"uid": None}])
async def test_user_identity_requires_the_real_positive_shop_uid(data):
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data), lambda shop: shop.get_user(AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"


@pytest.mark.parametrize("body", [
    {"status": 200, "data": {"uid": 33}},
    {"status": "200", "success": True, "data": {"uid": 33}},
    {"status": 200, "success": "true", "data": {"uid": 33}},
    [AUTHORIZATION],
])
async def test_malformed_shop_envelope_never_authenticates_user(body):
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: httpx.Response(200, json=body),
                        lambda shop: shop.get_user(AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"
    assert AUTHORIZATION not in str(caught.value)


@pytest.mark.parametrize("failure", ["http_error", "invalid_json"])
async def test_read_errors_are_sanitized_and_business_responses_are_not_retried(failure):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503 if failure == "http_error" else 200, text=AUTHORIZATION)

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.get_user(AUTHORIZATION))
    assert len(requests) == 1
    assert AUTHORIZATION not in str(caught.value)
    assert caught.value.http_status == 502


async def test_product_response_id_must_match_requested_product():
    data = detail()
    data["storeInfo"]["id"] = 99
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data),
                        lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"


async def test_duplicate_current_sku_unique_ids_are_rejected():
    data = detail()
    data["productValue"]["蓝色"]["unique"] = "sku-red"
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data),
                        lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"


@pytest.mark.parametrize("stock", [-1, True, None])
async def test_invalid_current_sku_stock_is_rejected(stock):
    data = detail()
    data["productValue"]["红色"]["stock"] = stock
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data),
                        lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"


async def test_numeric_shop_prices_preserve_json_decimal_precision():
    import json

    body = json.dumps({"status": 200, "success": True, "data": detail()})
    body = body.replace('"29.99"', "0.12345678901234567890123456789")
    product = await with_shop(lambda request: httpx.Response(200, text=body),
                              lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    assert product.skus[0].price == Decimal("0.12345678901234567890123456789")


@pytest.mark.parametrize("price", [None, "-1.00", "NaN"])
async def test_invalid_current_sku_price_is_rejected_without_leaking_shop_input(price):
    data = detail()
    data["productValue"]["红色"]["price"] = price
    data["storeInfo"]["description"] = AUTHORIZATION
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data),
                        lambda shop: shop.get_product_detail(7, AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"
    assert AUTHORIZATION not in str(caught.value)


async def test_add_cart_sends_ordinary_product_parameters_and_returns_real_cart_id():
    import json

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/cart/add"
        assert request.headers["Authorization"] == AUTHORIZATION
        assert json.loads(request.content) == {
            "productId": 7, "uniqueId": "sku-red", "cartNum": 2,
            "new": 0, "combinationId": 0, "secKillId": 0,
        }
        return ok({"cartId": 123})

    item = CartAddInput(product_id=7, unique_id="sku-red", quantity=2)
    result = await with_shop(handler, lambda shop: shop.add_to_cart(item, AUTHORIZATION))
    assert result.model_dump() == {"cart_id": 123, "status": "succeeded"}


@pytest.mark.parametrize("quantity", [0, -1, 10000, 1.5, True])
def test_cart_rejects_invalid_quantity(quantity):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CartAddInput(product_id=7, unique_id="sku-red", quantity=quantity)


@pytest.mark.parametrize("unique_id", ["", " ", "\n"])
def test_cart_requires_real_specification_id(unique_id):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CartAddInput(product_id=7, unique_id=unique_id)


def test_cart_does_not_accept_client_user_id_or_promotion_overrides():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CartAddInput(product_id=7, unique_id="sku-red", uid=999, new=1)


@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "server_error"])
async def test_uncertain_cart_write_is_not_retried_and_reports_unknown(failure):
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout(AUTHORIZATION, request=request)
        if failure == "invalid_json":
            return httpx.Response(200, text=AUTHORIZATION)
        return httpx.Response(502, text=AUTHORIZATION)

    item = CartAddInput(product_id=7, unique_id="sku-red")
    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.add_to_cart(item, AUTHORIZATION))
    assert len(requests) == 1
    assert caught.value.code == "cart_result_unknown"
    assert caught.value.outcome == "unknown"
    assert AUTHORIZATION not in str(caught.value)


async def test_cart_business_rejection_is_failed_and_not_retried():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": 5101, "success": False,
                                        "msg": "库存不足", "data": None})

    item = CartAddInput(product_id=7, unique_id="sku-red")
    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.add_to_cart(item, AUTHORIZATION))
    assert len(requests) == 1
    assert caught.value.code == "shop_rejected"
    assert caught.value.outcome == "failed"


async def test_cart_snapshot_preserves_quantity_and_excludes_user_and_cost():
    data = {"valid": [{
        "id": 123, "uid": 33, "productId": 7, "productAttrUnique": "sku-red",
        "cartNum": 2, "truePrice": "29.99", "trueStock": 8, "costPrice": "3.50",
        "productInfo": {"storeName": "保温杯", "image": "cup.jpg",
                        "attrInfo": {"sku": "红色", "unique": "sku-red", "cost": "3.50"}},
    }], "invalid": []}
    cart = await with_shop(lambda request: ok(data), lambda shop: shop.get_cart(AUTHORIZATION))
    assert cart.model_dump(mode="json") == {
        "valid": [{"cart_id": 123, "product_id": 7, "unique_id": "sku-red",
                   "quantity": 2, "name": "保温杯", "sku_name": "红色",
                   "image": "cup.jpg", "price": "29.99", "stock": 8}],
        "invalid": [],
    }


@pytest.mark.parametrize("cart_id", [None, 0, -1, True, "123", 123.0, [], {}])
async def test_malformed_successful_cart_id_is_unknown_and_never_retried(cart_id):
    requests = []

    def handler(request):
        requests.append(request)
        return ok({"cartId": cart_id, "private": AUTHORIZATION})

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.add_to_cart(
            CartAddInput(product_id=7, unique_id="sku-red"), AUTHORIZATION))
    assert len(requests) == 1
    assert caught.value.code == "cart_result_unknown"
    assert caught.value.outcome == "unknown"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ or caught.value.__context__ is None
    assert AUTHORIZATION not in str(caught.value)


@pytest.mark.parametrize("body", [
    {"status": 200, "success": True, "data": {}},
    {"status": 200, "success": True, "data": None},
    {"status": 200, "success": True, "data": [123]},
    {"status": "200", "success": True, "data": {"cartId": 123}},
    {"status": 200, "success": "true", "data": {"cartId": 123}},
    {"status": 200, "success": False, "data": {"cartId": 123}},
    {"status": 5100, "success": True, "data": {"cartId": 123}},
    [AUTHORIZATION],
])
async def test_untrusted_cart_write_envelope_is_unknown(body):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=body)

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.add_to_cart(
            CartAddInput(product_id=7, unique_id="sku-red"), AUTHORIZATION))
    assert len(requests) == 1
    assert caught.value.outcome == "unknown"
    assert caught.value.code == "cart_result_unknown"


async def test_connect_error_cart_write_is_unknown_without_credential_error_chaining():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ConnectError(AUTHORIZATION, request=request)

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.add_to_cart(
            CartAddInput(product_id=7, unique_id="sku-red"), AUTHORIZATION))
    assert len(requests) == 1
    assert caught.value.outcome == "unknown"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("http_status, shop_status", [(401, 200), (200, 401), (200, 5104), (200, 5107)])
async def test_cart_authentication_failure_is_sanitized_401(http_status, shop_status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(http_status, json={"status": shop_status, "success": False,
                                                "msg": AUTHORIZATION, "data": None})

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.add_to_cart(
            CartAddInput(product_id=7, unique_id="sku-red"), AUTHORIZATION))
    assert len(requests) == 1
    assert caught.value.code == "login_required"
    assert caught.value.http_status == 401
    assert AUTHORIZATION not in str(caught.value)


@pytest.mark.parametrize("http_status", [302, 400, 502])
async def test_cart_http_error_with_untrusted_body_is_unknown(http_status):
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: httpx.Response(http_status, text=AUTHORIZATION),
                        lambda shop: shop.add_to_cart(
                            CartAddInput(product_id=7, unique_id="sku-red"), AUTHORIZATION))
    assert caught.value.code == "cart_result_unknown"
    assert caught.value.outcome == "unknown"


async def test_cart_revalidates_constructed_input_before_post():
    item = CartAddInput.model_construct(product_id=7, unique_id="sku-red", quantity=True)

    def handler(request):
        pytest.fail("invalid input must never reach the shop")

    with pytest.raises(ShopError) as caught:
        await with_shop(handler, lambda shop: shop.add_to_cart(item, AUTHORIZATION))
    assert caught.value.code == "invalid_request"


def cart_row():
    return {"id": 123, "productId": 7, "productAttrUnique": "sku-red", "cartNum": 2,
            "truePrice": "29.99", "trueStock": 8,
            "productInfo": {"id": 7, "storeName": "保温杯", "image": "cup.jpg",
                            "attrInfo": {"id": 101, "productId": 7, "sku": "红色",
                                         "unique": "sku-red"}}}


async def test_invalid_cart_item_preserves_unknown_price_stock_and_missing_legacy_sku_id():
    row = cart_row()
    row.update(productAttrUnique="", truePrice=None, trueStock=None, costPrice=AUTHORIZATION)
    row["productInfo"].update(attrInfo=None, price="99.00", stock=50)
    cart = await with_shop(lambda request: ok({"valid": [], "invalid": [row]}),
                           lambda shop: shop.get_cart(AUTHORIZATION))
    assert cart.model_dump(mode="json") == {"valid": [], "invalid": [{
        "cart_id": 123, "product_id": 7, "unique_id": None, "quantity": 2,
        "name": "保温杯", "sku_name": "", "image": "cup.jpg", "price": None, "stock": None,
    }]}


async def test_legacy_valid_cart_row_does_not_invent_a_sku_unique_from_row_id():
    row = cart_row()
    row.pop("productAttrUnique")
    row["productInfo"]["attrInfo"] = {"id": 101, "sku": "历史规格"}
    cart = await with_shop(lambda request: ok({"valid": [row], "invalid": []}),
                           lambda shop: shop.get_cart(AUTHORIZATION))
    assert cart.valid[0].unique_id is None


@pytest.mark.parametrize("data", [
    {}, {"valid": []}, {"valid": [], "invalid": None},
    {"valid": {}, "invalid": []}, {"valid": [None], "invalid": []},
    {"valid": [], "invalid": ["bad-row"]},
])
async def test_cart_list_requires_the_actual_valid_and_invalid_arrays(data):
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok(data), lambda shop: shop.get_cart(AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"


@pytest.mark.parametrize("field, value", [
    ("id", True), ("productId", 0), ("cartNum", 0), ("cartNum", "2"),
    ("truePrice", None), ("truePrice", "NaN"), ("trueStock", None),
    ("trueStock", True), ("productAttrUnique", 101), ("productInfo", None),
])
async def test_cart_structural_mismatch_is_sanitized(field, value):
    row = cart_row()
    row[field] = value
    row["private"] = AUTHORIZATION
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok({"valid": [row], "invalid": []}),
                        lambda shop: shop.get_cart(AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"
    assert AUTHORIZATION not in str(caught.value)


@pytest.mark.parametrize("mismatch", ["product", "sku_product", "sku_id", "attr_shape", "duplicate_cart"])
async def test_inconsistent_cart_product_or_sku_relationship_is_rejected(mismatch):
    row = cart_row()
    rows = [row]
    if mismatch == "product":
        row["productInfo"]["id"] = 99
    elif mismatch == "sku_product":
        row["productInfo"]["attrInfo"]["productId"] = 99
    elif mismatch == "sku_id":
        row["productInfo"]["attrInfo"]["unique"] = "sku-other"
    elif mismatch == "attr_shape":
        row["productInfo"]["attrInfo"] = []
    else:
        rows.append(row.copy())
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: ok({"valid": rows, "invalid": []}),
                        lambda shop: shop.get_cart(AUTHORIZATION))
    assert caught.value.code == "shop_invalid_response"


async def test_cart_read_uses_the_existing_single_network_retry():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert request.method == "GET"
        assert request.url.path == "/api/cart/list"
        if calls == 1:
            raise httpx.ReadTimeout(AUTHORIZATION, request=request)
        return ok({"valid": [], "invalid": []})

    cart = await with_shop(handler, lambda shop: shop.get_cart(AUTHORIZATION))
    assert cart.model_dump() == {"valid": [], "invalid": []}
    assert calls == 2


@pytest.mark.parametrize("status", [500, 5000, 5100, 5102, 9999, -1])
async def test_cart_body_system_or_unknown_error_code_does_not_prove_rejection(status):
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={
            "status": status, "success": False, "msg": AUTHORIZATION, "data": None})

    with pytest.raises(ShopError) as caught:
        await with_shop(handler,
            lambda shop: shop.add_to_cart(CartAddInput(product_id=7, unique_id="sku-red"), AUTHORIZATION))
    assert requests == [("POST", "/api/cart/add")]
    assert caught.value.code == "cart_result_unknown"
    assert caught.value.outcome == "unknown"
    assert AUTHORIZATION not in str(caught.value)


async def test_invalid_constructed_input_is_sanitized_without_serializer_warning():
    import warnings

    item = CartAddInput.model_construct(product_id=AUTHORIZATION, unique_id="sku-red", quantity=1)

    def handler(request):
        pytest.fail("constructed invalid input must not reach network")

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(ShopError) as caught:
            await with_shop(handler, lambda shop: shop.add_to_cart(item, AUTHORIZATION))
    assert caught.value.code == "invalid_request"
    assert emitted == []


@pytest.mark.parametrize("raw", [
    '{"status":5101,"status":200,"success":true,"data":{"cartId":123}}',
    '{"status":200,"success":true,"data":{"cartId":123,"cartId":456}}',
])
async def test_duplicate_write_json_keys_are_untrusted(raw):
    with pytest.raises(ShopError) as caught:
        await with_shop(lambda request: httpx.Response(200, text=raw),
                        lambda shop: shop.add_to_cart(CartAddInput(product_id=7, unique_id="sku-red"), AUTHORIZATION))
    assert caught.value.code == "cart_result_unknown"
    assert caught.value.outcome == "unknown"
