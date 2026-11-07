import httpx
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.config import Settings
from app.main import create_app
from app.tools.yshop import YShopClient


def make_client(handler):
    application = create_app(Settings(_env_file=None))
    client = TestClient(application)
    client.__enter__()
    # Replace only the external HTTP transport; keep real API and adapter code.
    client.http = httpx.AsyncClient(base_url="http://shop.test/api/",
                                  transport=httpx.MockTransport(handler))
    application.state.shop = YShopClient(client.http)
    return client


def test_test_lifespan_does_not_open_operational_default_session_database(
        monkeypatch, isolated_session_settings):
    import app.main as main_module
    from app.storage.actions import CartActionStore

    attempted_paths = []

    def redirected_store(path):
        attempted_paths.append(Path(path).resolve())
        return CartActionStore(isolated_session_settings.db_path)

    monkeypatch.setattr(main_module, "CartActionStore", redirected_store)
    application = create_app(Settings(_env_file=None))
    with TestClient(application):
        pass
    assert attempted_paths == [isolated_session_settings.db_path]


@pytest.mark.parametrize("authorization", [None, "", "Basic token", "Bearer", "Bearer a b"])
def test_protected_routes_reject_missing_or_malformed_identity_without_shop_request(authorization):
    def handler(request):
        pytest.fail("invalid identity must be rejected before contacting shop")

    client = make_client(handler)
    try:
        headers = {} if authorization is None else {"Authorization": authorization}
        for path in ["/agent/me", "/agent/products/7", "/agent/cart"]:
            response = client.get(path, headers=headers)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "login_required"
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)


def test_identity_response_and_expired_token_never_expose_credentials():
    def handler(request):
        if request.headers["Authorization"] == "Bearer expired-secret":
            return httpx.Response(200, json={"status": 401, "success": False,
                                            "msg": "expired-secret", "data": None})
        return httpx.Response(200, json={"status": 200, "success": True,
                                        "data": {"uid": 33, "nickname": "测试用户",
                                                 "phone": "13800000000", "password": "hash"}})

    client = make_client(handler)
    try:
        response = client.get("/agent/me", headers={"Authorization": "Bearer valid-secret"})
        assert response.status_code == 200
        assert response.json() == {"user_id": 33, "nickname": "测试用户"}
        response = client.get("/agent/me", headers={"Authorization": "Bearer expired-secret"})
        assert response.status_code == 401
        assert "expired-secret" not in response.text
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)


def test_product_request_checks_login_before_fetching_product():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"status": 401, "success": False, "data": None})

    client = make_client(handler)
    try:
        response = client.get("/agent/products/7", headers={"Authorization": "Bearer expired"})
        assert response.status_code == 401
        assert paths == ["/api/userinfo"]
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)


def test_health_is_local_and_service_does_not_expose_an_unconfirmed_cart_write():
    def handler(request):
        pytest.fail("health must not contact shop")

    client = make_client(handler)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "yshop-agent"}
        response = client.post("/agent/cart", json={"product_id": 7, "unique_id": "sku-red"})
        assert response.status_code == 405
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)


def test_cart_route_checks_identity_and_only_outputs_allowed_cart_fields():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer cart-route-secret"
        if request.url.path == "/api/userinfo":
            data = {"uid": 33, "nickname": "测试用户", "password": "private-password"}
        else:
            assert request.url.path == "/api/cart/list"
            data = {"valid": [], "invalid": [{
                "id": 123, "uid": 33, "productId": 7, "productAttrUnique": "sku-red",
                "cartNum": 2, "truePrice": None, "trueStock": None, "costPrice": "3.50",
                "productInfo": {"storeName": "保温杯", "image": "cup.jpg", "attrInfo": None},
            }]}
        return httpx.Response(200, json={"status": 200, "success": True, "data": data})

    client = make_client(handler)
    try:
        response = client.get("/agent/cart?uid=999", headers={"Authorization": "Bearer cart-route-secret"})
        assert response.status_code == 200
        assert paths == ["/api/userinfo", "/api/cart/list"]
        assert response.json() == {"valid": [], "invalid": [{
            "cart_id": 123, "product_id": 7, "unique_id": "sku-red", "quantity": 2,
            "name": "保温杯", "sku_name": "", "image": "cup.jpg", "price": None, "stock": None,
        }]}
        assert "private-password" not in response.text
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)


def test_cart_route_stops_before_cart_read_when_shop_rejects_identity():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"status": 5107, "success": False,
                                        "msg": "cart-expired-secret", "data": None})

    client = make_client(handler)
    try:
        response = client.get("/agent/cart", headers={"Authorization": "Bearer cart-expired-secret"})
        assert response.status_code == 401
        assert paths == ["/api/userinfo"]
        assert "cart-expired-secret" not in response.text
    finally:
        client.portal.call(client.http.aclose)
        client.__exit__(None, None, None)
