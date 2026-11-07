from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.schemas import CartAddInput
from app.tools.yshop import YShopClient


@contextmanager
def api_client(handler):
    application = create_app(Settings(_env_file=None))
    with TestClient(application) as client:
        http = httpx.AsyncClient(
            base_url="http://shop.test/api/", transport=httpx.MockTransport(handler)
        )
        application.state.shop = YShopClient(http)
        try:
            yield client
        finally:
            client.portal.call(http.aclose)


def test_settings_defaults_are_local_and_do_not_enable_cors(monkeypatch):
    for name in ("YSHOP_API_URL", "HTTP_TIMEOUT_SECONDS", "CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.model_dump() == {
        "yshop_api_url": "http://127.0.0.1:8008/api/",
        "http_timeout_seconds": 10.0,
        "cors_origins": [],
    }


def test_settings_load_service_dotenv_regardless_of_working_directory(tmp_path, monkeypatch):
    for name in ("YSHOP_API_URL", "HTTP_TIMEOUT_SECONDS", "CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)
    before = Settings()
    (tmp_path / ".env").write_text(
        "YSHOP_API_URL=http://unrelated.test/api/\nHTTP_TIMEOUT_SECONDS=3.0\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert Settings().model_dump() == before.model_dump()


def test_cli_credentials_are_not_kept_in_service_settings(monkeypatch):
    monkeypatch.setenv("YSHOP_AUTHORIZATION", "Bearer settings-secret")
    settings = Settings(_env_file=None)
    assert "settings-secret" not in repr(settings)
    assert set(settings.model_dump()) == {
        "yshop_api_url", "http_timeout_seconds", "cors_origins"
    }


def test_invalid_setting_never_displays_embedded_credentials():
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, yshop_api_url="http://user:embedded-secret@shop.test/api/")
    assert "embedded-secret" not in str(caught.value)
    assert "embedded-secret" not in repr(caught.value)


@pytest.mark.parametrize("origin", [
    "*", "http://user:origin-secret@h5.test", "http://h5.test/path",
    "http://h5.test?token=origin-secret", "not-an-origin",
])
def test_cors_configuration_requires_an_actual_origin_and_hides_invalid_input(origin):
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, cors_origins=[origin])
    assert "origin-secret" not in str(caught.value)


def test_api_url_normalizes_missing_trailing_slash():
    settings = Settings(_env_file=None, yshop_api_url="http://shop.test/api")
    assert settings.yshop_api_url == "http://shop.test/api/"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, http_timeout_seconds=timeout)


def test_health_is_local_and_default_cors_does_not_allow_browser_origins():
    def handler(request):
        pytest.fail("health must not contact YShop")

    with api_client(handler) as client:
        response = client.get("/health", headers={"Origin": "http://h5.test"})
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "yshop-agent"}
        assert "access-control-allow-origin" not in response.headers


def test_configured_h5_origin_is_allowed():
    application = create_app(Settings(_env_file=None, cors_origins=["http://h5.test"]))
    with TestClient(application) as client:
        response = client.options("/agent/me", headers={
            "Origin": "http://h5.test",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        })
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://h5.test"


def test_app_lifespan_owns_and_closes_the_shared_http_connection_pool():
    application = create_app(Settings(_env_file=None, http_timeout_seconds=3.0))
    with TestClient(application):
        http = application.state.shop.http
        assert not http.is_closed
        assert str(http.base_url) == "http://127.0.0.1:8008/api/"
        assert http.timeout.read == 3.0
        assert "authorization" not in http.headers
    assert http.is_closed


def test_shop_connection_pool_does_not_use_environment_proxies(monkeypatch):
    # Bearer-authenticated shop requests must use the configured shop connection.
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.test:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:8080")
    calls = []
    real_client = httpx.AsyncClient

    def make_http_client(*args, **kwargs):
        client = real_client(*args, **kwargs)
        calls.append((client, kwargs))
        return client

    monkeypatch.setattr("app.main.httpx.AsyncClient", make_http_client)
    settings = Settings(_env_file=None)
    application = create_app(settings)
    with TestClient(application):
        shop_calls = [kwargs for client, kwargs in calls if client is application.state.shop.http]
        assert len(shop_calls) == 1
        assert shop_calls[0]["base_url"] == settings.yshop_api_url
        assert shop_calls[0].get("trust_env") is False
        assert shop_calls[0].get("proxy") is None


@pytest.mark.parametrize("authorization", [None, "", "Basic token", "Bearer", "Bearer a b"])
def test_identity_header_is_checked_before_any_shop_read(authorization):
    def handler(request):
        pytest.fail("invalid authorization must not contact YShop")

    with api_client(handler) as client:
        headers = {} if authorization is None else {"Authorization": authorization}
        for path in ("/agent/me", "/agent/products/7"):
            response = client.get(path, headers=headers)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "login_required"


def test_duplicate_authorization_headers_are_rejected_before_shop_access():
    def handler(request):
        pytest.fail("duplicate authorization must not contact YShop")

    with api_client(handler) as client:
        response = client.get("/agent/me", headers=[
            ("Authorization", "Bearer user-a"), ("Authorization", "Bearer user-b")
        ])
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "login_required"


def test_product_flow_uses_current_request_identity_and_live_sku_data():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer product-flow-secret"
        if request.url.path == "/api/userinfo":
            data = {"uid": 33, "nickname": "测试用户", "password": "private-hash"}
        else:
            assert request.url.path == "/api/product/detail/7"
            data = {
                "storeInfo": {"id": 7, "storeName": "保温杯", "storeInfo": "日常使用",
                              "image": "cup.jpg", "isShow": 1, "isIntegral": 0},
                "productValue": {
                    "红色": {"id": 101, "productId": 7, "unique": "sku-red",
                             "sku": "红色", "price": "29.99", "stock": 8,
                             "cost": "3.50", "image": ""}
                },
                "uid": 999,
            }
        return httpx.Response(200, json={"status": 200, "success": True, "data": data})

    with api_client(handler) as client:
        response = client.get("/agent/products/7?uid=999",
                              headers={"Authorization": "Bearer product-flow-secret"})
        assert response.status_code == 200
        assert paths == ["/api/userinfo", "/api/product/detail/7"]
        assert response.json() == {
            "product_id": 7, "name": "保温杯", "description": "日常使用",
            "image": "cup.jpg", "skus": [
                {"name": "红色", "unique_id": "sku-red", "price": "29.99",
                 "stock": 8, "image": "cup.jpg"}
            ],
        }


def test_validation_response_does_not_reflect_raw_request_input():
    def handler(request):
        assert request.url.path == "/api/userinfo"
        return httpx.Response(200, json={"status": 200, "success": True,
                                        "data": {"uid": 33, "nickname": ""}})

    with api_client(handler) as client:
        response = client.get("/agent/products/request-secret",
                              headers={"Authorization": "Bearer valid-secret"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert "request-secret" not in response.text
        assert "valid-secret" not in response.text
        assert "input" not in response.text


@pytest.mark.parametrize("product_id", [0, -1, True])
def test_input_product_id_must_be_a_positive_integer(product_id):
    with pytest.raises(ValidationError):
        CartAddInput(product_id=product_id, unique_id="sku-red")
