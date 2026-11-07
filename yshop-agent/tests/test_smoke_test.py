import importlib
import io
import json

import httpx
import pytest

from app.config import Settings


AUTHORIZATION = "Bearer smoke-private-token"
WRITE_ARGS = ["--product-id", "7", "--unique-id", "sku-red", "--add-to-cart", "--confirm"]


def ok(data):
    return httpx.Response(200, json={"status": 200, "success": True,
                                    "msg": "safe result", "data": data})


def product(stock=8):
    return {"storeInfo": {"id": 7, "isIntegral": 0, "isShow": 1,
                          "storeName": "保温杯", "storeInfo": "日常使用", "image": "cup.jpg"},
            "productValue": {"红色": {"id": 101, "productId": 7, "unique": "sku-red",
                                     "sku": "红色", "price": "29.99", "stock": stock,
                                     "image": "", "cost": "private-cost"}},
            "private": AUTHORIZATION}


def row(quantity=2, *, cart_id=123, product_id=7, unique_id="sku-red"):
    return {"id": cart_id, "uid": 33, "productId": product_id,
            "productAttrUnique": unique_id, "cartNum": quantity,
            "truePrice": "29.99", "trueStock": 8, "costPrice": "private-cost",
            "productInfo": {"storeName": "保温杯", "image": "cup.jpg",
                            "attrInfo": {"sku": "红色", "unique": unique_id}}}


def run_cli(tmp_path, handler, argv=None, *, environ=None, env_file=None):
    try:
        smoke = importlib.import_module("scripts.smoke_test")
    except ModuleNotFoundError:
        pytest.fail("the required real smoke-test CLI is not implemented")
    report_path = tmp_path / "data" / "day1-smoke-report.json"
    stdout = io.StringIO()
    code = smoke.main(
        ["--product-id", "7"] if argv is None else argv,
        environ={"YSHOP_AUTHORIZATION": AUTHORIZATION} if environ is None else environ,
        env_file=tmp_path / "missing.env" if env_file is None else env_file,
        settings=Settings(_env_file=None, yshop_api_url="http://shop.test/api/"),
        transport=httpx.MockTransport(handler), report_path=report_path, stdout=stdout,
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert AUTHORIZATION not in report_text + stdout.getvalue()
    assert "smoke-private-token" not in report_text + stdout.getvalue()
    return code, json.loads(report_text), stdout.getvalue()


def test_default_cli_reads_identity_live_skus_and_cart_without_writing(tmp_path):
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        assert request.headers["Authorization"] == AUTHORIZATION
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33, "nickname": "private-name", "password": "private-password"})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        assert request.url.path == "/api/cart/list"
        return ok({"valid": [row()], "invalid": []})

    code, report, output = run_cli(tmp_path, handler)
    assert code == 0
    assert requests == [("GET", "/api/userinfo"), ("GET", "/api/product/detail/7"),
                        ("GET", "/api/cart/list")]
    assert report["status"] == "succeeded"
    assert report["identity"] == {"user_id": 33}
    assert report["write"]["status"] == "not_requested"
    assert report["product"]["skus"][0]["unique_id"] == "sku-red"
    assert report["cart_before"]["valid"][0]["quantity"] == 2
    for private in ("private-name", "private-password", "private-cost", '"uid"'):
        assert private not in json.dumps(report) + output


@pytest.mark.parametrize("authorization", [None, "", "Basic secret", "Bearer", "Bearer a b", "Bearer a\n"])
def test_missing_or_malformed_cli_credentials_make_no_network_requests(tmp_path, authorization):
    def handler(request):
        pytest.fail("invalid credentials must not contact the shop")

    environ = {} if authorization is None else {"YSHOP_AUTHORIZATION": authorization}
    code, report, _ = run_cli(tmp_path, handler, environ=environ)
    assert code != 0
    assert report["status"] == "failed"
    assert report["error"]["code"] == "login_required"


@pytest.mark.parametrize("argv", [
    ["--add-to-cart", "--product-id", "7", "--unique-id", "sku-red"],
    ["--add-to-cart", "--confirm", "--product-id", "7"],
    ["--add-to-cart", "--confirm", "--unique-id", "sku-red"],
    ["--confirm", "--product-id", "7"],
    WRITE_ARGS + ["--quantity", "0"], WRITE_ARGS + ["--quantity", "10000"],
    WRITE_ARGS + ["--quantity", "1.5"], WRITE_ARGS + ["--quantity", "True"],
    ["--product-id", "0"], ["--product-id", "7", "--unexpected", AUTHORIZATION],
    ["--product-id", "7", "--unique-id", " ", "--add-to-cart", "--confirm"],
])
def test_invalid_write_flags_and_quantities_are_refused_before_network(tmp_path, argv):
    def handler(request):
        pytest.fail("invalid CLI input must never contact the shop")

    code, report, _ = run_cli(tmp_path, handler, argv)
    assert code != 0
    assert report["status"] == "failed"
    assert report["error"]["code"] == "invalid_request"
    assert report["write"]["status"] == "not_attempted"


def test_cli_reads_credentials_from_explicit_service_env_and_ignores_local_db_fields(tmp_path):
    env_file = tmp_path / "service.env"
    env_file.write_text(f"YSHOP_AUTHORIZATION={AUTHORIZATION}\nYSHOP_DB_PASSWORD=private-db-password\n",
                        encoding="utf-8")

    def handler(request):
        assert request.headers["Authorization"] == AUTHORIZATION
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        return ok({"valid": [], "invalid": []})

    code, report, output = run_cli(tmp_path, handler, environ={}, env_file=env_file)
    assert code == 0
    assert "private-db-password" not in json.dumps(report) + output
    settings = Settings(_env_file=env_file)
    assert set(settings.model_dump()) == {"yshop_api_url", "http_timeout_seconds", "cors_origins"}


def test_empty_environment_authorization_overrides_local_env_without_network(tmp_path):
    env_file = tmp_path / "service.env"
    env_file.write_text(f"YSHOP_AUTHORIZATION={AUTHORIZATION}\n", encoding="utf-8")

    def handler(request):
        pytest.fail("an explicit empty environment credential must not use a local fallback")

    code, report, _ = run_cli(tmp_path, handler, environ={"YSHOP_AUTHORIZATION": ""}, env_file=env_file)
    assert code != 0
    assert report["error"]["code"] == "login_required"


@pytest.mark.parametrize("problem", ["wrong_sku", "insufficient_stock", "existing_quantity", "duplicate_baseline"])
def test_write_preconditions_check_live_sku_and_existing_cart_stock(tmp_path, problem):
    requests = []
    argv = WRITE_ARGS.copy()
    if problem == "wrong_sku":
        argv[3] = "101"
    if problem == "insufficient_stock":
        argv += ["--quantity", "2"]

    def handler(request):
        requests.append((request.method, request.url.path))
        assert request.method == "GET"
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product(stock=1 if problem == "insufficient_stock" else 8))
        rows = [row(quantity=8)] if problem == "existing_quantity" else []
        if problem == "duplicate_baseline":
            rows = [row(cart_id=123), row(cart_id=124)]
        return ok({"valid": rows, "invalid": []})

    code, report, _ = run_cli(tmp_path, handler, argv)
    assert code != 0
    assert report["status"] == "failed"
    assert report["write"]["status"] == "not_attempted"
    assert all(method == "GET" for method, _ in requests)


def test_confirmed_write_posts_once_and_verifies_actual_cart_id_and_quantity_increase(tmp_path):
    calls = []
    cart_reads = 0

    def handler(request):
        nonlocal cart_reads
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            assert request.url.path == "/api/cart/add"
            assert json.loads(request.content) == {"productId": 7, "uniqueId": "sku-red", "cartNum": 2,
                                                   "new": 0, "combinationId": 0, "secKillId": 0}
            return ok({"cartId": 123})
        cart_reads += 1
        return ok({"valid": [row(quantity=2 if cart_reads == 1 else 4)], "invalid": []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS + ["--quantity", "2"])
    assert code == 0
    assert [method for method, _ in calls].count("POST") == 1
    assert cart_reads == 2
    assert report["status"] == "succeeded"
    assert report["write"] == {"status": "succeeded", "cart_id": 123}
    assert report["verification"]["status"] == "passed"
    assert report["verification"]["before_quantity"] == 2
    assert report["verification"]["observed_quantity"] == 4


@pytest.mark.parametrize("mismatch", ["cart_id", "product_id", "unique_id", "quantity", "invalid"])
def test_cart_id_return_alone_does_not_claim_success_when_followup_cart_mismatches(tmp_path, mismatch):
    cart_reads = 0
    posts = 0

    def handler(request):
        nonlocal cart_reads, posts
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            posts += 1
            return ok({"cartId": 123})
        cart_reads += 1
        if cart_reads == 1:
            return ok({"valid": [], "invalid": []})
        kwargs = {mismatch: {"cart_id": 999, "product_id": 99, "unique_id": "sku-other", "quantity": 2}[mismatch]} if mismatch != "invalid" else {}
        after = row(**kwargs) if mismatch == "quantity" else row(quantity=1, **kwargs)
        return ok({"valid": [] if mismatch == "invalid" else [after],
                   "invalid": [after] if mismatch == "invalid" else []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS)
    assert code != 0
    assert posts == 1
    assert report["write"]["status"] == "succeeded"
    assert report["status"] == "failed"
    assert report["verification"]["status"] == "failed"


@pytest.mark.parametrize("failure", ["timeout", "missing_cart_id", "system_error"])
@pytest.mark.parametrize("changed", [False, True])
def test_unknown_write_queries_cart_once_and_keeps_unknown_even_if_change_is_observed(tmp_path, failure, changed):
    cart_reads = 0
    posts = 0

    def handler(request):
        nonlocal cart_reads, posts
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            posts += 1
            if failure == "timeout":
                raise httpx.ReadTimeout(AUTHORIZATION, request=request)
            if failure == "system_error":
                return httpx.Response(200, json={"status": 5100, "success": False,
                                                "msg": AUTHORIZATION, "data": None})
            return ok({"private": AUTHORIZATION})
        cart_reads += 1
        return ok({"valid": [row(quantity=1)] if changed and cart_reads == 2 else [], "invalid": []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS)
    assert code != 0
    assert posts == 1
    assert cart_reads == 2
    assert report["status"] == "unknown"
    assert report["write"]["status"] == "unknown"
    assert report["verification"]["status"] == "unknown"
    assert report["verification"]["observation"] == ("quantity_increase_observed" if changed else "unchanged")
    assert report["error"]["code"] == "cart_result_unknown"
    assert report["error"]["outcome"] == "unknown"


@pytest.mark.parametrize("network_failures", [1, 2])
def test_system_error_followup_network_read_retries_at_most_once(tmp_path, network_failures):
    requests = []
    cart_reads = 0

    def handler(request):
        nonlocal cart_reads
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            return httpx.Response(200, json={"status": 5100, "success": False,
                                            "msg": AUTHORIZATION, "data": None})
        cart_reads += 1
        if cart_reads == 1:
            return ok({"valid": [], "invalid": []})
        if cart_reads - 1 <= network_failures:
            raise httpx.ConnectError(AUTHORIZATION, request=request)
        return ok({"valid": [row(quantity=1)], "invalid": []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS)
    assert code != 0
    assert requests == [
        ("GET", "/api/userinfo"), ("GET", "/api/product/detail/7"),
        ("GET", "/api/cart/list"), ("POST", "/api/cart/add"),
        ("GET", "/api/cart/list"), ("GET", "/api/cart/list"),
    ]
    assert report["status"] == "unknown"
    assert report["write"]["status"] == "unknown"
    assert report["verification"]["status"] == "unknown"
    assert report["error"]["code"] == "cart_result_unknown"
    assert report["error"]["outcome"] == "unknown"
    if network_failures == 1:
        assert report["verification"]["observation"] == "quantity_increase_observed"
        assert report["verification"]["observed_quantity"] == 1
    else:
        assert report["verification"]["observation"] == "read_failed"
        assert report["verification"]["error"]["code"] == "shop_unavailable"


def test_business_rejection_is_failed_without_repeating_post(tmp_path):
    posts = 0
    cart_reads = 0

    def handler(request):
        nonlocal posts, cart_reads
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            posts += 1
            return httpx.Response(200, json={"status": 5101, "success": False,
                                            "msg": AUTHORIZATION, "data": None})
        cart_reads += 1
        return ok({"valid": [], "invalid": []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS)
    assert code != 0
    assert posts == 1
    assert cart_reads == 1
    assert report["status"] == "failed"
    assert report["write"]["status"] == "failed"


def test_followup_read_failure_is_reported_without_repeat_or_false_success(tmp_path):
    posts = 0
    cart_reads = 0

    def handler(request):
        nonlocal posts, cart_reads
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            posts += 1
            return ok({"cartId": 123})
        cart_reads += 1
        if cart_reads > 1:
            raise httpx.ConnectError(AUTHORIZATION, request=request)
        return ok({"valid": [], "invalid": []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS)
    assert code != 0
    assert posts == 1
    assert cart_reads == 3  # one before write, then the read adapter's two attempts
    assert report["status"] == "unknown"
    assert report["write"]["status"] == "succeeded"
    assert report["verification"]["status"] == "unknown"


def test_allowed_upstream_fields_cannot_echo_credentials_into_report(tmp_path):
    def handler(request):
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33, "nickname": AUTHORIZATION})
        if request.url.path == "/api/product/detail/7":
            data = product()
            data["storeInfo"]["storeInfo"] = f"echo {AUTHORIZATION} and smoke-private-token"
            return ok(data)
        return ok({"valid": [], "invalid": []})

    code, _, _ = run_cli(tmp_path, handler)
    assert code == 0


def test_quantity_increase_in_an_invalid_row_is_not_reported_as_a_valid_observation(tmp_path):
    reads = 0

    def handler(request):
        nonlocal reads
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            raise httpx.ReadTimeout(AUTHORIZATION, request=request)
        reads += 1
        return ok({"valid": [row(quantity=1)] if reads == 2 else [],
                   "invalid": [row(quantity=1, cart_id=124)] if reads == 2 else []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS)
    assert code != 0
    assert report["status"] == "unknown"
    assert report["verification"]["observation"] == "unexpected_cart_state"


@pytest.mark.parametrize("duplicate", [["--product-id", "7"], ["--unique-id=sku-red"], ["--confirm"]])
def test_cli_refuses_duplicate_identity_flags_before_network(tmp_path, duplicate):
    def handler(request):
        pytest.fail("duplicate write identities must be refused")

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS + duplicate)
    assert code != 0
    assert report["error"]["code"] == "invalid_request"


def test_cli_prints_report_write_failure_explicitly_and_makes_no_network_requests(tmp_path):
    smoke = importlib.import_module("scripts.smoke_test")
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    def handler(request):
        pytest.fail("an unwritable report must prevent the write and all network traffic")

    output = io.StringIO()
    code = smoke.main(WRITE_ARGS, environ={"YSHOP_AUTHORIZATION": AUTHORIZATION},
                      env_file=tmp_path / "absent.env", settings=Settings(_env_file=None),
                      transport=httpx.MockTransport(handler), report_path=blocked / "report.json", stdout=output)
    result = json.loads(output.getvalue())
    assert code != 0
    assert result["status"] == "error"
    assert result["error"]["code"] == "report_write_failed"
    assert result["report_status"] == "failed"


def test_unexpected_followup_error_preserves_successful_write_and_unknown_verification(tmp_path):
    reads = 0
    posts = 0

    def handler(request):
        nonlocal reads, posts
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            posts += 1
            return ok({"cartId": 123})
        reads += 1
        if reads == 2:
            raise ValueError(AUTHORIZATION)
        return ok({"valid": [], "invalid": []})

    code, report, _ = run_cli(tmp_path, handler, WRITE_ARGS)
    assert code != 0
    assert posts == 1
    assert report["status"] == "unknown"
    assert report["write"] == {"status": "succeeded", "cart_id": 123}
    assert report["verification"]["status"] == "unknown"


def test_final_report_error_preserves_successful_write_and_verified_cart_in_output(tmp_path):
    smoke = importlib.import_module("scripts.smoke_test")
    report_path = tmp_path / "report.json"
    reads = 0
    posts = 0

    def handler(request):
        nonlocal reads, posts
        if request.url.path == "/api/userinfo":
            return ok({"uid": 33})
        if request.url.path == "/api/product/detail/7":
            return ok(product())
        if request.method == "POST":
            posts += 1
            return ok({"cartId": 123})
        reads += 1
        if reads == 2:
            report_path.unlink()
            report_path.mkdir()
        return ok({"valid": [row(quantity=1)] if reads == 2 else [], "invalid": []})

    output = io.StringIO()
    code = smoke.main(WRITE_ARGS, environ={"YSHOP_AUTHORIZATION": AUTHORIZATION},
                      env_file=tmp_path / "absent.env", settings=Settings(_env_file=None),
                      transport=httpx.MockTransport(handler), report_path=report_path, stdout=output)
    result = json.loads(output.getvalue())
    assert code != 0
    assert posts == 1
    assert result["status"] == "error"
    assert result["write"] == {"status": "succeeded", "cart_id": 123}
    assert result["verification"]["status"] == "passed"
    assert result["error"]["code"] == "report_write_failed"
    assert result["report_status"] == "failed"
