"""Read live shop data; add an ordinary SKU only with explicit confirmation."""

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx
from dotenv import dotenv_values
from pydantic import ValidationError

from app.config import SERVICE_ENV_FILE, Settings
from app.errors import ShopError
from app.schemas import CartAddInput, CartItem, CartSnapshot
from app.tools.yshop import YShopClient


REPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "day1-smoke-report.json"
_BEARER = re.compile(r"Bearer [A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse normally echoes invalid arguments, which may contain secrets.
        raise ShopError("invalid_request")


def _integer(value: str) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError("invalid integer")
    return int(value)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _SafeParser(description="查询实时商品和购物车；加购必须显式确认。", allow_abbrev=False)
    parser.add_argument("--product-id", type=_integer, required=True, help="实际普通商品 ID")
    parser.add_argument("--unique-id", help="商品详情返回的实际 SKU unique ID")
    parser.add_argument("--quantity", type=_integer, default=1, help="加购数量，1–9999；默认 1")
    parser.add_argument("--add-to-cart", action="store_true", help="请求普通商品加购")
    parser.add_argument("--confirm", action="store_true", help="确认本次购物车写入")
    arguments = list(sys.argv[1:] if argv is None else argv)
    flags = {"--product-id", "--unique-id", "--quantity", "--add-to-cart", "--confirm"}
    seen = set()
    for argument in arguments:
        flag = argument.partition("=")[0]
        if flag in flags:
            if flag in seen:
                raise ShopError("invalid_request")
            seen.add(flag)
    options = parser.parse_args(arguments)
    if options.product_id <= 0 or not 1 <= options.quantity <= 9999:
        raise ShopError("invalid_request")
    if options.confirm != options.add_to_cart:
        raise ShopError("invalid_request")
    if options.add_to_cart:
        try:
            options.item = CartAddInput(product_id=options.product_id,
                                        unique_id=options.unique_id, quantity=options.quantity)
        except ValidationError:
            raise ShopError("invalid_request") from None
    else:
        options.item = None
    return options


def _authorization(environ: Mapping[str, str], env_file: Path) -> str:
    value = (environ["YSHOP_AUTHORIZATION"] if "YSHOP_AUTHORIZATION" in environ else
             dotenv_values(env_file, interpolate=False).get("YSHOP_AUTHORIZATION"))
    if not isinstance(value, str) or _BEARER.fullmatch(value) is None:
        raise ShopError("login_required")
    return value


def _error(error: ShopError) -> dict[str, Any]:
    result: dict[str, Any] = {"code": error.code, "message": error.message}
    if error.outcome is not None:
        result["outcome"] = error.outcome
    return result


def _redacted(value: Any, authorization: str) -> Any:
    if isinstance(value, str):
        for secret in (authorization, authorization.partition(" ")[2]):
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redacted(item, authorization) for item in value]
    if isinstance(value, dict):
        return {key: _redacted(item, authorization) for key, item in value.items()}
    return value


def _save_report(path: Path, report: dict[str, Any], authorization: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(_redacted(report, authorization), ensure_ascii=False,
                                        indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _matching(cart: CartSnapshot, item: CartAddInput, *, invalid: bool = False) -> list[CartItem]:
    rows = cart.invalid if invalid else cart.valid
    return [row for row in rows if row.product_id == item.product_id and row.unique_id == item.unique_id]


def _verify_cart(
    after: CartSnapshot, item: CartAddInput, before: list[CartItem], cart_id: int | None,
) -> dict[str, Any]:
    before_quantity = before[0].quantity if before else 0
    expected = before_quantity + item.quantity
    matching = _matching(after, item)
    observed = sum(row.quantity for row in matching)
    result: dict[str, Any] = {
        "before_quantity": before_quantity, "expected_quantity": expected,
        "observed_quantity": observed, "observed_cart_ids": [row.cart_id for row in matching],
    }
    if cart_id is None:
        if len(matching) == 1 and observed == expected and not _matching(after, item, invalid=True):
            observation = "quantity_increase_observed"
        elif observed == before_quantity and not _matching(after, item, invalid=True):
            observation = "unchanged"
        else:
            observation = "unexpected_cart_state"
        result.update(status="unknown", observation=observation)
        return result
    verified = (
        len(matching) == 1 and matching[0].cart_id == cart_id and observed == expected
        and not _matching(after, item, invalid=True)
        and (not before or before[0].cart_id == cart_id)
    )
    result.update(status="passed" if verified else "failed",
                  observation="expected_cart_observed" if verified else "cart_mismatch")
    return result


async def _run_checks(
    shop: YShopClient, options: argparse.Namespace, authorization: str,
    report: dict[str, Any], report_path: Path,
) -> None:
    checks = report["checks"]
    checks["identity"] = "running"
    user = await shop.get_user(authorization)
    report["identity"] = {"user_id": user.user_id}
    checks["identity"] = "passed"
    checks["product"] = "running"
    product = await shop.get_product_detail(options.product_id, authorization)
    report["product"] = product.model_dump(mode="json")
    checks["product"] = "passed"
    checks["cart_before"] = "running"
    before = await shop.get_cart(authorization)
    report["cart_before"] = before.model_dump(mode="json")
    checks["cart_before"] = "passed"
    item: CartAddInput | None = options.item
    if item is None:
        report["status"] = "succeeded"
        return

    checks["sku"] = "running"
    sku = next((sku for sku in product.skus if sku.unique_id == item.unique_id), None)
    if sku is None:
        raise ShopError("sku_unavailable")
    checks["sku"] = "passed"
    checks["stock"] = "running"
    matching = _matching(before, item)
    if len(matching) > 1:
        raise ShopError("cart_state_ambiguous")
    if _matching(before, item, invalid=True):
        raise ShopError("cart_item_invalid")
    before_quantity = matching[0].quantity if matching else 0
    stocks = [sku.stock] + ([matching[0].stock] if matching and matching[0].stock is not None else [])
    if before_quantity + item.quantity > min(stocks):
        raise ShopError("insufficient_stock")
    checks["stock"] = "passed"
    # Persist the execution state before the sole POST; a crash cannot imply no attempt.
    report["status"] = "pending"
    report["write"] = {"status": "executing"}
    checks["write"] = "running"
    try:
        _save_report(report_path, report, authorization)
    except OSError:
        report["write"] = {"status": "not_attempted"}
        raise
    cart_id = None
    try:
        added = await shop.add_to_cart(item, authorization)
    except ShopError as error:
        report["error"] = _error(error)
        report["write"] = {"status": error.outcome or "failed"}
        checks["write"] = error.outcome or "failed"
        if error.outcome != "unknown":
            report["status"] = "failed"
            report["verification"] = {"status": "not_attempted"}
            return
    else:
        report["write"] = added.model_dump()
        checks["write"] = "passed"
        cart_id = added.cart_id

    # One follow-up read operation; only the existing GET adapter may retry once.
    checks["cart_after"] = "running"
    try:
        after = await shop.get_cart(authorization)
    except ShopError as error:
        checks["cart_after"] = "failed"
        report["status"] = "unknown"
        report["verification"] = {"status": "unknown", "observation": "read_failed", "error": _error(error)}
        if "error" not in report:
            report["error"] = _error(ShopError("cart_result_unknown", outcome="unknown"))
        return
    report["cart_after"] = after.model_dump(mode="json")
    checks["cart_after"] = "passed"
    report["verification"] = _verify_cart(after, item, matching, cart_id)
    verification = report["verification"]["status"]
    report["status"] = "succeeded" if verification == "passed" else verification
    if verification == "failed":
        report["error"] = _error(ShopError("cart_verification_failed"))


def main(
    argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
    env_file: Path = SERVICE_ENV_FILE, settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None, report_path: Path = REPORT_PATH,
    stdout: TextIO | None = None,
) -> int:
    """Run real CLI checks, with only external transport/configuration injectable."""
    stdout = sys.stdout if stdout is None else stdout
    report: dict[str, Any] = {
        "schema_version": 1, "mode": "pending", "status": "pending",
        "checks": {key: "pending" for key in
                   ("arguments", "authorization", "identity", "product", "cart_before")},
        "write": {"status": "not_attempted"}, "verification": {"status": "not_attempted"},
    }
    authorization = ""
    try:
        report["checks"]["arguments"] = "running"
        options = _arguments(argv)
        report["checks"]["arguments"] = "passed"
        report["mode"] = "add_to_cart" if options.add_to_cart else "read"
        if not options.add_to_cart:
            report["write"] = {"status": "not_requested"}
            report["verification"] = {"status": "not_requested"}
        report["checks"]["authorization"] = "running"
        authorization = _authorization(os.environ if environ is None else environ, Path(env_file))
        report["checks"]["authorization"] = "passed"
        current_settings = settings if settings is not None else Settings(_env_file=env_file)
        _save_report(Path(report_path), report, authorization)

        async def execute() -> None:
            async with httpx.AsyncClient(
                base_url=current_settings.yshop_api_url, timeout=current_settings.http_timeout_seconds,
                follow_redirects=False, trust_env=False, transport=transport,
            ) as http:
                await _run_checks(YShopClient(http), options, authorization, report, Path(report_path))

        asyncio.run(execute())
    except SystemExit as error:
        return int(error.code or 0)  # argparse --help
    except ShopError as error:
        report["status"] = "failed"
        report["error"] = _error(error)
    except ValidationError:
        report["status"] = "failed"
        report["error"] = _error(ShopError("invalid_request"))
    except OSError:
        _execution_error_status(report)
        report["error"] = {"code": "report_write_failed", "message": "无法保存联调报告，请检查本地数据目录权限。"}
    except Exception:
        _execution_error_status(report)
        report["error"] = {"code": "smoke_error", "message": "联调执行出错，请检查本地配置；勿直接重复加购。"}
    for key, status in report["checks"].items():
        if status == "running":
            report["checks"][key] = "failed"
    try:
        _save_report(Path(report_path), report, authorization)
    except OSError:
        report["report_status"] = "failed"
        if report["status"] == "succeeded":
            report["status"] = "error"
        report["error"] = {"code": "report_write_failed", "message": "无法保存联调报告，请检查本地数据目录权限。"}
    summary = {key: report[key] for key in ("status", "write", "verification")}
    if "error" in report:
        summary["error"] = report["error"]
    if "report_status" in report:
        summary["report_status"] = report["report_status"]
    print(json.dumps(_redacted(summary, authorization), ensure_ascii=False), file=stdout)
    return 0 if report["status"] == "succeeded" and "report_status" not in report else 1


def _execution_error_status(report: dict[str, Any]) -> None:
    write_status = report["write"]["status"]
    uncertain = write_status in {"executing", "succeeded", "unknown"} and report["verification"]["status"] != "passed"
    report["status"] = "unknown" if uncertain else "error"
    if uncertain:
        report["verification"] = {"status": "unknown", "observation": "verification_incomplete"}
    if write_status == "executing":
        report["write"] = {"status": "unknown"}


if __name__ == "__main__":
    raise SystemExit(main())
