"""Query actual purchasable SKU prices using the locally configured shop identity."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

import httpx
from pydantic import ValidationError

from app.config import SERVICE_ENV_FILE, Settings
from app.errors import ShopError
from app.schemas import ProductSearchInput
from app.tools.yshop import YShopClient
from scripts.smoke_test import _SafeParser, _authorization, _save_report


REPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "day2-search-smoke-report.json"


async def run(query: ProductSearchInput, authorization: str) -> dict:
    settings = Settings()
    async with httpx.AsyncClient(base_url=settings.yshop_api_url,
                                timeout=settings.http_timeout_seconds,
                                trust_env=False, follow_redirects=False) as http:
        shop = YShopClient(http)
        user = await shop.get_user(authorization)
        result = await shop.search_products(query, authorization)
    return {"status": "passed", "user_id": user.user_id,
            "query": query.model_dump(mode="json"), "result": result.model_dump(mode="json"),
            "shop_cart_post_attempts": 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeParser(description="按实时可购规格、预算和关键词查询商品。", allow_abbrev=False)
    parser.add_argument("--keyword", default="")
    for name in ("category-id", "min-price", "max-price"):
        parser.add_argument("--" + name)
    parser.add_argument("--page", default="1")
    parser.add_argument("--limit", default="10")
    authorization = ""
    try:
        query = ProductSearchInput(**vars(parser.parse_args(argv)))
        authorization = _authorization(os.environ, SERVICE_ENV_FILE)
        report = asyncio.run(run(query, authorization))
        _save_report(REPORT_PATH, report, authorization)
        print(f"Search passed: {len(report['result']['products'])} products, "
              f"total {report['result']['total']}. Report: data/{REPORT_PATH.name}")
        return 0
    except (ValidationError, ShopError) as error:
        code = error.code if isinstance(error, ShopError) else "invalid_request"
        print(f"Search failed: {code}.", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("Search report could not be saved or configuration is invalid.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
