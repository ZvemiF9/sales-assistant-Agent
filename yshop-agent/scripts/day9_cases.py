"""Frozen Day 9 schema, independently checked facts, and honest denominators."""

import ast
from collections import Counter
from decimal import Decimal
import hashlib
import math
from pathlib import Path
import re
from statistics import median

from app.agent.model import _loads
from app.rag.config import SERVICE_ROOT


CASES_PATH = SERVICE_ROOT / "evals/day9_cases.json"
SUITE_PATHS = {
    "day9-v1": CASES_PATH,
    "day9-v2": SERVICE_ROOT / "evals/day9_cases_v2.json",
}
GROUPS = ("product", "policy", "multi_turn", "confirmation", "identity_failure")
MODES = ("live", "deterministic")
STATUSES = ("passed", "failed", "skipped", "not_run")
QUERY_KEYS = {"keyword", "category_id", "min_price", "max_price", "page", "limit"}
EXPECT_KEYS = {"status", "tool", "query", "product_ids", "empty", "document_ids",
               "reference_index", "preserve_product_ids"}
NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
SELECTOR = re.compile(r"tests/test_[a-z0-9_]+\.py::test_[a-z0-9_]+\Z")


def digest(path=CASES_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids(values, *, text=False):
    if type(values) is not list or len(values) > 20 or len(set(values)) != len(values):
        raise ValueError("invalid expected IDs")
    if any((type(v) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", v)) if text else
           (type(v) is not int or not 0 < v < 10**19) for v in values):
        raise ValueError("invalid expected ID")


def _expect(value):
    if (type(value) is not dict or not {"status", "tool"} <= value.keys()
            or value.keys() - EXPECT_KEYS or type(value["status"]) is not str or type(value["tool"]) is not str
            or value["status"] not in {"answered", "needs_input", "no_evidence"}
            or value["tool"] not in {"none", "search_products", "get_product_detail", "search_policy"}):
        raise ValueError("invalid expectation")
    for key in ("empty", "preserve_product_ids"):
        if key in value and type(value[key]) is not bool:
            raise ValueError("invalid boolean expectation")
    if "reference_index" in value and (type(value["reference_index"]) is not int
                                      or not 0 <= value["reference_index"] <= 2):
        raise ValueError("invalid reference")
    for key in ("product_ids", "document_ids"):
        if key in value:
            _ids(value[key], text=key == "document_ids")
    if "query" in value:
        query = value["query"]
        if (type(query) is not dict or query.keys() - QUERY_KEYS
                or any(type(v) is not str or not v or len(v) > 100 for v in query.values())):
            raise ValueError("invalid expected query")
        for key in ("min_price", "max_price"):
            if key in query and (not re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,2})?", query[key])):
                raise ValueError("invalid expected price")
        for key in ("page", "limit", "category_id"):
            if key in query and (not re.fullmatch(r"[1-9][0-9]{0,8}", query[key])
                                 or key == "limit" and int(query[key]) != 3):
                raise ValueError("invalid expected count")
        if ("min_price" in query and "max_price" in query
                and Decimal(query["min_price"]) > Decimal(query["max_price"])):
            raise ValueError("invalid expected interval")
    if value["tool"] == "search_products" and "query" not in value:
        raise ValueError("missing expected query")
    if value["tool"] == "get_product_detail" and not (
            value.get("product_ids") or "reference_index" in value):
        raise ValueError("missing detail target")


def load_cases(path=CASES_PATH):
    path = Path(path)
    if path.stat().st_size > 256 * 1024:
        raise ValueError("oversized manifest")
    value = _loads(path.read_bytes())
    if (type(value) is not dict or set(value) != {"schema_version", "frozen_before_live_requests", "scope", "cases"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["frozen_before_live_requests"] is not True
            or value["scope"] != "development_suite_live_read_only_and_deterministic_boundaries"
            or type(value["cases"]) is not list or len(value["cases"]) != 50):
        raise ValueError("invalid frozen manifest")
    cases = value["cases"]
    ids, selectors = set(), set()
    for case in cases:
        if (type(case) is not dict or type(case.get("id")) is not str or not NAME.fullmatch(case["id"])
                or type(case.get("group")) is not str or type(case.get("mode")) is not str
                or case.get("group") not in GROUPS or case.get("mode") not in MODES or case["id"] in ids):
            raise ValueError("invalid case")
        ids.add(case["id"])
        if case["mode"] == "deterministic":
            if (set(case) != {"id", "group", "mode", "selector"} or case["group"] not in GROUPS[3:]
                    or type(case["selector"]) is not str or not SELECTOR.fullmatch(case["selector"])
                    or case["selector"] in selectors):
                raise ValueError("invalid deterministic selector")
            selectors.add(case["selector"])
            file, name = case["selector"].split("::")
            source = SERVICE_ROOT / file
            if not source.is_file():
                raise ValueError("missing selected file")
            tree = ast.parse(source.read_text(encoding="utf-8"))
            if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body):
                raise ValueError("missing selected function")
        else:
            if (set(case) != {"id", "group", "mode", "steps"} or case["group"] not in GROUPS[:3]
                    or type(case["steps"]) is not list
                    or not (2 <= len(case["steps"]) <= 4 if case["group"] == "multi_turn" else len(case["steps"]) == 1)):
                raise ValueError("invalid live steps")
            for step in case["steps"]:
                if (type(step) is not dict or set(step) != {"message", "sse", "expect"}
                        or type(step["message"]) is not str or not step["message"].strip()
                        or len(step["message"]) > 2000 or type(step["sse"]) is not bool):
                    raise ValueError("invalid live message")
                _expect(step["expect"])
    if Counter(c["group"] for c in cases) != Counter({g: 10 for g in GROUPS}):
        raise ValueError("invalid group counts")
    if Counter(c["mode"] for c in cases) != Counter(live=30, deterministic=20):
        raise ValueError("invalid mode counts")
    return cases


def _same_query(actual, expected):
    if set(actual) != set(expected):
        return False
    for key, value in expected.items():
        if key in {"min_price", "max_price"}:
            try:
                if Decimal(actual[key]) != Decimal(value):
                    return False
            except (ValueError, ArithmeticError):
                return False
        elif str(actual[key]) != value:
            return False
    return True


def tool_checks(expect, observations, previous):
    tool = expect["tool"]
    tools = [e for e in observations if e.get("category") == "shop_read"
             and e.get("forwarded") is True and e.get("path") != "userinfo"]
    if tool == "none":
        return ["unexpected_product_tool"] if tools else []
    if tool == "search_policy":
        return ["unexpected_product_tool"] if tools else []
    if tool == "search_products":
        if any(e.get("path") != "agent/products" for e in tools):
            return ["unexpected_product_tool"]
        matches = [e for e in tools if e.get("path") == "agent/products"]
        if not matches:
            return ["missing_tool_observation"]
        return [] if all(_same_query(e.get("query", {}), expect["query"]) for e in matches) else ["tool_arguments_mismatch"]
    targets = expect.get("product_ids", [])
    if "reference_index" in expect:
        index = expect["reference_index"]
        if len(previous) <= index:
            return ["missing_reference_candidate"]
        targets = [previous[index]["product_id"]]
    matches = [e for e in tools if e.get("path") == "product/detail"]
    if any(e.get("path") != "product/detail" for e in tools):
        return ["unexpected_product_tool"]
    if not matches:
        return ["missing_tool_observation"]
    return [] if all(e.get("product_id") in targets for e in matches) else ["tool_arguments_mismatch"]


def product_checks(products, truth, expect):
    failures = []
    ids = [p["product_id"] for p in products]
    if "product_ids" in expect and ids != expect["product_ids"]:
        failures.append("product_ids_mismatch")
    if "empty" in expect and bool(products) == expect["empty"]:
        failures.append("product_count_mismatch")
    query = expect.get("query", {})
    for product in products:
        original = truth.get(product["product_id"])
        if original is None or any(product[k] != original[k] for k in ("name", "image", "description")):
            failures.append("product_truth_mismatch")
            continue
        skus = {s["unique_id"]: s for s in original["skus"]}
        if expect.get("tool") == "get_product_detail" and {
                s["unique_id"] for s in product["skus"]} != set(skus):
            failures.append("product_truth_mismatch")
        for sku in product["skus"]:
            actual = skus.get(sku["unique_id"])
            if (actual is None or any(sku[k] != actual[k] for k in ("name", "image", "stock"))
                    or Decimal(str(sku["price"])) != Decimal(str(actual["price"]))):
                failures.append("product_truth_mismatch")
            if "query" in expect and (sku["stock"] <= 0
                    or "min_price" in query and Decimal(str(sku["price"])) < Decimal(query["min_price"])
                    or "max_price" in query and Decimal(str(sku["price"])) > Decimal(query["max_price"])):
                failures.append("product_bounds_mismatch")
    return list(dict.fromkeys(failures))


def policy_checks(result, originals, expect):
    sources = result.get("sources", [])
    ids = {s["document_id"] for s in sources}
    failures, checks = [], []
    if not set(expect.get("document_ids", [])) <= ids:
        failures.append("expected_documents_missing")
    for source in sources:
        original = originals.get(source["chunk_id"])
        checks.append(original is not None and source["text"] == original.text
                      and all(source.get(k) == v for k, v in original.metadata.items())
                      and source["text"] in result["answer"])
    fidelity = all(checks) if checks else None
    if checks and not fidelity:
        failures.append("citation_fidelity_failed")
    if expect.get("status") == "no_evidence" and sources:
        failures.append("unexpected_citations")
    return {"count": len(sources), "fidelity": fidelity, "failures": failures,
            "checks": [{"document_id": s["document_id"], "chunk_id": s["chunk_id"], "exact": ok}
                       for s, ok in zip(sources, checks)]}


def _totals(rows):
    counts = Counter(row["status"] for row in rows)
    return {"total": len(rows), **{s: counts[s] for s in STATUSES},
            "case_success_rate": counts["passed"] / len(rows) if rows else None}


def _timing_stats(values):
    measured = sorted(value for value in values if type(value) in {int, float}
                      and math.isfinite(value) and value >= 0)
    return {"count": len(measured), "median": median(measured) if measured else None,
            "p95": measured[math.ceil(.95 * len(measured)) - 1] if measured else None,
            "max": measured[-1] if measured else None}


def summarize(rows):
    if any(r.get("status") not in STATUSES or r.get("group") not in GROUPS or r.get("mode") not in MODES for r in rows):
        raise ValueError("invalid case outcome")
    result = _totals(rows)
    result["groups"] = {g: _totals([r for r in rows if r["group"] == g]) for g in GROUPS}
    result["modes"] = {m: _totals([r for r in rows if r["mode"] == m]) for m in MODES}
    steps = [step for row in rows for step in row.get("steps", [])]
    timing_keys = ("client_elapsed_ms", "status_first_ms", "business_text_first_ms", "terminal_ms",
                   "model_elapsed_ms", "tool_elapsed_ms", "independent_verification_elapsed_ms")
    result["timings"] = {key: _timing_stats(s.get(key) for s in steps) for key in timing_keys}
    result["timings_by_mode"] = {
        "live": {key: dict(result["timings"][key]) for key in timing_keys},
        "deterministic": {"case_elapsed_ms": _timing_stats(
            row.get("elapsed_ms") for row in rows if row["mode"] == "deterministic")},
    }
    result["first_use"] = {label: None for label in ("model", "product", "rag")}
    for row in rows:
        for index, step in enumerate(row.get("steps", [])):
            for label in step.get("first_use_labels", []):
                if label in result["first_use"] and result["first_use"][label] is None:
                    result["first_use"][label] = {"case_id": row["id"], "step_index": index}
    usage = [s["usage"] for s in steps if isinstance(s.get("usage"), dict)
             and all(type(s["usage"].get(k)) is int for k in ("input_tokens", "output_tokens"))]
    result["usage"] = {"measured_steps": len(usage), "unmeasured_steps": len(steps) - len(usage),
                       **{k: sum(u[k] for u in usage) if usage else None for k in ("input_tokens", "output_tokens")}}
    citations = [s["citations"] for s in steps if isinstance(s.get("citations"), dict)]
    measured = [c for c in citations if c.get("fidelity") is not None]
    result["citations"] = {"count": sum(c["count"] for c in citations), "measured_steps": len(measured),
                           "fidelity_rate": sum(c["fidelity"] is True for c in measured) / len(measured) if measured else None}
    checked = [s for s in steps if s.get("tool_parameters_match") is not None]
    result["tool_parameters"] = {"checked_steps": len(checked),
                                "matched_steps": sum(s["tool_parameters_match"] is True for s in checked)}
    return result
