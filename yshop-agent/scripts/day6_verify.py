"""Opt-in frozen cart-action acceptance using an owned, restarted local service."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from time import perf_counter, time
from typing import Mapping, Sequence, TextIO

import httpx
from pydantic import ValidationError

from app.actions.schemas import ActionId, CartActionCard
from app.agent.model import _loads
from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from app.schemas import CartAddInput, CartItem, CartSnapshot, UserIdentity
from pydantic import TypeAdapter
from scripts.action_smoke_test import (
    _http_error_code, _public_action_response, _public_session_response, _redact_secrets, _secrets,
)
from scripts.chat_smoke_test import _contains_secret, _crop
from scripts.eval_policy import _arguments
from scripts.session_smoke_test import _save_owned_report
from scripts.smoke_test import _SafeParser, _authorization, _integer, _matching, _verify_cart


SERVICE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVICE_ROOT.parent
CASES_PATH = SERVICE_ROOT / "evals" / "action_cases.json"
QUERY = "预算3元，查询华为商品，请使用商品查询工具核实实时数据。"
PREPARE = "选择第一个商品的A4,白色规格，加入一件购物车。"
TARGET = {"product_id": 36, "unique_id": "c38b4340485b465d92b787dd68dcc26e",
          "sku_name": "A4,白色", "quantity": 1, "unit_price": "1.00"}
STEPS = [
    "query", "cart_before_cancel", "prepare_cancel", "cancel", "cart_after_cancel",
    "prepare_saved", "stop_agent", "start_agent_same_database", "recover_saved_action",
    "cart_before_confirm", "confirm_once", "cart_after_confirm", "confirm_duplicate", "cart_after_duplicate",
]
EXPECTED = {"cancellation_cart_writes": 0, "confirmation_cart_writes": 1,
            "duplicate_cart_writes": 0, "same_sku_quantity_growth": 1}
MODEL_OVERRIDES = {"LLM_ENABLE_THINKING": "false", "LLM_TIMEOUT_SECONDS": "60",
                   "LLM_PROXY_URL": "http://127.0.0.1:7892", "HF_HUB_OFFLINE": "1"}
PROTOCOL_REASONS = {
    "invalid completion envelope", "expected one completion choice", "invalid assistant message",
    "invalid tool calls", "invalid function call", "invalid call ID", "invalid function",
    "invalid function name", "invalid function arguments", "function arguments must be an object",
    "invalid assistant content", "credential echoed in assistant fields", "invalid token usage",
    "inconsistent token usage", "duplicate JSON key", "non-finite JSON number", "invalid_schema_or_json",
}
_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "other"}
_SHOP_PATHS = {"/api/userinfo", "/api/agent/products", "/api/cart/list", "/api/cart/add", "other_shop_path"}


class VerificationStopped(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _load_cases(path):
    raw = Path(path).read_bytes()
    value = _loads(raw)
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("frozen_before_live_requests") is not True
            or value.get("scope") != "confirmed_cart_action_restart_workflow"
            or value.get("live_acceptance_status") != "pending"):
        raise ValueError("invalid frozen manifest")
    scenario = value.get("scenario")
    if (not isinstance(scenario, dict) or scenario.get("id") != "cancel_restart_confirm_duplicate"
            or scenario.get("query") != QUERY or scenario.get("prepare") != PREPARE
            or scenario.get("target") != TARGET or scenario.get("steps") != STEPS
            or scenario.get("expected") != EXPECTED
            or not isinstance(scenario.get("preflight_declaration"), dict)
            or type(scenario["preflight_declaration"].get("stock")) is not int
            or scenario["preflight_declaration"]["stock"] != 8882
            or scenario["preflight_declaration"].get("unit_price") != "1.00"):
        raise ValueError("frozen scenario changed")
    return scenario, hashlib.sha256(raw).hexdigest()


def _confined_path(value, root):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if (any(item.is_symlink() or item.is_junction() for item in (path, *path.parents))
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("path is outside its owned workspace")
    return path.resolve()


def _evidence_path(value, *, project=None):
    root = (PROJECT_ROOT if project is None else project) / "logs" / "agent-day6"
    path = _confined_path(value, root)
    if path.parent != root.resolve() or path.suffix not in {".json", ".jsonl"}:
        raise ValueError("invalid Day 6 evidence path")
    return path


def _database_path(value, *, service=None):
    root = (SERVICE_ROOT if service is None else service) / ".tmp"
    path = _confined_path(value, root)
    relative = path.relative_to(root.resolve())
    if (len(relative.parts) != 2 or path.name != "sessions.sqlite3"
            or re.fullmatch(r"day6-live-[A-Za-z0-9][A-Za-z0-9_-]{0,79}", relative.parts[0]) is None):
        raise ValueError("a new Day 6 service database is required")
    return path


def _validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("invalid observation")
    kind = event.get("kind")
    if kind == "startup":
        if (set(event) != {"kind", "pid", "port"} or type(event["pid"]) is not int or event["pid"] <= 0
                or type(event["port"]) is not int or not 1024 <= event["port"] <= 65535):
            raise ValueError("invalid startup observation")
    elif kind == "model_configuration":
        if (set(event) != {"kind", "configured", "timeout_seconds", "enable_thinking"}
                or type(event["configured"]) is not bool or event["timeout_seconds"] != 60.0
                or event["enable_thinking"] is not False):
            raise ValueError("invalid model configuration observation")
    elif kind in {"shop_request", "model_request", "unexpected_request"}:
        keys = {"kind", "method", "elapsed_ms"}
        keys.add("status" if "status" in event else "transport_error")
        if kind == "shop_request":
            keys.add("path")
            path = event.get("path")
            if path not in _SHOP_PATHS and (not isinstance(path, str) or
                    re.fullmatch(r"/api/product/detail/[1-9][0-9]{0,18}", path) is None):
                raise ValueError("invalid shop path observation")
        if kind == "model_request":
            keys.add("enable_thinking")
            mode = event.get("enable_thinking")
            if mode is not None and type(mode) is not bool and mode not in {"unexpected", "unobserved"}:
                raise ValueError("invalid thinking observation")
        if (set(event) != keys or event.get("method") not in _METHODS
                or type(event.get("elapsed_ms")) is not int or event["elapsed_ms"] < 0):
            raise ValueError("invalid HTTP observation")
        if "status" in event:
            if type(event["status"]) is not int or not 100 <= event["status"] <= 599:
                raise ValueError("invalid HTTP status observation")
        elif event.get("transport_error") not in {"timeout", "network_error", "unexpected_error"}:
            raise ValueError("invalid transport observation")
    elif kind == "model_protocol_check":
        if event.get("status") == "valid":
            if set(event) != {"kind", "status"}:
                raise ValueError("invalid protocol observation")
        elif (set(event) != {"kind", "status", "phase", "reason"} or event.get("status") != "invalid"
              or event.get("phase") not in {"envelope", "reply"} or event.get("reason") not in PROTOCOL_REASONS):
            raise ValueError("invalid protocol failure observation")
    else:
        raise ValueError("unknown observation category")
    return event


class OwnedAgent:
    def __init__(self, *, port, database, event_paths, environ):
        self.port = port
        self.database = _database_path(database)
        self.event_paths = [_evidence_path(path) for path in event_paths]
        self.process = None
        self.environment = dict(environ)
        self.environment.update(MODEL_OVERRIDES)
        self.environment.update(TEMP=str(self.database.parent), TMP=str(self.database.parent),
            PYTHONPYCACHEPREFIX=str(self.database.parent / "pycache"),
            SESSION_DB_PATH=str(self.database))

    def start(self, number):
        if number not in {1, 2} or self.process is not None and self.process.poll() is None:
            raise VerificationStopped("process_already_running")
        events = _evidence_path(self.event_paths[number - 1])
        if events.exists():
            raise VerificationStopped("existing_observations")
        executable, environment = sys.executable, self.environment
        if sys.platform == "win32" and sys.executable != sys._base_executable:
            # Bypass the Windows venv redirector while retaining its interpreter context.
            executable = sys._base_executable
            environment = {**self.environment, "__PYVENV_LAUNCHER__": sys.executable}
        self.process = subprocess.Popen([
            executable, "-X", "utf8", "-W", "error",
            str(SERVICE_ROOT / "scripts" / "day6_observer.py"),
            "--port", str(self.port), "--events", str(events), "--database", str(self.database),
        ], cwd=SERVICE_ROOT, env=environment, shell=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return self.process.pid

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # Only the exact Popen object created by this invocation is eligible.
                self.process.kill()
                self.process.wait(timeout=20)

    async def ready(self, client):
        for _ in range(60):
            if self.process is None or self.process.poll() is not None:
                raise VerificationStopped("owned_service_exited")
            try:
                response = await client.get("/health", timeout=2)
                if (response.status_code == 200 and _loads(response.content) ==
                        {"status": "ok", "service": "yshop-agent"} and
                        any(event == {"kind": "startup", "pid": self.process.pid, "port": self.port}
                            for event in self.events())):
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        raise VerificationStopped("owned_service_not_ready")

    def events(self):
        events = []
        for path in self.event_paths:
            path = _evidence_path(path)
            if path.exists():
                for line in path.read_bytes().splitlines():
                    if len(line) > 4096:
                        raise ValueError("oversized metadata observation")
                    events.append(_validate_event(_loads(line)))
        return events


def _new_report(digest, *, observation_source):
    return {
        "schema_version": 1, "status": "pending", "frozen_cases_sha256": digest,
        "declarations": {"live_write_opt_in": False, "model_environment": dict(MODEL_OVERRIDES),
                         "response_replacement": False},
        "provenance": {"declaration_source": "runner", "observation_source": observation_source,
                       "external_model_verified": False},
        "observation_scope": "Metadata from all httpx requests in this owned acceptance service only.",
        "steps": [], "checks": [], "processes": [], "decision_attempts": [],
        "cart_checks": {}, "downstream_writes": {}, "observations": {},
    }


def _counts(events):
    model = [item for item in events if item["kind"] == "model_request"]
    protocols = [item for item in events if item["kind"] == "model_protocol_check"]
    shop = [item for item in events if item["kind"] == "shop_request"]
    return {
        "model_http_requests": len(model),
        "model_http_successes": sum(item.get("status") == 200 for item in model),
        "model_protocol_valid": sum(item["status"] == "valid" for item in protocols),
        "model_protocol_invalid": sum(item["status"] == "invalid" for item in protocols),
        "model_thinking_false_observed": bool(model) and all(item.get("enable_thinking") is False for item in model),
        "business_write_requests": sum(item["method"] not in {"GET", "HEAD"} for item in shop),
        "cart_add_requests": sum(item["method"] == "POST" and item["path"] == "/api/cart/add" for item in shop),
        "unexpected_requests": sum(item["kind"] == "unexpected_request" for item in events),
    }


def _cart_response(value, secrets):
    if not isinstance(value, dict) or not set(CartSnapshot.model_fields) <= set(value):
        raise ValueError("incomplete public cart")
    cropped = _crop(value, CartSnapshot)
    for name in ("valid", "invalid"):
        if not isinstance(cropped[name], list):
            raise ValueError("invalid public cart")
        rows = []
        for item in cropped[name]:
            if not isinstance(item, dict) or not set(CartItem.model_fields) <= set(item):
                raise ValueError("incomplete public cart item")
            rows.append(_crop(item, CartItem))
        cropped[name] = rows
    if _contains_secret(cropped, secrets):
        raise ValueError("credential in public cart")
    return CartSnapshot.model_validate(cropped)


def _target_card(card, target, product_name):
    from decimal import Decimal

    return (isinstance(card, CartActionCard) and card.product_id == target["product_id"]
            and card.unique_id == target["unique_id"] and card.quantity == target["quantity"]
            and card.sku_name == target["sku_name"] and card.unit_price == Decimal(target["unit_price"])
            and card.product_name == product_name and card.expires_at > time())


async def _workflow(client, agent, report, *, scenario, authorization, secrets, save, load_saved,
                    definition_unchanged):
    from decimal import Decimal

    target = scenario["target"]
    item = CartAddInput(product_id=target["product_id"], unique_id=target["unique_id"], quantity=target["quantity"])
    headers = {"Authorization": authorization}

    def guard(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        save()
        if not condition:
            raise VerificationStopped(name)

    async def request(name, method, endpoint, parser, *, payload=None, decision=None):
        report["phase"] = name
        row = {"name": name, "method": method, "endpoint": endpoint, "request_status": "not_attempted"}
        report["steps"].append(row)
        if decision is not None:
            guard(name + ".frozen_definition", definition_unchanged())
            attempt = {"step": name, "action_id": payload["action_id"], "decision": decision,
                       "request_status": "sending", "retry_permitted": False}
            report["decision_attempts"].append(attempt)
        save()
        started = perf_counter()
        try:
            response = await client.request(method, endpoint, headers=headers,
                **({"json": {"decision": decision}} if decision is not None else
                   {"json": payload} if payload is not None else {}))
        except httpx.HTTPError:
            row["request_status"] = "unknown" if decision is not None else "failed"
            if decision is not None:
                attempt["request_status"] = "unknown"
            raise VerificationStopped("decision_transport_error" if decision is not None else "local_http_error") from None
        row.update(request_status="completed", http_status=response.status_code,
                   elapsed_ms=round((perf_counter() - started) * 1000))
        if decision is not None:
            attempt["request_status"] = "completed"
        if response.status_code != 200:
            row["error_code"] = _http_error_code(response)
            save()
            raise VerificationStopped("decision_http_error" if decision is not None else "local_http_error")
        try:
            result = parser(_loads(response.content))
        except (ValueError, TypeError, UnicodeError, ValidationError, RecursionError):
            row["error_code"] = "invalid_public_response"
            save()
            raise VerificationStopped("invalid_public_response") from None
        if name == "identity":
            row["result"] = {"authenticated": True}
        elif isinstance(result, CartSnapshot):
            matching = _matching(result, item)
            row["result"] = {"same_sku_quantity": sum(value.quantity for value in matching),
                             "same_sku_cart_ids": [value.cart_id for value in matching],
                             "invalid_same_sku_count": len(_matching(result, item, invalid=True))}
        else:
            row["result"] = result.model_dump(mode="json")
        save()
        return result

    def cart_rows(name, cart):
        matching = _matching(cart, item)
        guard(name + ".unambiguous_valid_target", len(matching) <= 1 and not _matching(cart, item, invalid=True))
        return matching

    async def read_cart(name):
        cart = await request(name, "GET", "/agent/cart", lambda value: _cart_response(value, secrets))
        cart_rows(name, cart)
        return cart

    async def prepare(name, session_id, product_name):
        result = await request(name, "POST", "/agent/chat",
            lambda value: _public_session_response(value, secrets, session_id=session_id),
            payload={"message": scenario["prepare"], "session_id": session_id})
        guard(name + ".frozen_target", result.status == "awaiting_confirmation"
              and _target_card(result.pending_action, target, product_name))
        return result.pending_action

    async def decide(name, card, decision, product_name):
        guard(name + ".frozen_target", _target_card(card, target, product_name))
        before = len(agent.events())
        result = await request(name, "POST", f"/agent/actions/{card.action_id}/decision",
            lambda value: _public_action_response(value, secrets, action_id=card.action_id),
            payload={"action_id": card.action_id}, decision=decision)
        events = agent.events()[before:]
        count = _counts(events)["business_write_requests"]
        return result, count, events

    guard("frozen_definition_before_launch", definition_unchanged())
    first_pid = agent.start(1)
    report["processes"].append({"launch": 1, "pid": first_pid})
    save()
    await agent.ready(client)
    identity = await request("identity", "GET", "/agent/me", lambda value: UserIdentity.model_validate(_crop(value, UserIdentity)))
    secrets = (*secrets, identity.nickname)
    model_before = len(agent.events())
    queried = await request("query", "POST", "/agent/chat", lambda value: _public_session_response(value, secrets),
                            payload={"message": scenario["query"]})
    products = queried.products
    guard("query.frozen_target_first", queried.status == "answered" and bool(products)
          and products[0].product_id == target["product_id"]
          and any(sku.unique_id == target["unique_id"] and sku.name == target["sku_name"]
                  and sku.price == Decimal(target["unit_price"]) and sku.stock > 0 for sku in products[0].skus)
          and all(sku.price <= Decimal("3") and sku.stock > 0 for product in products for sku in product.skus)
          and any(trace.name == "search_products" and trace.status == "succeeded" for trace in queried.tool_calls))
    query_counts = _counts(agent.events()[model_before:])
    report["observations"]["query"] = query_counts
    guard("query.observed_model_http_and_strict_protocol", query_counts["model_http_requests"] > 0
          and query_counts["model_http_requests"] == query_counts["model_http_successes"]
          == query_counts["model_protocol_valid"] and query_counts["model_protocol_invalid"] == 0
          and query_counts["model_thinking_false_observed"] and query_counts["business_write_requests"] == 0)
    product_name = products[0].name
    session_id = queried.session_id
    report["session_id"] = session_id
    before_cancel = await read_cart("cart_before_cancel")
    first_card = await prepare("prepare_cancel", session_id, product_name)
    cancelled, count, _ = await decide("cancel", first_card, "cancel", product_name)
    report["downstream_writes"]["cancel"] = count
    guard("cancel.saved_id_and_zero_writes", cancelled.status == "cancelled" and count == 0)
    after_cancel = await read_cart("cart_after_cancel")
    before_rows, after_rows = _matching(before_cancel, item), _matching(after_cancel, item)
    report["cart_checks"]["cancel"] = {
        "before_quantity": sum(row.quantity for row in before_rows),
        "after_quantity": sum(row.quantity for row in after_rows),
        "before_cart_ids": [row.cart_id for row in before_rows], "after_cart_ids": [row.cart_id for row in after_rows],
    }
    guard("cancel.same_sku_unchanged", report["cart_checks"]["cancel"]["before_quantity"] ==
          report["cart_checks"]["cancel"]["after_quantity"] and
          report["cart_checks"]["cancel"]["before_cart_ids"] == report["cart_checks"]["cancel"]["after_cart_ids"])
    saved_card = await prepare("prepare_saved", session_id, product_name)
    guard("prepare_saved.new_action_id", saved_card.action_id != first_card.action_id)
    report["saved_action_id"] = saved_card.action_id
    report["phase"] = "stop_agent"
    save()
    agent.stop()
    report["phase"] = "start_agent_same_database"
    second_pid = agent.start(2)
    report["processes"].append({"launch": 2, "pid": second_pid})
    save()
    await agent.ready(client)
    guard("restart.new_owned_process", first_pid != second_pid)
    restored_id = TypeAdapter(ActionId).validate_python(load_saved())
    guard("restart.saved_report_action_id", restored_id == saved_card.action_id)
    recovered = await request("recover_saved_action", "GET", f"/agent/actions/{restored_id}",
        lambda value: _public_action_response(value, secrets, action_id=restored_id))
    guard("restart.exact_saved_card", recovered.status == "awaiting_confirmation"
          and recovered.pending_action is not None and recovered.pending_action == saved_card
          and recovered.session_id == session_id)
    before_confirm = await read_cart("cart_before_confirm")
    guard("no_downstream_writes_before_confirmation", _counts(agent.events())["business_write_requests"] == 0)
    confirmed, count, events = await decide("confirm_once", saved_card, "confirm", product_name)
    report["downstream_writes"]["confirm"] = count
    guard("confirm.saved_id_succeeded", confirmed.status == "succeeded")
    guard("confirm.exactly_one_downstream_cart_add", count == 1 and
          all(event["method"] == "POST" and event["path"] == "/api/cart/add" and event.get("status") == 200
              for event in events if event["kind"] == "shop_request" and event["method"] not in {"GET", "HEAD"}))
    after_confirm = await read_cart("cart_after_confirm")
    verification = _verify_cart(after_confirm, item, _matching(before_confirm, item), confirmed.action_result.cart_id)
    verification["returned_cart_id"] = confirmed.action_result.cart_id
    report["cart_checks"]["confirm"] = verification
    guard("confirm.exact_quantity_growth_and_cart_id", verification["status"] == "passed")
    duplicate, count, _ = await decide("confirm_duplicate", saved_card, "confirm", product_name)
    report["downstream_writes"]["duplicate"] = count
    guard("duplicate.same_result_and_zero_writes", count == 0 and duplicate.action_result == confirmed.action_result
          and duplicate.status == "succeeded")
    after_duplicate = await read_cart("cart_after_duplicate")
    previous_rows, duplicate_rows = _matching(after_confirm, item), _matching(after_duplicate, item)
    report["cart_checks"]["duplicate"] = {
        "before_quantity": sum(row.quantity for row in previous_rows),
        "after_quantity": sum(row.quantity for row in duplicate_rows),
        "before_cart_ids": [row.cart_id for row in previous_rows],
        "after_cart_ids": [row.cart_id for row in duplicate_rows],
    }
    guard("duplicate.same_sku_unchanged", report["cart_checks"]["duplicate"]["before_quantity"] ==
          report["cart_checks"]["duplicate"]["after_quantity"] and
          report["cart_checks"]["duplicate"]["before_cart_ids"] == report["cart_checks"]["duplicate"]["after_cart_ids"])
    guard("frozen_definition_after_workflow", definition_unchanged())


def _free_port(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise VerificationStopped("dedicated_port_unavailable") from None


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_file: Path = SERVICE_ENV_FILE, stdout: TextIO | None = None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    environment = os.environ if environ is None else environ
    report, path, agent, owns_report = None, None, None, False
    authorization, secrets = "", ()
    try:
        parser = _SafeParser(description="One frozen real workflow; explicit live-write opt-in is mandatory.",
                             allow_abbrev=False)
        parser.add_argument("--suffix", required=True)
        parser.add_argument("--port", type=_integer, default=8016)
        parser.add_argument("--allow-live-write", action="store_true")
        options = _arguments(argv, parser=parser, flags={"--suffix", "--port", "--allow-live-write"})
        if not options.allow_live_write:
            raise VerificationStopped("live_write_opt_in_required")
        if (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", options.suffix) is None
                or not 1024 <= options.port <= 65535):
            raise ShopError("invalid_request")
        path = _evidence_path(PROJECT_ROOT / "logs" / "agent-day6" / f"live-action-{options.suffix}.json")
        database = _database_path(SERVICE_ROOT / ".tmp" / f"day6-live-{options.suffix}" / "sessions.sqlite3")
        event_paths = [_evidence_path(PROJECT_ROOT / "logs" / "agent-day6" /
                       f"live-events-{options.suffix}-{number}.jsonl") for number in (1, 2)]
        if path.exists() or database.parent.exists() or any(value.exists() for value in event_paths):
            raise VerificationStopped("existing_evidence")
        scenario, digest = _load_cases(CASES_PATH)
        authorization = _authorization(environment, Path(env_file))
        secrets = _secrets(environment, env_file, authorization)
        if _contains_secret(options.suffix, secrets):
            raise ShopError("invalid_request")
        _free_port(options.port)
        report = _new_report(digest, observation_source="owned_service_http")
        report["declarations"]["live_write_opt_in"] = True
        report.update(agent_port=options.port, database=database.relative_to(SERVICE_ROOT).as_posix(),
                      observation_files=[value.relative_to(PROJECT_ROOT).as_posix() for value in event_paths])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            owns_report = True
            stream.write(json.dumps(_redact_secrets(report, secrets), ensure_ascii=False, indent=2) + "\n")
        database.parent.mkdir(parents=True, exist_ok=False)
        agent = OwnedAgent(port=options.port, database=database, event_paths=event_paths, environ=environment)

        def save():
            _save_owned_report(_evidence_path(path), _redact_secrets(report, secrets), authorization)

        def load_saved():
            stored = _loads(_evidence_path(path).read_bytes())
            return stored["saved_action_id"]

        async def execute():
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{options.port}", timeout=600.0,
                                         trust_env=False, follow_redirects=False) as client:
                await _workflow(client, agent, report, scenario=scenario, authorization=authorization,
                    secrets=secrets, save=save, load_saved=load_saved,
                    definition_unchanged=lambda: hashlib.sha256(CASES_PATH.read_bytes()).hexdigest() == digest)
        asyncio.run(execute())
        report["status"] = "passed"
    except SystemExit as error:
        return int(error.code or 0)
    except VerificationStopped as error:
        if report is None:
            print(json.dumps({"status": "not_run", "error_code": error.code}), file=stdout)
            return 2
        report.update(status="failed", error_code=error.code)
    except ShopError as error:
        if report is None:
            print(json.dumps({"status": "not_run", "error_code": error.code}), file=stdout)
            return 2
        report.update(status="failed", error_code=error.code)
    except OSError:
        if report is None:
            print(json.dumps({"status": "not_run", "error_code": "local_file_or_process_error"}), file=stdout)
            return 2
        report.update(status="failed", error_code="local_file_or_process_error")
    except Exception:
        if report is None:
            print(json.dumps({"status": "not_run", "error_code": "invalid_verifier_configuration"}), file=stdout)
            return 2
        report.update(status="failed", error_code="verification_error")
    finally:
        if agent is not None:
            try:
                agent.stop()
            except (OSError, subprocess.TimeoutExpired):
                report.update(status="failed", error_code="owned_process_stop_failed")
            try:
                counts = _counts(agent.events())
                report["observations"]["total"] = counts
                report["provenance"]["external_model_verified"] = (
                    counts["model_http_requests"] > 0 and counts["model_http_requests"] ==
                    counts["model_http_successes"] == counts["model_protocol_valid"]
                    and counts["model_protocol_invalid"] == 0 and counts["model_thinking_false_observed"])
                if (report["status"] == "passed" and (counts["business_write_requests"] != 1
                        or counts["cart_add_requests"] != 1 or counts["unexpected_requests"] != 0
                        or not report["provenance"]["external_model_verified"])):
                    report.update(status="failed", error_code="observed_acceptance_incomplete")
            except (OSError, ValueError, TypeError):
                report.update(status="failed", error_code="invalid_observations")
        if owns_report:
            report["passed"] = sum(value["passed"] for value in report["checks"])
            report["failed"] = len(report["checks"]) - report["passed"]
            try:
                _save_owned_report(_evidence_path(path), _redact_secrets(report, secrets), authorization)
            except (OSError, ValueError):
                report.update(status="failed", error_code="report_write_failed")
    summary = {"status": report["status"], "passed": report.get("passed", 0),
               "failed": report.get("failed", 0), "report": str(path)}
    if "error_code" in report:
        summary["error_code"] = report["error_code"]
    print(json.dumps(_redact_secrets(summary, secrets)), file=stdout)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
