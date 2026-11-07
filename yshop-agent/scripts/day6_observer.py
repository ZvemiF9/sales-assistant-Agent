"""Owned Day 6 service with metadata-only HTTP observations, no replacements."""

import json
import os
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "yshop-agent"
sys.path.insert(0, str(SERVICE))

import httpx

from app.agent.model import _loads, _parse_reply
from scripts.day6_verify import (
    MODEL_OVERRIDES, PROTOCOL_REASONS, _database_path, _evidence_path, _validate_event,
)
from scripts.eval_policy import _arguments
from scripts.smoke_test import _SafeParser, _integer


def _observed_send(original, record, *, shop_url, model_url, model_key):
    shop = httpx.URL(shop_url)
    model = httpx.URL(model_url)

    async def send(client, request, *args, **kwargs):
        origin = request.url.copy_with(path=shop.path, query=None, fragment=None)
        kind = ("shop_request" if origin == shop.copy_with(query=None, fragment=None) else
                "model_request" if request.url == model else "unexpected_request")
        method = request.method if request.method in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"} else "other"
        entry = {"kind": kind, "method": method}
        if kind == "shop_request":
            from scripts.day6_verify import _SHOP_PATHS
            import re

            path = request.url.path
            entry["path"] = path if path in _SHOP_PATHS or re.fullmatch(r"/api/product/detail/[1-9][0-9]{0,18}", path) else "other_shop_path"
        elif kind == "model_request":
            try:
                value = _loads(request.content)
                mode = value.get("enable_thinking") if isinstance(value, dict) else "unexpected"
                entry["enable_thinking"] = mode if mode is None or type(mode) is bool else "unexpected"
            except (ValueError, TypeError, UnicodeError):
                entry["enable_thinking"] = "unobserved"
        started = perf_counter()
        try:
            response = await original(client, request, *args, **kwargs)
        except Exception as error:
            category = ("timeout" if isinstance(error, httpx.TimeoutException) else
                        "network_error" if isinstance(error, httpx.RequestError) else "unexpected_error")
            record({**entry, "transport_error": category,
                    "elapsed_ms": round((perf_counter() - started) * 1000)})
            raise
        record({**entry, "status": response.status_code, "elapsed_ms": round((perf_counter() - started) * 1000)})
        if kind == "model_request" and response.status_code == 200:
            phase = "envelope"
            try:
                parsed = _loads(response.content)
                phase = "reply"
                _parse_reply(parsed, model_key)
            except Exception as error:
                reason = str(error) if type(error) is ValueError and str(error) in PROTOCOL_REASONS else "invalid_schema_or_json"
                record({"kind": "model_protocol_check", "status": "invalid", "phase": phase, "reason": reason})
            else:
                record({"kind": "model_protocol_check", "status": "valid"})
        return response

    return send


def _run_server(port, events, database):
    import uvicorn
    from app.agent.config import AgentSettings
    from app.config import Settings
    from app.main import create_app
    from app.storage.config import SessionSettings

    os.environ.update(MODEL_OVERRIDES)
    settings = AgentSettings()
    shop_settings = Settings()
    original = httpx.AsyncClient.send
    with events.open("x", encoding="utf-8") as stream:
        def record(value):
            stream.write(json.dumps(_validate_event(value)) + "\n")
            stream.flush()

        record({"kind": "startup", "pid": os.getpid(), "port": port})
        record({"kind": "model_configuration", "configured": settings.configured,
                "timeout_seconds": settings.llm_timeout_seconds, "enable_thinking": settings.llm_enable_thinking})
        httpx.AsyncClient.send = _observed_send(original, record, shop_url=shop_settings.yshop_api_url,
            model_url=settings.llm_base_url + "chat/completions" if settings.configured else "http://127.0.0.1:1/",
            model_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else "")
        try:
            app = create_app(settings=shop_settings, agent_settings=settings,
                session_settings=SessionSettings(_env_file=None, db_path=database))
            uvicorn.run(app, host="127.0.0.1", port=port, access_log=False, log_level="critical")
        finally:
            httpx.AsyncClient.send = original
    return 0


def main(argv=None):
    try:
        parser = _SafeParser(description="Run only an owned Day 6 observed service.", allow_abbrev=False)
        parser.add_argument("--port", type=_integer, required=True)
        parser.add_argument("--events", type=Path, required=True)
        parser.add_argument("--database", type=Path, required=True)
        options = _arguments(argv, parser=parser, flags={"--port", "--events", "--database"})
        if not 1024 <= options.port <= 65535:
            return 2
        events = _evidence_path(options.events, project=ROOT)
        database = _database_path(options.database, service=SERVICE)
        if events.exists():
            return 2
        return _run_server(options.port, events, database)
    except SystemExit as error:
        return int(error.code or 0)
    except Exception:
        # Never print configuration, upstream content or exception text.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
