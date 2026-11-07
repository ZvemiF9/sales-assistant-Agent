"""Server-generated request correlation across HTTP, graph and SQLite threads."""

import asyncio
import logging
import re

import httpx
from langchain_core.messages import AIMessage

from tests.test_agent_graph import AUTH, ScriptedModel, call
from tests.test_cart_action_api import client_for, database, decision, pending
from tests.test_action_history import history


def test_server_request_id_ignores_client_header_and_is_fresh(database, caplog):
    with client_for(database) as (client, app, shop):
        caplog.set_level(logging.INFO, logger="app")
        supplied = "private-client-id"
        first = client.get("/health?token=private-query", headers={"X-Request-ID": supplied})
        second = client.get("/private-path-secret", headers={"Authorization": AUTH})
        assert re.fullmatch("[0-9a-f]{32}", first.headers.get("X-Request-ID", ""))
        assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
        messages = [r.getMessage() for r in caplog.records if r.name.startswith("app")]
        assert any(first.headers["X-Request-ID"] in text and "status=200" in text for text in messages)
        assert any(second.headers["X-Request-ID"] in text and "status=404" in text for text in messages)
        assert not any(secret in "\n".join(messages) for secret in (supplied, "private-query", "private-path-secret", AUTH))


def test_history_events_use_current_request_id_in_threadpool(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        prepared = history(client, aid).json()["events"][0]
        response = decision(client, aid)
        events = history(client, aid).json()["events"]
        assert re.fullmatch("[0-9a-f]{32}", prepared["request_id"] or "")
        assert all(e["request_id"] == response.headers["X-Request-ID"] for e in events[1:])
        assert prepared["request_id"] != response.headers["X-Request-ID"]


def test_model_and_tool_logs_correlate_without_message_or_credentials(database, caplog):
    with client_for(database) as (client, app, shop):
        app.state.chat_model = ScriptedModel([
            AIMessage(content="", tool_calls=[call("search_products", {"keyword": "Phone"})],
                      usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            AIMessage(content=""),
        ])
        caplog.set_level(logging.INFO, logger="app")
        response = client.post("/agent/chat", json={"message": "查Phone private-message"},
                               headers={"Authorization": AUTH})
        assert response.status_code == 200
        rid = response.headers.get("X-Request-ID")
        assert rid is not None
        records = [r.getMessage() for r in caplog.records if r.name in {"app.agent.graph", "app.agent.tools"}]
        assert len(records) == 3 and all(f"request_id={rid}" in r for r in records)
        assert any("input_tokens=3" in r and "output_tokens=2" in r for r in records)
        assert not any(secret in "\n".join(records) for secret in ("private-message", AUTH, "secret-nickname"))


def test_concurrent_requests_and_error_context_are_isolated(database):
    with client_for(database) as (client, app, shop):
        from app.telemetry import request_id

        @app.get("/test-context")
        async def context():
            before = request_id()
            await asyncio.sleep(0.01)
            return {"before": before, "after": request_id()}

        @app.get("/test-error")
        async def broken():
            raise RuntimeError("private-exception")

        async def invoke():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app, raise_app_exceptions=False),
                                         base_url="http://test") as http:
                responses = await asyncio.gather(*[http.get("/test-context") for _ in range(4)])
                failed = await http.get("/test-error")
                assert failed.status_code == 500
                assert re.fullmatch("[0-9a-f]{32}", failed.headers.get("X-Request-ID", ""))
                assert failed.headers["X-Request-ID"] not in {r.headers["X-Request-ID"] for r in responses}
                assert request_id() is None
                return responses

        responses = client.portal.call(invoke)
        ids = [r.headers["X-Request-ID"] for r in responses]
        assert len(set(ids)) == 4
        assert [r.json() for r in responses] == [{"before": rid, "after": rid} for rid in ids]
        assert request_id() is None
