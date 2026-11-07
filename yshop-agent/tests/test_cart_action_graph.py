"""Inspect the real persisted next node and interrupt, then resume after reopen."""

import importlib
import importlib.util
import json
import sqlite3
from functools import partial

import pytest

from app.agent.tools import QueryTools, function_schemas
from app.errors import ShopError
from tests.test_agent_graph import AUTH
from tests.test_cart_action_api import (
    ShopTransport, client_for, database, decision, pending, prepare, selected_session,
)


def service(app):
    assert importlib.util.find_spec("app.agent.cart_action"), "real interrupt cart service missing"
    module = importlib.import_module("app.agent.cart_action")
    return module.CartActionService(app.state.sessions, app.state.checkpoints)


def graph(client, app):
    tools = QueryTools(app.state.shop, app.state.policy, AUTH, public_secrets=("secret-nickname",))
    return service(app).graph(33, AUTH, tools)


def inspect_current_resume(monkeypatch, aid, shop):
    from app.agent.cart_action import CartActionService
    from langgraph.types import Command

    original_graph = CartActionService.graph
    resumed = []

    def inspect_graph(self, *args, **kwargs):
        compiled = original_graph(self, *args, **kwargs)
        original_invoke = compiled.ainvoke

        async def inspect_resume(value, config, *invoke_args, **invoke_kwargs):
            if isinstance(value, Command):
                state = await compiled.aget_state(config)
                assert state.next == ("wait",)
                assert len(state.tasks) == 1 and len(state.tasks[0].interrupts) == 1
                assert state.tasks[0].name == "wait"
                assert state.tasks[0].interrupts[0].value["action_id"] == aid
                assert not shop.writes
                resumed.append(value.resume)
            return await original_invoke(value, config, *invoke_args, **invoke_kwargs)

        monkeypatch.setattr(compiled, "ainvoke", inspect_resume)
        return compiled

    monkeypatch.setattr(CartActionService, "graph", inspect_graph)
    return resumed


def test_real_checkpoint_wait_interrupt_survives_application_reopen(database):
    shop = ShopTransport()
    with client_for(database, shop) as (client, app, shop):
        aid = pending(client, app)
        compiled = graph(client, app)
        state = client.portal.call(compiled.aget_state, {"configurable": {"thread_id": aid}})
        assert state.next == ("wait",)
        assert len(state.tasks) == 1 and len(state.tasks[0].interrupts) == 1
        assert state.tasks[0].interrupts[0].value["action_id"] == aid
        assert state.values["action_id"] == aid
        serialized = json.dumps(state.values)
        assert AUTH not in serialized and "secret-nickname" not in serialized
    with client_for(database, shop) as (client, app, shop):
        assert app.state.sessions.get(aid, 33).status == "pending"
        assert decision(client, aid).json()["status"] == "succeeded"
        assert len(shop.writes) == 1


def test_checkpoint_has_no_authoritative_write_parameters(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        compiled = graph(client, app)
        config = {"configurable": {"thread_id": aid}}
        state = client.portal.call(compiled.aget_state, config)
        tampered = dict(state.values["card"])
        tampered.update(quantity=9999, product_id=99, unique_id="wrong", unit_price="0", total_price="0")
        client.portal.call(partial(compiled.aupdate_state, config, {"card": tampered}, as_node="prepare"))
        client.portal.call(compiled.ainvoke, None, config)
        assert decision(client, aid).json()["status"] == "succeeded"
        assert shop.existing == 1


def test_checkpointer_guard_rejects_foreign_application_id_without_mutation(database):
    assert importlib.util.find_spec("app.storage.checkpoints"), "guarded async checkpointer missing"
    module = importlib.import_module("app.storage.checkpoints")
    path = database.parent / "cart-checkpoints.sqlite3"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE protected (value TEXT)")
        connection.execute("INSERT INTO protected VALUES ('unchanged')")
    before = path.read_bytes()
    import asyncio
    async def open_foreign():
        async with module.open_checkpoints(path):
            pytest.fail("foreign database accepted")
    with pytest.raises(ShopError) as error:
        asyncio.run(open_foreign())
    assert error.value.code == "session_unavailable" and path.read_bytes() == before


def test_direct_chat_contract_and_three_read_only_tools_stay_fixed():
    from app.agent.schemas import ChatResponse, SessionChatResponse
    assert [entry["function"]["name"] for entry in function_schemas()] == [
        "search_products", "get_product_detail", "search_policy",
    ]
    with pytest.raises(ValueError):
        ChatResponse(answer="confirm", status="awaiting_confirmation", model_rounds=0)
    assert SessionChatResponse.model_fields["model_rounds"].metadata[1].ge == 0


def test_checkpointer_path_matches_approved_dedicated_name(database):
    from app.storage.config import SessionSettings
    settings = SessionSettings(_env_file=None, db_path=database)
    assert settings.checkpoint_path == database.parent / "cart-checkpoints.sqlite3"


def test_serializer_explicitly_disables_pickle_and_user_modules(database):
    with client_for(database) as (client, app, shop):
        serde = app.state.checkpoints.serde
        assert serde.pickle_fallback is False
        assert serde._allowed_json_modules in (None, set())
        assert serde._allowed_msgpack_modules in (None, set())


def test_missing_saved_interrupt_is_rebuilt_before_current_confirm(database, monkeypatch):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        client.portal.call(app.state.checkpoints.adelete_thread, aid)
        resumed = inspect_current_resume(monkeypatch, aid, shop)
        response = decision(client, aid)
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "succeeded"
        assert resumed == [{"decision": "confirm"}]
        assert app.state.sessions.get(aid, 33).status == "succeeded" and len(shop.writes) == 1


def test_action_commit_before_checkpoint_failure_recovers_same_wait_after_reopen(database, monkeypatch):
    shop = ShopTransport()
    with client_for(database, shop) as (client, app, shop):
        sid = selected_session(app)

        async def unavailable(*args, **kwargs):
            raise ShopError("session_unavailable")

        monkeypatch.setattr(app.state.checkpoints, "aput", unavailable)
        failed = prepare(client, sid)
        assert failed.status_code == 503
        record = app.state.sessions.pending(sid, 33)
        assert record is not None and record.status == "pending" and not shop.writes
        card = record.card.model_dump(mode="json")
        aid = record.card.action_id
    with client_for(database, shop) as (client, app, shop):
        recovered = prepare(client, sid, "确认，继续加入购物车9999件")
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["pending_action"] == card
        state = client.portal.call(graph(client, app).aget_state, {"configurable": {"thread_id": aid}})
        assert state.next == ("wait",)
        assert len(state.tasks) == 1 and len(state.tasks[0].interrupts) == 1
        assert state.tasks[0].interrupts[0].value == card
        assert app.state.sessions.pending(sid, 33).card.action_id == aid
        assert not shop.writes and app.state.chat_model.seen == []
        assert decision(client, aid).json()["status"] == "succeeded"
        assert decision(client, aid).json()["status"] == "succeeded"
        assert len(shop.writes) == 1


@pytest.mark.parametrize("checkpoint", ["missing", "execute", "completed"])
@pytest.mark.parametrize("current_decision", ["cancel", "confirm"])
@pytest.mark.parametrize("recovery_entry", ["chat", "decision"])
def test_recovery_does_not_replay_old_checkpoint_confirm(
        database, checkpoint, current_decision, recovery_entry, monkeypatch):
    shop = ShopTransport()
    with client_for(database, shop) as (client, app, shop):
        aid = pending(client, app)
        config = {"configurable": {"thread_id": aid}}
        compiled = graph(client, app)
        if checkpoint == "missing":
            client.portal.call(app.state.checkpoints.adelete_thread, aid)
        else:
            if checkpoint == "execute":
                from app.actions.schemas import ActionDecision
                lease = app.state.sessions.claim(app.state.sessions.get(aid, 33).card.session_id, 33)
                tools = QueryTools(app.state.shop, app.state.policy, AUTH)
                compiled = service(app).graph(
                    33, AUTH, tools, lease=lease, action_id=aid, approved=ActionDecision(decision="confirm"),
                )
            try:
                client.portal.call(partial(
                    compiled.aupdate_state, config, {"decision": {"decision": "confirm"}},
                    as_node="wait" if checkpoint == "execute" else "execute",
                ))
            finally:
                if checkpoint == "execute":
                    app.state.sessions.release(lease)
            saved = client.portal.call(compiled.aget_state, config)
            assert saved.values["decision"] == {"decision": "confirm"}
            assert saved.next == (("execute",) if checkpoint == "execute" else ())
        assert app.state.sessions.get(aid, 33).status == "pending" and not shop.writes
    with client_for(database, shop) as (client, app, shop):
        record = app.state.sessions.get(aid, 33)
        resumed = inspect_current_resume(monkeypatch, aid, shop)
        if recovery_entry == "chat":
            recovered = prepare(client, record.card.session_id, "确认，继续加入购物车9999件")
            assert recovered.status_code == 200, recovered.text
            assert recovered.json()["pending_action"] == record.card.model_dump(mode="json")
            saved = client.portal.call(graph(client, app).aget_state, config)
            assert saved.next == ("wait",)
            assert len(saved.tasks) == 1 and len(saved.tasks[0].interrupts) == 1
            assert saved.tasks[0].interrupts[0].value == record.card.model_dump(mode="json")
            assert "decision" not in saved.values
            assert not shop.writes and app.state.chat_model.seen == []
        response = decision(client, aid, current_decision)
        assert response.status_code == 200, response.text
        expected = "cancelled" if current_decision == "cancel" else "succeeded"
        assert response.json()["status"] == expected
        assert app.state.sessions.get(aid, 33).status == expected
        assert resumed == [{"decision": current_decision}]
        assert len(shop.writes) == (0 if current_decision == "cancel" else 1)
        assert decision(client, aid).json()["status"] == expected
        assert len(shop.writes) == (0 if current_decision == "cancel" else 1)


@pytest.mark.parametrize("terminal", ["executing", "cancelled"])
def test_missing_checkpoint_never_reconstructs_executing_or_terminal_actions(database, terminal):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        record = app.state.sessions.get(aid, 33)
        if terminal == "executing":
            lease = app.state.sessions.claim(record.card.session_id, 33)
            app.state.sessions.begin_execution(lease, 33, aid)
            app.state.sessions.release(lease)
        else:
            assert decision(client, aid, "cancel").json()["status"] == "cancelled"
        client.portal.call(app.state.checkpoints.adelete_thread, aid)
        response = decision(client, aid)
        expected = "unknown" if terminal == "executing" else "cancelled"
        assert response.status_code == 200 and response.json()["status"] == expected
        saved = client.portal.call(graph(client, app).aget_state, {"configurable": {"thread_id": aid}})
        assert saved.next == () and saved.tasks == () and not saved.values
        assert not shop.writes


def test_chat_recovers_interrupted_execution_as_unknown_without_rebuilding_wait(database):
    shop = ShopTransport()
    with client_for(database, shop) as (client, app, shop):
        aid = pending(client, app)
        record = app.state.sessions.get(aid, 33)
        lease = app.state.sessions.claim(record.card.session_id, 33)
        app.state.sessions.begin_execution(lease, 33, aid)
        app.state.sessions.release(lease)
        client.portal.call(app.state.checkpoints.adelete_thread, aid)
    with client_for(database, shop) as (client, app, shop):
        response = prepare(client, record.card.session_id, "继续加入一件购物车")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "unknown"
        assert response.json()["action_result"]["action_id"] == aid
        assert app.state.sessions.get(aid, 33).status == "unknown"
        assert app.state.sessions.load(record.card.session_id, 33).pending_action_id is None
        saved = client.portal.call(graph(client, app).aget_state, {"configurable": {"thread_id": aid}})
        assert saved.next == () and saved.tasks == () and not saved.values
        assert not shop.writes and app.state.chat_model.seen == []
        assert decision(client, aid).json()["status"] == "unknown" and not shop.writes
