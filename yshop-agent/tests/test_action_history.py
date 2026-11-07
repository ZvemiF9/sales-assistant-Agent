"""Durable metadata history over the real API, store and checkpoint workflow."""

import sqlite3

import pytest
from pydantic import ValidationError

from app.errors import ShopError
from app.storage.actions import CartActionStore
from app.actions.history import ActionHistory
from tests.test_cart_action_api import client_for, database, decision, pending
from tests.test_agent_graph import AUTH


def history(client, aid):
    return client.get(f"/agent/actions/{aid}/history", headers={"Authorization": AUTH})


def test_history_reopens_and_terminal_replay_does_not_append(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        first = history(client, aid)
        assert first.status_code == 200, first.text
        assert [e["kind"] for e in first.json()["events"]] == ["prepared"]
        assert decision(client, aid).json()["status"] == "succeeded"
        before = history(client, aid).json()
        assert [e["kind"] for e in before["events"]] == ["prepared", "executing", "finished"]
        assert [e["status"] for e in before["events"]] == ["pending", "executing", "succeeded"]
        assert [e["sequence"] for e in before["events"]] == [1, 2, 3]
        assert before["action_id"] == aid
        decision(client, aid)
        assert history(client, aid).json() == before
        assert len(shop.writes) == 1
    with client_for(database) as (client, app, shop):
        assert history(client, aid).json() == before
        assert shop.writes == []


def test_history_owner_and_absence_are_indistinguishable(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        shop.user_id = 34
        foreign, missing = history(client, aid), history(client, "f" * 32)
        assert foreign.status_code == missing.status_code == 404
        assert foreign.json() == missing.json()
        assert shop.writes == []


def test_replacement_history_links_both_immutable_actions(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        original = app.state.sessions.get(aid, 33).card
        shop.price = "12.00"
        changed = decision(client, aid).json()
        fresh = changed["pending_action"]["action_id"]
        old_events = history(client, aid).json()["events"]
        new_events = history(client, fresh).json()["events"]
        assert old_events[-1]["kind"] == "replaced"
        assert old_events[-1]["status"] == "cancelled"
        assert old_events[-1]["related_action_id"] == fresh
        assert new_events[0]["related_action_id"] == aid
        assert app.state.sessions.get(aid, 33).card == original
        assert shop.writes == []


@pytest.mark.parametrize("fault", ["timeout", "mismatch"])
def test_unknown_history_is_persistent_and_replay_free(database, fault):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        setattr(shop, fault, True)
        assert decision(client, aid).json()["status"] == "unknown"
        events = history(client, aid).json()["events"]
        assert events[-1]["status"] == "unknown"
        assert events[-1]["error_code"] == ("cart_result_unknown" if fault == "timeout" else "cart_verification_failed")
        decision(client, aid)
        assert history(client, aid).json()["events"] == events
        assert len(shop.writes) == 1


def test_event_insert_failure_rolls_back_execution_before_shop_write(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        assert history(client, aid).status_code == 200
        with sqlite3.connect(database) as connection:
            connection.execute("""CREATE TRIGGER block_event BEFORE INSERT ON cart_action_events
                BEGIN SELECT RAISE(ABORT, 'private event failure'); END""")
        response = decision(client, aid)
        assert response.status_code == 503
        assert "private" not in response.text
        assert app.state.sessions.get(aid, 33).status == "pending"
        assert len(history(client, aid).json()["events"]) == 1
        assert shop.writes == []


def test_result_event_failure_preserves_executing_fence_then_recovers_unknown(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        assert history(client, aid).status_code == 200
        with sqlite3.connect(database) as connection:
            connection.execute("""CREATE TRIGGER block_finish_event BEFORE INSERT ON cart_action_events
                WHEN NEW.kind = 'finished' BEGIN SELECT RAISE(ABORT, 'private result failure'); END""")
        assert decision(client, aid).status_code == 503
        assert app.state.sessions.get(aid, 33).status == "executing"
        assert [e["kind"] for e in history(client, aid).json()["events"]] == ["prepared", "executing"]
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TRIGGER block_finish_event")
        assert decision(client, aid).json()["status"] == "unknown"
        assert history(client, aid).json()["events"][-1]["status"] == "unknown"
        assert len(shop.writes) == 1


def test_legacy_observation_is_idempotent_without_invented_execution_history(database):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        decision(client, aid, "cancel")
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE IF EXISTS cart_action_events")
    with client_for(database) as (client, app, shop):
        response = history(client, aid)
        assert response.status_code == 200, response.text
        events = response.json()["events"]
        assert len(events) == 1 and events[0]["kind"] == "legacy"
        assert events[0]["status"] == "cancelled"
        assert events[0]["request_id"] is None
        assert shop.writes == []
    with client_for(database) as (client, app, shop):
        assert history(client, aid).json()["events"] == events


@pytest.mark.parametrize("column,value", [
    ("status", "succeeded"), ("sequence", 9), ("request_id", "private-token"),
    ("error_code", "private-error"), ("related_action_id", "f" * 32),
])
def test_corrupt_history_fails_closed_without_leaking_metadata(database, column, value):
    with client_for(database) as (client, app, shop):
        aid = pending(client, app)
        assert history(client, aid).status_code == 200
        with sqlite3.connect(database) as connection:
            connection.execute(f"UPDATE cart_action_events SET {column} = ?", (value,))
        response = history(client, aid)
        assert response.status_code == 503
        assert "private" not in response.text
        assert shop.writes == []


@pytest.mark.parametrize("change", ["no_unique", "nullable_status", "wrong_type", "no_primary_key"])
def test_same_column_names_with_incompatible_event_schema_are_rejected(database, change):
    with client_for(database) as (client, app, shop):
        pending(client, app)
    ddl = """CREATE TABLE cart_action_events (
        event_id INTEGER PRIMARY KEY, action_id TEXT NOT NULL, sequence INTEGER NOT NULL,
        kind TEXT NOT NULL, status TEXT NOT NULL, occurred_at REAL NOT NULL,
        request_id TEXT, related_action_id TEXT, error_code TEXT, UNIQUE(action_id, sequence))"""
    ddl = {
        "no_unique": ddl.replace(", UNIQUE(action_id, sequence)", ""),
        "nullable_status": ddl.replace("status TEXT NOT NULL", "status TEXT"),
        "wrong_type": ddl.replace("sequence INTEGER", "sequence TEXT"),
        "no_primary_key": ddl.replace("event_id INTEGER PRIMARY KEY", "event_id INTEGER"),
    }[change]
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE cart_action_events")
        connection.execute(ddl)
    with pytest.raises(ShopError) as captured:
        store = CartActionStore(database)
        store.close()
    assert captured.value.code == "session_unavailable"


def test_history_rejects_cancel_after_execution():
    with pytest.raises(ValidationError):
        ActionHistory(action_id="a" * 32, session_id="b" * 32, events=[
            {"sequence": 1, "kind": "prepared", "status": "pending", "occurred_at": 1.0},
            {"sequence": 2, "kind": "executing", "status": "executing", "occurred_at": 2.0},
            {"sequence": 3, "kind": "finished", "status": "cancelled", "occurred_at": 3.0},
        ])
