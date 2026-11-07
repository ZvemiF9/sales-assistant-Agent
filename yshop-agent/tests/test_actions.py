"""Immutable public actions and durable, lease-owned SQLite transitions."""

import importlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agent.state import ConversationState, PublicMessage
from app.errors import ShopError
from app.storage.sessions import SessionStore


def api():
    root = Path(__file__).resolve().parents[1]
    for relative in ("app/actions/__init__.py", "app/actions/schemas.py", "app/storage/actions.py"):
        assert (root / relative).is_file(), f"missing day6 module: {relative}"
    return importlib.import_module("app.actions.schemas"), importlib.import_module("app.storage.actions")


def card(schemas, sid, **updates):
    values = dict(
        kind="add_to_cart", action_id=uuid4().hex, session_id=sid,
        product_id=7, product_name="Phone", unique_id="sku-7", sku_name="Black",
        quantity=2, unit_price="19.99", total_price="39.98", image="phone.jpg",
        expires_at=time.time() + 900,
    )
    values.update(updates)
    return schemas.CartActionCard(**values)


def outcome(schemas, saved, status="succeeded", **updates):
    values = dict(action_id=saved.card.action_id, session_id=saved.card.session_id,
                  status=status, cart_id=23 if status == "succeeded" else None,
                  error_code=None if status in {"succeeded", "cancelled"} else "cart_result_unknown")
    values.update(updates)
    return schemas.CartActionResult(**values)


@contextmanager
def prepared(tmp_path):
    schemas, actions = api()
    store = actions.CartActionStore(tmp_path / "sessions.sqlite3")
    try:
        sid = store.create(11)
        lease = store.claim(sid, 11)
        state = ConversationState(history=[PublicMessage(role="user", content="Add phone")])
        saved = store.prepare(lease, 11, state, card(schemas, sid))
        yield schemas, actions, store, sid, saved
    finally:
        store.close()


def test_action_modules_exist():
    api()


def test_old_snapshot_has_no_pending_action():
    assert ConversationState.model_validate_json("{}").pending_action_id is None


def test_legacy_snapshot_at_exact_byte_limit_keeps_all_data():
    snapshot = ConversationState(
        candidates=[{"product_id": 7, "name": "phone", "description": "", "image": "", "skus": []}],
    ).model_dump(mode="json", exclude={"pending_action_id"})
    compact = json.dumps(snapshot, separators=(",", ":"))
    snapshot["candidates"][0]["description"] = "x" * (65536 - len(compact.encode("utf-8")))
    legacy = json.dumps(snapshot, separators=(",", ":"))
    assert len(legacy.encode("utf-8")) == 65536
    recovered = SessionStore._decode(legacy)
    assert recovered.pending_action_id is None
    assert recovered.candidates[0].description == snapshot["candidates"][0]["description"]
    assert len(SessionStore._encode(recovered).encode("utf-8")) == 65536
    snapshot["candidates"][0]["description"] += "x"
    with pytest.raises(ShopError) as captured:
        SessionStore._decode(json.dumps(snapshot, separators=(",", ":")))
    assert captured.value.code == "session_unavailable"


def test_exact_public_card_contract_is_frozen_and_has_correct_total():
    schemas, _ = api()
    saved = card(schemas, "a" * 32)
    assert set(saved.model_dump()) == {
        "kind", "action_id", "session_id", "product_id", "product_name", "unique_id",
        "sku_name", "quantity", "unit_price", "total_price", "image", "expires_at",
    }
    assert saved.total_price == saved.unit_price * saved.quantity == Decimal("39.98")
    with pytest.raises(ValidationError):
        saved.unit_price = Decimal("1")


@pytest.mark.parametrize("updates", [
    {"authorization": "hidden"}, {"user_id": 11}, {"nickname": "hidden"},
    {"quantity": True}, {"quantity": "2"}, {"quantity": 1.5},
    {"quantity": 0}, {"quantity": 10000}, {"product_id": True},
    {"product_id": 9223372036854775808}, {"action_id": "A" * 32},
    {"action_id": "a" * 32 + "\n"}, {"session_id": "a" * 31},
    {"unique_id": ""}, {"unique_id": "x" * 129}, {"product_name": "x" * 501},
    {"sku_name": "x" * 501}, {"image": "x" * 2049},
    {"unit_price": "NaN"}, {"unit_price": "-1"}, {"total_price": "39.97"},
    {"expires_at": float("inf")}, {"expires_at": True},
])
def test_card_rejects_private_unbounded_or_inconsistent_values(updates):
    schemas, _ = api()
    with pytest.raises(ValidationError):
        card(schemas, "a" * 32, **updates)


def test_decision_and_result_are_bounded_public_contracts():
    schemas, _ = api()
    assert schemas.ActionDecision(decision="confirm").decision == "confirm"
    for payload in ({"decision": "retry"}, {"decision": "confirm", "quantity": 9}):
        with pytest.raises(ValidationError):
            schemas.ActionDecision(**payload)
    for status in ("failed", "unknown", "cancelled", "executing"):
        with pytest.raises(ValidationError):
            schemas.CartActionResult(action_id="a" * 32, session_id="b" * 32,
                                     status=status, cart_id=23)
    with pytest.raises(ValidationError):
        schemas.CartActionResult(action_id="a" * 32, session_id="b" * 32, status="succeeded")
    with pytest.raises(ValidationError):
        schemas.CartActionResult(action_id="a" * 32, session_id="b" * 32,
                                 status="failed", error_code="x" * 65)
    with pytest.raises(ValidationError):
        schemas.CartActionResult(action_id="a" * 32, session_id="b" * 32,
                                 status="unknown", cart_path="https://private.invalid/")


def test_response_rejects_mismatched_action_and_session():
    schemas, _ = api()
    pending = card(schemas, "a" * 32)
    response = schemas.ActionResponse(session_id="a" * 32, status="awaiting_confirmation",
                                     answer="Confirm addition", pending_action=pending)
    assert response.pending_action == pending
    with pytest.raises(ValidationError):
        schemas.ActionResponse(session_id="b" * 32, status="awaiting_confirmation",
                               answer="Confirm addition", pending_action=pending)
    with pytest.raises(ValidationError):
        schemas.ActionResponse(session_id="a" * 32, status="succeeded",
                               answer="Done", pending_action=pending)


def test_prepare_reopens_public_snapshot_preserves_database_identity(tmp_path):
    with prepared(tmp_path) as (schemas, actions, store, sid, saved):
        assert saved.status == "pending" and saved.result is None
        assert set(vars(saved)) == {"card", "status", "result"}
        assert store.load(sid, 11).pending_action_id == saved.card.action_id
        next_lease = store.claim(sid, 11)
        assert next_lease.version == 1
        store.release(next_lease)
        with sqlite3.connect(store.path) as connection:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(sessions)")]
            assert columns == ["session_id", "user_id", "version", "state", "lease_id", "lease_expires"]
            assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x59533501
            encoded = connection.execute("SELECT state FROM sessions WHERE session_id = ?", (sid,)).fetchone()[0]
            action_json = connection.execute("SELECT card, result FROM cart_actions").fetchone()
        for public_json in (encoded, action_json[0]):
            assert not any(secret in public_json.lower() for secret in
                           ("authorization", "nickname", "user_id", "executor", "api_key"))
        store.close()
        reopened = actions.CartActionStore(store.path)
        try:
            assert reopened.get(saved.card.action_id, 11) == saved
            assert reopened.pending(sid, 11) == saved
            assert reopened.load(sid, 11).history[0].content == "Add phone"
        finally:
            reopened.close()


def test_foreign_actions_are_indistinguishable_from_absence(tmp_path):
    with prepared(tmp_path) as (_, _, store, _, saved):
        errors = []
        for supplied, owner in ((saved.card.action_id, 12), ("f" * 32, 11),
                                ("'; DROP TABLE cart_actions;--", 11), (saved.card.action_id, True)):
            with pytest.raises(ShopError) as captured:
                store.get(supplied, owner)
            errors.append((captured.value.code, captured.value.http_status, captured.value.message))
        assert errors[0] == errors[1] == errors[2] == errors[3]
        assert errors[0][:2] == ("action_not_found", 404)


def test_second_pending_action_is_rejected_without_committing_or_releasing_lease(tmp_path):
    with prepared(tmp_path) as (schemas, _, store, sid, saved):
        lease = store.claim(sid, 11)
        with pytest.raises(ShopError) as captured:
            store.prepare(lease, 11, ConversationState(), card(schemas, sid))
        assert captured.value.code == "action_conflict"
        assert store.pending(sid, 11) == saved
        assert store.load(sid, 11).history[0].content == "Add phone"
        with pytest.raises(ShopError) as captured:
            store.claim(sid, 11)
        assert captured.value.code == "session_busy"
        store.release(lease)


def test_two_connections_grant_exactly_one_execution(tmp_path):
    with prepared(tmp_path) as (_, actions, store, sid, saved):
        lease = store.claim(sid, 11)
        barrier = Barrier(2)

        def execute():
            competing = actions.CartActionStore(store.path)
            try:
                barrier.wait(timeout=5)
                try:
                    return competing.begin_execution(lease, 11, saved.card.action_id)
                except ShopError as error:
                    return error.code
            finally:
                competing.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: execute(), range(2)))
        assert sum(isinstance(result, actions.ActionRecord) for result in results) == 1
        assert results.count("action_conflict") == 1
        assert store.get(saved.card.action_id, 11).status == "executing"
        assert store.load(sid, 11).pending_action_id == saved.card.action_id
        with pytest.raises(ShopError) as captured:
            store.claim(sid, 11)
        assert captured.value.code == "session_busy"
        store.release(lease)


def test_execution_rejects_lease_expiring_after_initial_check_without_mutation(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    with prepared(tmp_path) as (_, _, store, sid, saved):
        lease = store.claim(sid, 11)
        with sqlite3.connect(store.path) as connection:
            before_session = connection.execute(
                "SELECT version, state, lease_id, lease_expires FROM sessions WHERE session_id = ?", (sid,),
            ).fetchone()
            before_action = connection.execute(
                "SELECT * FROM cart_actions WHERE action_id = ?", (saved.card.action_id,),
            ).fetchone()
        active = store._active

        def expire_after_initial_check(connection, session_id, user_id, session_row):
            record = active(connection, session_id, user_id, session_row)
            now[0] = before_session[3] + 1
            assert now[0] < saved.card.expires_at
            return record

        monkeypatch.setattr(store, "_active", expire_after_initial_check)
        with pytest.raises(ShopError) as captured:
            store.begin_execution(lease, 11, saved.card.action_id)
        assert (captured.value.code, captured.value.http_status) == ("session_conflict", 409)
        assert store.get(saved.card.action_id, 11) == saved
        assert store.load(sid, 11).pending_action_id == saved.card.action_id
        with sqlite3.connect(store.path) as connection:
            assert connection.execute(
                "SELECT version, state, lease_id, lease_expires FROM sessions WHERE session_id = ?", (sid,),
            ).fetchone() == before_session
            assert connection.execute(
                "SELECT * FROM cart_actions WHERE action_id = ?", (saved.card.action_id,),
            ).fetchone() == before_action


@pytest.mark.parametrize("status", ["succeeded", "failed", "unknown", "cancelled"])
def test_finish_commits_claimed_snapshot_releases_lease_and_replays_terminal(tmp_path, status):
    with prepared(tmp_path) as (schemas, _, store, sid, saved):
        lease = store.claim(sid, 11)
        lease.state.history.append(PublicMessage(role="assistant", content="Action finished"))
        if status != "cancelled":
            store.begin_execution(lease, 11, saved.card.action_id)
        result = outcome(schemas, saved, status)
        finished = store.finish(lease, 11, saved.card.action_id, result)
        assert finished.status == status and finished.result == result
        assert store.pending(sid, 11) is None
        assert store.load(sid, 11).pending_action_id is None
        assert store.load(sid, 11).history[-1].content == "Action finished"
        assert store.get(saved.card.action_id, 11) == finished
        current = store.claim(sid, 11)
        assert current.version == 2
        with pytest.raises(ShopError) as captured:
            store.begin_execution(current, 11, saved.card.action_id)
        assert captured.value.code == "action_conflict"
        assert store.get(saved.card.action_id, 11) == finished
        store.release(current)


def test_recovered_executing_action_can_only_finish_unknown(tmp_path, monkeypatch):
    with prepared(tmp_path) as (schemas, _, store, sid, saved):
        lease = store.claim(sid, 11)
        store.begin_execution(lease, 11, saved.card.action_id)
        sessions = importlib.import_module("app.storage.sessions")
        later = time.time() + sessions.LEASE_SECONDS + 1
        monkeypatch.setattr(sessions.time, "time", lambda: later)
        recovered = store.claim(sid, 11)
        with pytest.raises(ShopError) as captured:
            store.begin_execution(recovered, 11, saved.card.action_id)
        assert captured.value.code == "action_conflict"
        result = outcome(schemas, saved, "unknown")
        assert store.finish(recovered, 11, saved.card.action_id, result).result == result
        assert store.pending(sid, 11) is None


def test_replacement_cancels_immutable_old_card_and_requires_new_id(tmp_path):
    with prepared(tmp_path) as (schemas, _, store, sid, saved):
        lease = store.claim(sid, 11)
        fresh = card(schemas, sid, unit_price="22", total_price="44", sku_name="New Black")
        updated = ConversationState(history=[PublicMessage(role="assistant", content="Price changed")])
        replacement = store.replace(lease, 11, saved.card.action_id, updated, fresh)
        assert replacement.card == fresh and replacement.status == "pending"
        old = store.get(saved.card.action_id, 11)
        assert old.card == saved.card and old.status == "cancelled"
        assert old.result.status == "cancelled" and old.result.cart_id is None
        assert store.pending(sid, 11) == replacement
        assert store.load(sid, 11).pending_action_id == fresh.action_id
        assert store.load(sid, 11).history[0].content == "Price changed"
        current = store.claim(sid, 11)
        assert current.version == 2
        duplicate = fresh.model_copy(update={"unit_price": Decimal("1"), "total_price": Decimal("2")})
        with pytest.raises(ShopError) as captured:
            store.replace(current, 11, fresh.action_id, ConversationState(), duplicate)
        assert captured.value.code == "action_conflict"
        assert store.pending(sid, 11) == replacement
        store.release(current)


@pytest.mark.parametrize("operation", ["prepare", "begin_execution", "finish", "replace"])
@pytest.mark.parametrize("invalid", ["expired", "version", "lease_id", "foreign"])
def test_every_action_write_checks_lease_and_ownership(tmp_path, monkeypatch, operation, invalid):
    with prepared(tmp_path) as (schemas, _, store, sid, saved):
        lease = store.claim(sid, 11)
        if operation == "finish":
            store.begin_execution(lease, 11, saved.card.action_id)
        before = store.get(saved.card.action_id, 11)
        supplied = lease
        owner = 11
        if invalid == "expired":
            sessions = importlib.import_module("app.storage.sessions")
            later = time.time() + sessions.LEASE_SECONDS + 1
            monkeypatch.setattr(sessions.time, "time", lambda: later)
        elif invalid == "version":
            supplied = replace(lease, version=lease.version + 1)
        elif invalid == "lease_id":
            supplied = replace(lease, lease_id="f" * 32)
        else:
            owner = 12
        with pytest.raises(ShopError) as captured:
            if operation == "prepare":
                store.prepare(supplied, owner, ConversationState(), card(schemas, sid))
            elif operation == "begin_execution":
                store.begin_execution(supplied, owner, saved.card.action_id)
            elif operation == "finish":
                store.finish(supplied, owner, saved.card.action_id, outcome(schemas, saved))
            else:
                store.replace(supplied, owner, saved.card.action_id, ConversationState(), card(schemas, sid))
        assert captured.value.code == (
            ("session_not_found" if operation == "prepare" else "action_not_found")
            if invalid == "foreign" else "session_conflict"
        )
        assert store.get(saved.card.action_id, 11) == before
        assert store.load(sid, 11).pending_action_id == saved.card.action_id
        store.release(lease)


def test_invalid_card_and_snapshot_roll_back_prepare(tmp_path):
    schemas, actions = api()
    store = actions.CartActionStore(tmp_path / "sessions.sqlite3")
    try:
        sid = store.create(11)
        lease = store.claim(sid, 11)
        malformed = card(schemas, sid).model_copy(update={"total_price": Decimal("1")})
        with pytest.raises(ShopError):
            store.prepare(lease, 11, ConversationState(), malformed)
        assert store.pending(sid, 11) is None
        assert store.load(sid, 11) == ConversationState()
        oversized = ConversationState()
        oversized.history.extend([PublicMessage(role="user", content="x")] * 13)
        with pytest.raises(ShopError):
            store.prepare(lease, 11, oversized, card(schemas, sid))
        assert store.pending(sid, 11) is None
        store.release(lease)
    finally:
        store.close()


def test_invalid_outcome_rolls_back_executing_state_and_keeps_lease(tmp_path):
    with prepared(tmp_path) as (schemas, _, store, sid, saved):
        lease = store.claim(sid, 11)
        store.begin_execution(lease, 11, saved.card.action_id)
        for malformed in (
            outcome(schemas, saved).model_copy(update={"cart_id": None}),
            outcome(schemas, saved).model_copy(update={"action_id": "f" * 32}),
            outcome(schemas, saved, "unknown").model_copy(update={"cart_id": 23}),
            outcome(schemas, saved).model_copy(update={"status": "executing"}),
        ):
            with pytest.raises(ShopError):
                store.finish(lease, 11, saved.card.action_id, malformed)
            assert store.get(saved.card.action_id, 11).status == "executing"
            assert store.load(sid, 11).pending_action_id == saved.card.action_id
        store.finish(lease, 11, saved.card.action_id, outcome(schemas, saved, "unknown"))


def test_sql_failure_rolls_back_replacement_and_snapshot(tmp_path):
    with prepared(tmp_path) as (schemas, _, store, sid, saved):
        lease = store.claim(sid, 11)
        with sqlite3.connect(store.path) as connection:
            connection.execute("""CREATE TRIGGER block_action_insert BEFORE INSERT ON cart_actions
                BEGIN SELECT RAISE(ABORT, 'blocked'); END""")
        with pytest.raises(ShopError) as captured:
            store.replace(lease, 11, saved.card.action_id, ConversationState(), card(schemas, sid))
        assert captured.value.code == "session_unavailable"
        assert store.get(saved.card.action_id, 11) == saved
        assert store.load(sid, 11).history[0].content == "Add phone"
        with pytest.raises(ShopError) as captured:
            store.claim(sid, 11)
        assert captured.value.code == "session_busy"
        store.release(lease)


@pytest.mark.parametrize("corrupt", ["card", "result", "status", "pointer"])
def test_corrupt_persisted_actions_fail_closed_without_execution(tmp_path, corrupt):
    with prepared(tmp_path) as (_, _, store, sid, saved):
        with sqlite3.connect(store.path) as connection:
            if corrupt == "card":
                connection.execute("UPDATE cart_actions SET card = ?", ('{"authorization":"private"}',))
            elif corrupt == "result":
                connection.execute("UPDATE cart_actions SET result = ?", ('{"status":"succeeded"}',))
            elif corrupt == "status":
                connection.execute("UPDATE cart_actions SET status = ?", ("succeeded",))
            else:
                snapshot = json.loads(connection.execute("SELECT state FROM sessions").fetchone()[0])
                snapshot["pending_action_id"] = "f" * 32
                connection.execute("UPDATE sessions SET state = ?", (json.dumps(snapshot),))
        with pytest.raises(ShopError) as captured:
            store.pending(sid, 11)
        assert captured.value.code == "session_unavailable"
        assert "private" not in str(captured.value)
        lease = store.claim(sid, 11)
        with pytest.raises(ShopError):
            store.begin_execution(lease, 11, saved.card.action_id)
        store.release(lease)


def test_action_store_adds_table_to_existing_session_database(tmp_path):
    _, actions = api()
    original = SessionStore(tmp_path / "sessions.sqlite3")
    sid = original.create(11)
    original.close()
    upgraded = actions.CartActionStore(tmp_path / "sessions.sqlite3")
    try:
        assert upgraded.load(sid, 11) == ConversationState()
        assert upgraded.pending(sid, 11) is None
    finally:
        upgraded.close()
