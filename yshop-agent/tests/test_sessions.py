"""Durable completed-turn snapshots and atomic ownership/lease boundaries."""

import importlib
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from app.errors import ShopError
from app.schemas import ProductDetail, ProductSearchInput


def test_session_modules_exist():
    root = Path(__file__).resolve().parents[1]
    for relative in ("app/storage/__init__.py", "app/storage/sessions.py", "app/agent/state.py"):
        assert (root / relative).is_file(), f"missing day5 module: {relative}"


def test_default_lease_covers_maximum_bounded_model_rounds(api):
    assert api[0].LEASE_SECONDS == 600


@pytest.fixture
def api():
    return importlib.import_module("app.storage.sessions"), importlib.import_module("app.agent.state")


@pytest.fixture
def store(api, tmp_path):
    instance = api[0].SessionStore(tmp_path / "sessions.sqlite3")
    yield instance
    instance.close()


def sample_state(api):
    return api[1].ConversationState(
        history=[{"role": "user", "content": "一千元以内的手机"},
                 {"role": "assistant", "content": "找到这款手机。"}],
        conditions=ProductSearchInput(keyword="手机", max_price="1000", limit=3),
        candidates=[ProductDetail(product_id=7, name="手机", description="公开介绍", image="phone.jpg",
                                  skus=[{"name": "黑色", "unique_id": "sku-7", "price": "999.50",
                                         "stock": 4, "image": "phone.jpg"}])],
        selected_product_id=7, selected_sku_id="sku-7",
    )


def test_commit_reopens_completed_snapshot_without_owner_in_state(api, tmp_path):
    path = tmp_path / "sessions.sqlite3"
    first = api[0].SessionStore(path)
    try:
        session_id = first.create(43)
        assert re.fullmatch(r"[0-9a-f]{32}", session_id)
        lease = first.claim(session_id, 43)
        assert lease.version == 0
        assert set(vars(lease)) == {"session_id", "lease_id", "version", "state"}
        assert lease.state == api[1].ConversationState()
        first.commit(lease, sample_state(api))
    finally:
        first.close()
    second = api[0].SessionStore(path)
    try:
        assert second.load(session_id, 43) == sample_state(api)
        next_lease = second.claim(session_id, 43)
        assert next_lease.version == 1
        assert next_lease.lease_id != lease.lease_id
        second.release(next_lease)
        with sqlite3.connect(path) as connection:
            snapshot = connection.execute("SELECT state FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
        assert '"user_id"' not in snapshot
        assert '"authorization"' not in snapshot.lower()
        assert '"nickname"' not in snapshot.lower()
    finally:
        second.close()


@pytest.mark.parametrize("method", ["load", "claim"])
def test_other_owner_and_unknown_session_have_same_not_found_error(store, method):
    session_id = store.create(11)
    errors = []
    for supplied, owner in ((session_id, 12), ("f" * 32, 11), ("'; DROP TABLE sessions;--", 11)):
        with pytest.raises(ShopError) as captured:
            getattr(store, method)(supplied, owner)
        errors.append((captured.value.code, captured.value.http_status, captured.value.message))
    assert errors[0] == errors[1] == errors[2]
    assert errors[0][:2] == ("session_not_found", 404)
    assert store.load(session_id, 11).history == []


def test_active_lease_blocks_claim_on_another_connection(api, store, tmp_path):
    other = api[0].SessionStore(tmp_path / "sessions.sqlite3")
    try:
        session_id = store.create(11)
        lease = store.claim(session_id, 11)
        with pytest.raises(ShopError) as captured:
            other.claim(session_id, 11)
        assert (captured.value.code, captured.value.http_status) == ("session_busy", 409)
        store.release(lease)
        new_lease = other.claim(session_id, 11)
        other.release(new_lease)
    finally:
        other.close()


def test_simultaneous_claims_have_one_winner(api, store, tmp_path):
    session_id = store.create(11)
    barrier = Barrier(2)

    def acquire():
        competing = api[0].SessionStore(tmp_path / "sessions.sqlite3")
        try:
            barrier.wait(timeout=5)
            try:
                return competing.claim(session_id, 11)
            except ShopError as error:
                return error.code
        finally:
            competing.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: acquire(), range(2)))
    assert sum(isinstance(result, api[0].SessionLease) for result in results) == 1
    assert results.count("session_busy") == 1
    store.release(next(result for result in results if isinstance(result, api[0].SessionLease)))


def test_expired_lease_cannot_overwrite_or_release_new_lease(api, store, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(api[0].time, "time", lambda: now[0])
    session_id = store.create(11)
    stale = store.claim(session_id, 11)
    now[0] += api[0].LEASE_SECONDS + 1
    current = store.claim(session_id, 11)
    store.release(stale)
    with pytest.raises(ShopError) as captured:
        store.commit(stale, sample_state(api))
    assert (captured.value.code, captured.value.http_status) == ("session_conflict", 409)
    with pytest.raises(ShopError) as captured:
        store.claim(session_id, 11)
    assert captured.value.code == "session_busy"
    store.commit(current, api[1].ConversationState(history=[{"role": "user", "content": "新轮次"}]))
    with pytest.raises(ShopError) as captured:
        store.commit(stale, sample_state(api))
    assert captured.value.code == "session_conflict"
    assert store.load(session_id, 11).history[0].content == "新轮次"


def test_expired_lease_cannot_commit_even_without_replacement(api, store, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(api[0].time, "time", lambda: now[0])
    session_id = store.create(11)
    lease = store.claim(session_id, 11)
    now[0] += api[0].LEASE_SECONDS + 1
    with pytest.raises(ShopError) as captured:
        store.commit(lease, sample_state(api))
    assert captured.value.code == "session_conflict"
    assert store.load(session_id, 11) == api[1].ConversationState()


def test_exception_release_keeps_completed_state_and_is_idempotent(api, store):
    session_id = store.create(11)
    lease = store.claim(session_id, 11)
    store.commit(lease, sample_state(api))
    failed = store.claim(session_id, 11)
    try:
        failed.state.history.append(api[1].PublicMessage(role="user", content="未完成问题"))
        raise RuntimeError("model failed")
    except RuntimeError:
        store.release(failed)
    store.release(failed)
    assert store.load(session_id, 11) == sample_state(api)
    next_lease = store.claim(session_id, 11)
    store.release(next_lease)


@pytest.mark.parametrize("unsafe", [
    {"authorization": "Bearer hidden"}, {"user_id": 11}, {"nickname": "private"},
    {"history": [{"role": "tool", "content": "unfinished tool result"}]},
    {"history": [{"role": "assistant", "content": "ok", "tool_calls": []}]},
    {"history": [{"role": "user", "content": "x" * 4001}]},
    {"history": [{"role": "user", "content": "x"}] * 13},
    {"selected_product_id": True}, {"selected_sku_id": ""},
])
def test_state_rejects_private_unfinished_or_unbounded_fields(api, unsafe):
    with pytest.raises(ValidationError):
        api[1].ConversationState.model_validate(unsafe)


def test_state_caps_candidates_and_total_snapshot(api):
    candidate = sample_state(api).candidates[0]
    with pytest.raises(ValidationError):
        api[1].ConversationState(candidates=[candidate] * 4)
    oversized = candidate.model_copy(update={"description": "x" * 65536})
    with pytest.raises(ValidationError):
        api[1].ConversationState(candidates=[oversized])


def test_commit_revalidates_mutated_state_before_sql_write(api, store):
    session_id = store.create(11)
    lease = store.claim(session_id, 11)
    unsafe = sample_state(api)
    unsafe.history.extend([api[1].PublicMessage(role="user", content="x")] * 12)
    with pytest.raises(ShopError) as captured:
        store.commit(lease, unsafe)
    assert captured.value.code == "session_unavailable"
    assert store.load(session_id, 11) == api[1].ConversationState()
    store.release(lease)


def test_corrupt_snapshot_and_closed_store_raise_sanitized_unavailable(api, store, tmp_path):
    session_id = store.create(11)
    with sqlite3.connect(tmp_path / "sessions.sqlite3") as connection:
        connection.execute("UPDATE sessions SET state = ? WHERE session_id = ?", ('{"authorization":"private"}', session_id))
    with pytest.raises(ShopError) as captured:
        store.load(session_id, 11)
    assert (captured.value.code, captured.value.http_status) == ("session_unavailable", 503)
    assert "private" not in str(captured.value)
    store.close()
    store.close()
    with pytest.raises(ShopError) as captured:
        store.create(11)
    assert captured.value.code == "session_unavailable"


def test_cannot_overwrite_existing_non_session_file(api, tmp_path):
    path = tmp_path / "sessions.sqlite3"
    original = b"protected configuration, not a database"
    path.write_bytes(original)
    with pytest.raises(ShopError) as captured:
        api[0].SessionStore(path)
    assert captured.value.code == "session_unavailable"
    assert path.read_bytes() == original
