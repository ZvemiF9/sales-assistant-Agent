"""Atomic action transitions sharing the durable, owned session database."""

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from app.actions.schemas import ACTION_TTL_SECONDS, CartActionCard, CartActionResult
from app.actions.history import ActionEvent, ActionHistory
from app.agent.state import ConversationState
from app.errors import ShopError
from app.storage.sessions import DEFAULT_PATH, SessionLease, SessionStore
from app.telemetry import request_id


_ACTION_ID = re.compile(r"[0-9a-f]{32}\Z")
_ACTION_COLUMNS = ["action_id", "session_id", "user_id", "status", "card", "result", "execution_lease_id"]
_TERMINAL = {"succeeded", "failed", "unknown", "cancelled"}
_MAX_CARD_BYTES = 16384
_MAX_RESULT_BYTES = 2048
_EVENT_COLUMNS = ["event_id", "action_id", "sequence", "kind", "status", "occurred_at",
                  "request_id", "related_action_id", "error_code"]
_EVENT_LAYOUT = [
    ("event_id", "INTEGER", 0, None, 1, 0),
    ("action_id", "TEXT", 1, None, 0, 0),
    ("sequence", "INTEGER", 1, None, 0, 0),
    ("kind", "TEXT", 1, None, 0, 0),
    ("status", "TEXT", 1, None, 0, 0),
    ("occurred_at", "REAL", 1, None, 0, 0),
    ("request_id", "TEXT", 0, None, 0, 0),
    ("related_action_id", "TEXT", 0, None, 0, 0),
    ("error_code", "TEXT", 0, None, 0, 0),
]


@dataclass(frozen=True)
class ActionRecord:
    card: CartActionCard
    status: str
    result: CartActionResult | None


class CartActionStore(SessionStore):
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        super().__init__(path)
        try:
            with self._transaction() as connection:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(cart_actions)")]
                if columns and columns != _ACTION_COLUMNS:
                    raise ShopError("session_unavailable")
                connection.execute("""CREATE TABLE IF NOT EXISTS cart_actions (
                    action_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    card TEXT NOT NULL,
                    result TEXT,
                    execution_lease_id TEXT
                )""")
                connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_active_cart_action
                    ON cart_actions(session_id) WHERE status IN ('pending', 'executing')""")
                self._assert_event_schema(connection)
                connection.execute("""CREATE TABLE IF NOT EXISTS cart_action_events (
                    event_id INTEGER PRIMARY KEY,
                    action_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    occurred_at REAL NOT NULL,
                    request_id TEXT,
                    related_action_id TEXT,
                    error_code TEXT,
                    UNIQUE(action_id, sequence)
                )""")
                rows = connection.execute("""SELECT action_id, session_id, user_id, status, card, result,
                    execution_lease_id FROM cart_actions
                    WHERE NOT EXISTS (SELECT 1 FROM cart_action_events e
                                      WHERE e.action_id = cart_actions.action_id)""").fetchall()
                for row in rows:
                    record = self._owned_action(connection, row[0], row[2])[1]
                    self._append_event(connection, record.card.action_id, "legacy", record.status)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _assert_event_schema(connection):
        columns = [tuple(row[1:]) for row in connection.execute("PRAGMA table_xinfo(cart_action_events)")]
        if not columns:
            return
        indexes = connection.execute(
            """SELECT name, partial FROM pragma_index_list('cart_action_events') WHERE "unique" = 1"""
        ).fetchall()
        if columns != _EVENT_LAYOUT or len(indexes) != 1 or indexes[0][1] != 0:
            raise ShopError("session_unavailable")
        keys = [row[0] for row in connection.execute(
            "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (indexes[0][0],))]
        if keys != ["action_id", "sequence"]:
            raise ShopError("session_unavailable")

    @staticmethod
    def _public_json(model, value, limit: int):
        try:
            if not isinstance(value, model):
                raise ValueError("invalid public action")
            snapshot = value.model_dump_json(warnings="error")
            return CartActionStore._decode_public(model, snapshot, limit)
        except (ValueError, ValidationError, PydanticSerializationError):
            raise ShopError("session_unavailable") from None

    @staticmethod
    def _decode_public(model, snapshot, limit: int):
        try:
            if not isinstance(snapshot, str) or len(snapshot.encode("utf-8")) > limit:
                raise ValueError("invalid public action")
            return model.model_validate_json(snapshot)
        except (ValueError, ValidationError):
            raise ShopError("session_unavailable") from None

    @classmethod
    def _record(cls, row) -> ActionRecord:
        action_id, session_id, user_id, status, card_json, result_json, execution_lease_id = row
        card = cls._decode_public(CartActionCard, card_json, _MAX_CARD_BYTES)
        result = (None if result_json is None
                  else cls._decode_public(CartActionResult, result_json, _MAX_RESULT_BYTES))
        valid = card.action_id == action_id and card.session_id == session_id and cls._valid_user(user_id)
        if status in {"pending", "executing"}:
            valid = valid and result is None
            valid = valid and (
                execution_lease_id is None if status == "pending"
                else isinstance(execution_lease_id, str) and bool(_ACTION_ID.fullmatch(execution_lease_id))
            )
        elif status in _TERMINAL:
            valid = valid and result is not None and result.status == status
        else:
            valid = False
        if result is not None:
            valid = valid and result.action_id == action_id and result.session_id == session_id
        if not valid:
            raise ShopError("session_unavailable")
        return ActionRecord(card, status, result)

    def _owned_action(self, connection: sqlite3.Connection, action_id: str, user_id: int):
        if not isinstance(action_id, str) or not _ACTION_ID.fullmatch(action_id) or not self._valid_user(user_id):
            raise ShopError("action_not_found")
        row = connection.execute("""SELECT a.action_id, a.session_id, a.user_id, a.status,
            a.card, a.result, a.execution_lease_id FROM cart_actions a
            JOIN sessions s ON s.session_id = a.session_id AND s.user_id = a.user_id
            WHERE a.action_id = ? AND a.user_id = ?""", (action_id, user_id)).fetchone()
        if row is None:
            raise ShopError("action_not_found")
        return row, self._record(row)

    def _active(self, connection: sqlite3.Connection, session_id: str, user_id: int, session_row):
        state = self._decode(session_row[1])
        rows = connection.execute("""SELECT action_id, session_id, user_id, status, card, result,
            execution_lease_id FROM cart_actions
            WHERE session_id = ? AND status IN ('pending', 'executing')""", (session_id,)).fetchall()
        if len(rows) > 1:
            raise ShopError("session_unavailable")
        active = self._record(rows[0]) if rows else None
        if (active is not None and rows[0][2] != user_id
                or state.pending_action_id != (active.card.action_id if active is not None else None)):
            raise ShopError("session_unavailable")
        return active

    def _action_lease(self, connection, lease, user_id, action_id):
        action_row, record = self._owned_action(connection, action_id, user_id)
        if record.card.session_id != lease.session_id:
            raise ShopError("action_not_found")
        session_row = self._owned(connection, lease.session_id, user_id)
        self._assert_lease(lease, session_row)
        self._active(connection, lease.session_id, user_id, session_row)
        return action_row, record

    @classmethod
    def _snapshot_with_pending(cls, state: ConversationState, action_id: str | None) -> str:
        validated = cls._decode(cls._encode(state))
        return cls._encode(validated.model_copy(update={"pending_action_id": action_id}))

    def _new_card(self, card, session_id):
        validated = self._public_json(CartActionCard, card, _MAX_CARD_BYTES)
        if validated.session_id != session_id:
            raise ShopError("action_conflict")
        remaining = validated.expires_at - time.time()
        if not 0 < remaining <= ACTION_TTL_SECONDS:
            raise ShopError("action_expired")
        return validated

    @staticmethod
    def _insert(connection, user_id, card):
        if connection.execute("SELECT 1 FROM cart_actions WHERE action_id = ?", (card.action_id,)).fetchone():
            raise ShopError("action_conflict")
        connection.execute("""INSERT INTO cart_actions (action_id, session_id, user_id, status, card)
            VALUES (?, ?, ?, 'pending', ?)""", (card.action_id, card.session_id, user_id, card.model_dump_json()))

    def prepare(self, lease: SessionLease, user_id: int, state: ConversationState,
                card: CartActionCard) -> ActionRecord:
        with self._transaction() as connection:
            session_row = self._owned(connection, lease.session_id, user_id)
            self._assert_lease(lease, session_row)
            if self._active(connection, lease.session_id, user_id, session_row) is not None:
                raise ShopError("action_conflict")
            validated = self._new_card(card, lease.session_id)
            snapshot = self._snapshot_with_pending(state, validated.action_id)
            self._insert(connection, user_id, validated)
            self._append_event(connection, validated.action_id, "prepared", "pending")
            self._commit_snapshot(connection, lease, snapshot)
            return ActionRecord(validated, "pending", None)

    def get(self, action_id: str, user_id: int) -> ActionRecord:
        with self._transaction() as connection:
            return self._owned_action(connection, action_id, user_id)[1]

    @staticmethod
    def _events(connection, action_id):
        rows = connection.execute("""SELECT sequence, kind, status, occurred_at,
            request_id, related_action_id, error_code FROM cart_action_events
            WHERE action_id = ? ORDER BY sequence LIMIT 4""", (action_id,)).fetchall()
        try:
            return [ActionEvent(**dict(zip(_EVENT_COLUMNS[2:], row))) for row in rows]
        except (ValueError, ValidationError):
            raise ShopError("session_unavailable") from None

    @classmethod
    def _append_event(cls, connection, action_id, kind, status, *,
                      related_action_id=None, error_code=None):
        events = cls._events(connection, action_id)
        try:
            event = ActionEvent(sequence=len(events) + 1, kind=kind, status=status,
                                occurred_at=time.time(), related_action_id=related_action_id,
                                error_code=error_code, request_id=None if kind == "legacy" else request_id())
            session_id = connection.execute(
                "SELECT session_id FROM cart_actions WHERE action_id = ?", (action_id,)).fetchone()[0]
            ActionHistory(action_id=action_id, session_id=session_id, events=[*events, event])
        except (ValueError, ValidationError):
            raise ShopError("session_unavailable") from None
        connection.execute("""INSERT INTO cart_action_events
            (action_id, sequence, kind, status, occurred_at, request_id, related_action_id, error_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (action_id, event.sequence, event.kind, event.status, event.occurred_at, event.request_id,
             event.related_action_id, event.error_code))

    def history(self, action_id: str, user_id: int) -> ActionHistory:
        with self._transaction() as connection:
            record = self._owned_action(connection, action_id, user_id)[1]
            try:
                history = ActionHistory(action_id=action_id, session_id=record.card.session_id,
                                        events=self._events(connection, action_id))
                if history.events[-1].status != record.status:
                    raise ValueError("history disagrees with action")
                for event in history.events:
                    if event.related_action_id is not None:
                        related = self._owned_action(connection, event.related_action_id, user_id)[1]
                        if related.card.session_id != history.session_id:
                            raise ValueError("foreign related action")
                return history
            except (ValueError, ValidationError, ShopError):
                raise ShopError("session_unavailable") from None

    def pending(self, session_id: str, user_id: int) -> ActionRecord | None:
        with self._transaction() as connection:
            row = self._owned(connection, session_id, user_id)
            return self._active(connection, session_id, user_id, row)

    def begin_execution(self, lease: SessionLease, user_id: int, action_id: str) -> ActionRecord:
        with self._transaction() as connection:
            _, record = self._action_lease(connection, lease, user_id, action_id)
            if record.status != "pending":
                raise ShopError("action_conflict")
            if record.card.expires_at <= time.time():
                raise ShopError("action_expired")
            changed = connection.execute("""UPDATE cart_actions SET status = 'executing', execution_lease_id = ?
                WHERE action_id = ? AND session_id = ? AND user_id = ? AND status = 'pending'
                AND EXISTS (SELECT 1 FROM sessions s
                    WHERE s.session_id = cart_actions.session_id AND s.user_id = cart_actions.user_id
                    AND s.lease_id = ? AND s.version = ? AND s.lease_expires > ?)""",
                (lease.lease_id, action_id, lease.session_id, user_id,
                 lease.lease_id, lease.version, time.time()))
            if changed.rowcount != 1:
                raise ShopError("session_conflict")
            self._append_event(connection, action_id, "executing", "executing")
            return ActionRecord(record.card, "executing", None)

    def finish(self, lease: SessionLease, user_id: int, action_id: str,
               result: CartActionResult) -> ActionRecord:
        with self._transaction() as connection:
            row, record = self._action_lease(connection, lease, user_id, action_id)
            validated = self._public_json(CartActionResult, result, _MAX_RESULT_BYTES)
            if (validated.action_id != action_id or validated.session_id != lease.session_id
                    or validated.status not in _TERMINAL or record.status not in {"pending", "executing"}
                    or record.status == "pending" and validated.status not in {"failed", "cancelled"}
                    or record.status == "executing" and validated.status == "cancelled"
                    or record.status == "executing" and row[6] != lease.lease_id and validated.status != "unknown"
                    or lease.state.pending_action_id != action_id):
                raise ShopError("action_conflict")
            snapshot = self._snapshot_with_pending(lease.state, None)
            connection.execute("UPDATE cart_actions SET status = ?, result = ? WHERE action_id = ?",
                               (validated.status, validated.model_dump_json(), action_id))
            self._append_event(connection, action_id, "finished", validated.status,
                               error_code=validated.error_code)
            self._commit_snapshot(connection, lease, snapshot)
            return ActionRecord(record.card, validated.status, validated)

    def replace(self, lease: SessionLease, user_id: int, action_id: str,
                state: ConversationState, card: CartActionCard) -> ActionRecord:
        with self._transaction() as connection:
            _, record = self._action_lease(connection, lease, user_id, action_id)
            validated = self._new_card(card, lease.session_id)
            if record.status != "pending" or validated.action_id == action_id:
                raise ShopError("action_conflict")
            snapshot = self._snapshot_with_pending(state, validated.action_id)
            cancelled = CartActionResult(action_id=action_id, session_id=lease.session_id, status="cancelled")
            connection.execute("UPDATE cart_actions SET status = 'cancelled', result = ? WHERE action_id = ?",
                               (cancelled.model_dump_json(), action_id))
            self._insert(connection, user_id, validated)
            self._append_event(connection, action_id, "replaced", "cancelled",
                               related_action_id=validated.action_id)
            self._append_event(connection, validated.action_id, "prepared", "pending",
                               related_action_id=action_id)
            self._commit_snapshot(connection, lease, snapshot)
            return ActionRecord(validated, "pending", None)
