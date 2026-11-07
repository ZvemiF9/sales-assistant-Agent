"""SQLite snapshots with atomic ownership checks and expiring request leases."""

import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from app.agent.state import MAX_SNAPSHOT_BYTES, ConversationState
from app.errors import ShopError


# Five bounded model rounds plus sequential read tools fit within this lease.
LEASE_SECONDS = 600
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "day5-sessions.sqlite3"
_APPLICATION_ID = 0x59533501
_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_COLUMNS = ["session_id", "user_id", "version", "state", "lease_id", "lease_expires"]


@dataclass(frozen=True)
class SessionLease:
    session_id: str
    lease_id: str
    version: int
    state: ConversationState


class SessionStore:
    """The caller validates configured paths; explicit paths support test databases.

    No request identity, executor, Authorization, or model credential enters state.
    Every lease operation is a short transaction; model work happens outside it.
    """

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path).resolve()
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()
        try:
            if self.path.exists() and self.path.stat().st_size:
                # Refuse foreign databases before opening any writable connection.
                existing = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
                try:
                    application_id = existing.execute("PRAGMA application_id").fetchone()[0]
                    columns = [row[1] for row in existing.execute("PRAGMA table_info(sessions)")]
                    if application_id != _APPLICATION_ID or columns != _COLUMNS:
                        raise ShopError("session_unavailable")
                finally:
                    existing.close()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False,
                                              timeout=5)
            self._connection.execute("PRAGMA synchronous = FULL")
            with self._transaction() as connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    lease_id TEXT,
                    lease_expires REAL
                )""")
                connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        except (OSError, sqlite3.Error):
            self.close()
            raise ShopError("session_unavailable") from None
        except ShopError:
            self.close()
            raise

    @contextmanager
    def _transaction(self):
        with self._lock:
            connection = self._connection
            if connection is None:
                raise ShopError("session_unavailable")
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.execute("COMMIT")
            except BaseException as error:
                if connection.in_transaction:
                    connection.rollback()
                if isinstance(error, sqlite3.Error):
                    raise ShopError("session_unavailable") from None
                raise

    @staticmethod
    def _valid_user(user_id: int) -> bool:
        return type(user_id) is int and 0 < user_id <= 9223372036854775807

    @staticmethod
    def _decode(snapshot: str) -> ConversationState:
        try:
            if not isinstance(snapshot, str) or len(snapshot.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
                raise ValueError("invalid snapshot")
            return ConversationState.model_validate_json(snapshot)
        except (ValueError, ValidationError):
            raise ShopError("session_unavailable") from None

    @classmethod
    def _encode(cls, state: ConversationState) -> str:
        try:
            if not isinstance(state, ConversationState):
                raise ValueError("invalid snapshot")
            # Revalidate after serialization: callers may mutate lists or nested models.
            validated = cls._decode(state.model_dump_json(warnings="error"))
            return validated.model_dump_json()
        except (ValueError, ValidationError, PydanticSerializationError):
            raise ShopError("session_unavailable") from None

    def _owned(self, connection: sqlite3.Connection, session_id: str, user_id: int):
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id) or not self._valid_user(user_id):
            raise ShopError("session_not_found")
        row = connection.execute(
            "SELECT version, state, lease_id, lease_expires FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if row is None:
            raise ShopError("session_not_found")
        return row

    def create(self, user_id: int) -> str:
        if not self._valid_user(user_id):
            raise ShopError("invalid_request")
        session_id = uuid4().hex
        with self._transaction() as connection:
            connection.execute("INSERT INTO sessions (session_id, user_id, state) VALUES (?, ?, ?)",
                               (session_id, user_id, self._encode(ConversationState())))
        return session_id

    def load(self, session_id: str, user_id: int) -> ConversationState:
        with self._transaction() as connection:
            row = self._owned(connection, session_id, user_id)
            return self._decode(row[1])

    def claim(self, session_id: str, user_id: int) -> SessionLease:
        with self._transaction() as connection:
            version, snapshot, lease_id, lease_expires = self._owned(connection, session_id, user_id)
            now = time.time()
            if lease_id is not None and lease_expires is not None and lease_expires > now:
                raise ShopError("session_busy")
            state = self._decode(snapshot)
            new_lease_id = uuid4().hex
            connection.execute("UPDATE sessions SET lease_id = ?, lease_expires = ? WHERE session_id = ? AND version = ?",
                               (new_lease_id, now + LEASE_SECONDS, session_id, version))
            return SessionLease(session_id, new_lease_id, version, state)

    def commit(self, lease: SessionLease, state: ConversationState) -> None:
        snapshot = self._encode(state)
        with self._transaction() as connection:
            self._commit_snapshot(connection, lease, snapshot)

    @staticmethod
    def _assert_lease(lease: SessionLease, row) -> None:
        if (not isinstance(lease, SessionLease) or type(lease.version) is not int
                or lease.version < 0 or not isinstance(lease.lease_id, str)
                or not _SESSION_ID.fullmatch(lease.lease_id)
                or row[0] != lease.version or row[2] != lease.lease_id
                or not isinstance(row[3], (int, float)) or row[3] <= time.time()):
            raise ShopError("session_conflict")

    @staticmethod
    def _commit_snapshot(connection: sqlite3.Connection, lease: SessionLease, snapshot: str) -> None:
        """Commit inside the caller's transaction, including action-table updates."""
        result = connection.execute("""UPDATE sessions
            SET state = ?, version = version + 1, lease_id = NULL, lease_expires = NULL
            WHERE session_id = ? AND lease_id = ? AND version = ? AND lease_expires > ?""",
            (snapshot, lease.session_id, lease.lease_id, lease.version, time.time()))
        if result.rowcount != 1:
            raise ShopError("session_conflict")

    def release(self, lease: SessionLease) -> None:
        with self._transaction() as connection:
            connection.execute("""UPDATE sessions SET lease_id = NULL, lease_expires = NULL
                WHERE session_id = ? AND lease_id = ? AND version = ?""",
                (lease.session_id, lease.lease_id, lease.version))

    def close(self) -> None:
        with self._lock:
            connection, self._connection = self._connection, None
            if connection is not None:
                connection.close()
