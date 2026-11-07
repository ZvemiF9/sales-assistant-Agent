"""Guarded official async SQLite persistence dedicated to cart interrupts."""

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from starlette.concurrency import run_in_threadpool

from app.errors import ShopError
from app.storage.config import SERVICE_ROOT


APPLICATION_ID = 0x59533602
CHECKPOINT_NAME = "cart-checkpoints.sqlite3"
_TABLES = {"checkpoints", "writes"}


def validate_checkpoint_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve()
    if (not resolved.is_relative_to(SERVICE_ROOT)
            or resolved.relative_to(SERVICE_ROOT).parts[0] not in {"data", ".tmp"}
            or resolved.name != CHECKPOINT_NAME
            or "chroma" in resolved.relative_to(SERVICE_ROOT).parts
            or any(part.is_symlink() or part.is_junction() for part in (path, *path.parents))):
        raise ShopError("session_unavailable")
    return resolved


def _guard(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size:
            connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                application_id = connection.execute("PRAGMA application_id").fetchone()[0]
                tables = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'")}
                if application_id != APPLICATION_ID or not tables.issubset(_TABLES):
                    raise ShopError("session_unavailable")
            finally:
                connection.close()
        path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, sqlite3.Error):
        raise ShopError("session_unavailable") from None


@asynccontextmanager
async def open_checkpoints(value: str | Path):
    path = validate_checkpoint_path(value)
    await run_in_threadpool(_guard, path)
    connection = None
    try:
        connection = await aiosqlite.connect(path)
        await connection.execute("PRAGMA synchronous = FULL")
        await connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        await connection.commit()
        serde = JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=(),
                                   allowed_msgpack_modules=())
        saver = AsyncSqliteSaver(connection, serde=serde)
        saver.jsonplus_serde = serde
        await saver.setup()
    except (OSError, sqlite3.Error):
        if connection is not None:
            await connection.close()
        raise ShopError("session_unavailable") from None
    try:
        yield saver
    finally:
        await connection.close()
