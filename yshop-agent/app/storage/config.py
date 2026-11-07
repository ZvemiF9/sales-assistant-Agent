"""A dedicated SQLite session path confined to the service's writable data."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import SERVICE_ENV_FILE


SERVICE_ROOT = SERVICE_ENV_FILE.parent


class SessionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SESSION_", env_file=SERVICE_ENV_FILE,
        env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True)
    db_path: Path = SERVICE_ROOT / "data" / "day5-sessions.sqlite3"

    @property
    def checkpoint_path(self) -> Path:
        return self.db_path.with_name("cart-checkpoints.sqlite3")

    @field_validator("db_path")
    @classmethod
    def database_path(cls, value: Path) -> Path:
        path = value if value.is_absolute() else SERVICE_ROOT / value
        resolved = path.resolve()
        if (not resolved.is_relative_to(SERVICE_ROOT)
                or resolved.relative_to(SERVICE_ROOT).parts[0] not in {"data", ".tmp"}
                or resolved.name not in {"sessions.sqlite3", "day5-sessions.sqlite3"}
                or "chroma" in resolved.relative_to(SERVICE_ROOT).parts
                or any(p.is_symlink() or p.is_junction() for p in (path, *path.parents))):
            raise ValueError("SESSION_DB_PATH must be a dedicated sessions.sqlite3 inside service data or .tmp")
        return resolved
