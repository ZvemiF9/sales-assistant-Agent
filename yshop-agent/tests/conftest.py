"""Shared test-only configuration for application lifespans."""

from uuid import uuid4

import pytest

from app.config import SERVICE_ENV_FILE
from app.storage.config import SessionSettings


@pytest.fixture
def isolated_session_settings():
    return SessionSettings(
        _env_file=None,
        db_path=SERVICE_ENV_FILE.parent / ".tmp" / f"pytest-sessions-{uuid4().hex}" / "sessions.sqlite3",
    )


@pytest.fixture(autouse=True)
def isolate_default_application_session_database(monkeypatch, isolated_session_settings):
    """Keep every implicit create_app lifecycle away from the operational database."""
    import app.main as main_module

    monkeypatch.setattr(main_module, "SessionSettings", lambda: isolated_session_settings)
