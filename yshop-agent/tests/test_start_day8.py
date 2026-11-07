"""Day-eight startup isolation; no server or original index is opened."""

import importlib
from uuid import uuid4

import pytest

from app.config import SERVICE_ENV_FILE
from app.rag.config import RagSettings


@pytest.fixture
def index():
    source = SERVICE_ENV_FILE.parent / ".tmp" / f"day8-launch-test-{uuid4().hex}" / "index"
    source.mkdir(parents=True)
    (source / "chroma.sqlite3").write_bytes(b"test index bytes")
    (source / "collection").mkdir()
    (source / "collection" / "data.bin").write_bytes(b"test vector bytes")
    return RagSettings(_env_file=None, chroma_dir=source)


def launcher():
    try:
        return importlib.import_module("scripts.start_day8")
    except ModuleNotFoundError:
        pytest.fail("the isolated day-eight launcher is not implemented")


def test_runtime_uses_new_databases_and_complete_index_copy(index):
    settings, rag, sessions = launcher().prepare_runtime("http://127.0.0.1:8080", rag_settings=index)
    assert settings.cors_origins == ["http://127.0.0.1:8080"]
    assert sessions.db_path.parent.name.startswith("day8-live-")
    assert sessions.db_path.name == "sessions.sqlite3"
    assert not sessions.db_path.exists() and not sessions.checkpoint_path.exists()
    assert rag.chroma_dir == sessions.db_path.parent / "chroma"
    assert rag.chroma_dir != index.chroma_dir
    for path in index.chroma_dir.rglob("*"):
        if path.is_file():
            assert (rag.chroma_dir / path.relative_to(index.chroma_dir)).read_bytes() == path.read_bytes()
    second = launcher().prepare_runtime("http://127.0.0.1:8080", rag_settings=index)
    assert second[2].db_path != sessions.db_path


@pytest.mark.parametrize("origin", ["http://localhost:8080/path", "http://user:private@localhost:8080",
                                   "http://localhost:8080?token=private"])
def test_invalid_origin_is_rejected_before_copy(index, origin, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module.shutil, "copytree", lambda *a, **k: pytest.fail("invalid input must not copy"))
    with pytest.raises(ValueError):
        module.prepare_runtime(origin, rag_settings=index)


def test_missing_index_is_not_opened_or_silently_rebuilt(index):
    index = index.model_copy(update={"chroma_dir": index.chroma_dir / "missing"})
    with pytest.raises(ValueError):
        launcher().prepare_runtime("http://127.0.0.1:8080", rag_settings=index)


@pytest.mark.parametrize("args", [["--port", "0"], ["--unknown", "private"], ["--port", "private"]])
def test_invalid_cli_arguments_are_static_and_never_start_server(args, monkeypatch, capsys):
    module = launcher()
    monkeypatch.setattr(module.uvicorn, "run", lambda *a, **k: pytest.fail("must not start"))
    assert module.main(args) == 2
    output = capsys.readouterr()
    assert "private" not in output.out + output.err


def test_server_is_local_and_access_logging_disabled(index, monkeypatch):
    module = launcher()
    prepared = module.prepare_runtime("http://127.0.0.1:8080", rag_settings=index)
    monkeypatch.setattr(module, "prepare_runtime", lambda origin: prepared)
    observed = []
    monkeypatch.setattr(module.uvicorn, "run", lambda *a, **k: observed.append((a, k)))
    assert module.main(["--port", "18018"]) == 0
    assert len(observed) == 1
    assert observed[0][1]["host"] == "127.0.0.1"
    assert observed[0][1]["access_log"] is False
