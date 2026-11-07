"""Local Day 8 Agent with fresh databases and a copied retrieval index."""

import argparse
import shutil
import socket
import sys
from uuid import uuid4

import uvicorn

from app.config import SERVICE_ENV_FILE, Settings
from app.main import create_app
from app.rag.config import RagSettings
from app.storage.config import SessionSettings


def prepare_runtime(h5_origin: str, *, rag_settings: RagSettings | None = None):
    settings = Settings(cors_origins=[h5_origin])
    original = rag_settings if rag_settings is not None else RagSettings()
    source = original.chroma_dir
    if not source.is_dir() or any(path.is_symlink() or path.is_junction()
                                  for path in (source, *source.parents, *source.rglob("*"))):
        raise ValueError("A regular existing retrieval index is required.")
    root = SERVICE_ENV_FILE.parent.resolve()
    runtime = root / ".tmp" / f"day8-live-{uuid4().hex}"
    sessions = SessionSettings(db_path=runtime / "sessions.sqlite3")
    rag = RagSettings(**{**original.model_dump(), "chroma_dir": runtime / "chroma"})
    # CreateNew semantics: never reuse a prior runtime or its databases.
    runtime.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, rag.chroma_dir)
    return settings, rag, sessions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.start_day8",
                                     exit_on_error=False, allow_abbrev=False)
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--h5-origin", default="http://127.0.0.1:8080")
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown or not 1 <= args.port <= 65535:
            raise ValueError
    except (argparse.ArgumentError, ValueError):
        print("Invalid startup arguments. Use --help for supported options.", file=sys.stderr)
        return 2
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        settings, rag, sessions = prepare_runtime(args.h5_origin)
        application = create_app(settings, rag_settings=rag, session_settings=sessions)
        print(f"Agent: http://127.0.0.1:{args.port}; runtime: {sessions.db_path.parent}", flush=True)
        uvicorn.run(application, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("Cannot start isolated Agent. Check the port, settings and existing index.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
