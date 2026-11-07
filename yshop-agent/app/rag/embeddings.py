"""Lazy CPU embedding; API queries never acquire remote model files."""

import hashlib
from importlib.metadata import version
import logging
import os
from pathlib import Path
import re
from threading import Lock
from typing import Protocol
import warnings

from app.errors import ShopError
from app.rag.config import RagSettings, SERVICE_ROOT


class Embeddings(Protocol):
    dimension: int

    def fingerprint(self) -> dict: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _quiet_logger(namespace: str) -> None:
    logger = logging.getLogger(namespace)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


def configure_model_environment(cache_dir: Path) -> None:
    # Set before importing FastEmbed/HF, which capture default paths at import.
    os.environ["HF_HOME"] = str(SERVICE_ROOT / ".cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_XET_CACHE"] = str(SERVICE_ROOT / ".cache" / "hf-xet")
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    for namespace in ("huggingface_hub", "fastembed"):
        _quiet_logger(namespace)


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LocalEmbeddings:
    dimension = 512

    def __init__(self, settings: RagSettings, *, allow_download: bool = False):
        self.settings = settings
        self.allow_download = allow_download
        self._model = None
        self._lock = Lock()
        self._fingerprint = None
        self._model_path = None

    def _cached_path(self) -> Path:
        try:
            cache = self.settings.model_cache_dir
            repo = cache / "models--Qdrant--bge-small-zh-v1.5"
            revision = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[a-f0-9]{40}", revision) is None:
                raise ValueError
            path = repo / "snapshots" / revision
            files = ("model_optimized.onnx", "config.json", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json")
            for name in files:
                file = path / name
                if not file.is_file() or not file.resolve().is_relative_to(cache):
                    raise ValueError
            return path
        except (OSError, ValueError, UnicodeError):
            raise ShopError("rag_model_unavailable") from None

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            configure_model_environment(self.settings.model_cache_dir)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"Cannot enable progress bars: environment variable.*", category=UserWarning)
                    from fastembed import TextEmbedding
                    from loguru import logger
                    logger.disable("fastembed")
                    kwargs = {}
                    if self._model_path is not None or not self.allow_download:
                        kwargs["specific_model_path"] = str(self._model_path or self._cached_path())
                    self._model = TextEmbedding(
                        model_name=self.settings.embedding_model,
                        cache_dir=str(self.settings.model_cache_dir), threads=2,
                        providers=["CPUExecutionProvider"], cuda=False,
                        local_files_only=not self.allow_download, **kwargs,
                    )
                    self._model_path = self._cached_path()
                return self._model
            except ShopError:
                raise
            except Exception:
                raise ShopError("rag_model_unavailable") from None

    def fingerprint(self) -> dict:
        if self._fingerprint is None:
            try:
                path = self._cached_path()
            except ShopError:
                if not self.allow_download:
                    raise
                self._load()
                path = self._cached_path()
            try:
                self._model_path = path
                self._fingerprint = {
                    "model": self.settings.embedding_model,
                    "dimension": self.dimension,
                    "preprocessing": "PolicyChunk.text-exact;question-strip;no-prefix;fastembed-embed-v1",
                    "weights_sha256": _file_hash(path / "model_optimized.onnx"),
                    "tokenizer_sha256": {name: _file_hash(path / name) for name in
                        ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")},
                    "versions": {name: version(name) for name in
                        ("fastembed", "onnxruntime", "chromadb", "langchain-text-splitters", "tokenizers")},
                }
            except Exception:
                raise ShopError("rag_model_unavailable") from None
        return self._fingerprint

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return [vector.tolist() for vector in self._load().embed(texts, batch_size=32)]
        except ShopError:
            raise
        except Exception:
            raise ShopError("rag_unavailable") from None

    def embed_query(self, text: str) -> list[float]:
        # Deliberately use the same no-prefix embedding path as indexed text.
        return self.embed_documents([text])[0]
