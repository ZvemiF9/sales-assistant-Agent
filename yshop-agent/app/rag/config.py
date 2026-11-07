"""RAG settings are separate from the day-one shop settings."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import SERVICE_ENV_FILE


SERVICE_ROOT = SERVICE_ENV_FILE.parent


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=SERVICE_ENV_FILE, env_file_encoding="utf-8",
        extra="ignore", hide_input_in_errors=True,
    )

    embedding_model: Literal["BAAI/bge-small-zh-v1.5"] = "BAAI/bge-small-zh-v1.5"
    model_cache_dir: Path = SERVICE_ROOT / ".cache" / "fastembed"
    chroma_dir: Path = SERVICE_ROOT / "data" / "chroma"
    knowledge_dir: Path = SERVICE_ROOT / "knowledge" / "policies"
    min_score: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] = 0.53
    evidence_min_score: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] = 0.60
    top_k: Annotated[int, Field(ge=1, le=20)] = 4
    chunk_size: Annotated[int, Field(ge=1, le=600)] = 500
    chunk_overlap: Annotated[int, Field(ge=0)] = 50

    @field_validator("model_cache_dir", "chroma_dir", "knowledge_dir")
    @classmethod
    def project_directory(cls, value: Path) -> Path:
        path = value if value.is_absolute() else SERVICE_ROOT / value
        if not path.resolve().is_relative_to(SERVICE_ROOT):
            raise ValueError("RAG directories must remain inside the service project")
        if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
            raise ValueError("RAG directories cannot use filesystem links")
        return path.resolve()

    @model_validator(mode="after")
    def valid_overlap(self) -> "RagSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self
