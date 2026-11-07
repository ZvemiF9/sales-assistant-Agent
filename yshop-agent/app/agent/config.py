"""Generation settings, separate from shop credentials and RAG configuration."""

from typing import Annotated
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import SERVICE_ENV_FILE, Settings


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVICE_ENV_FILE, env_file_encoding="utf-8",
        extra="ignore", hide_input_in_errors=True,
    )

    llm_base_url: Annotated[str, Field(strict=True)] | None = None
    llm_model: Annotated[str, Field(strict=True)] | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 30.0
    llm_proxy_url: Annotated[str, Field(strict=True)] | None = None
    llm_enable_thinking: Annotated[bool, Field(strict=True)] | None = None

    @field_validator("llm_enable_thinking", mode="before")
    @classmethod
    def validate_thinking_mode(cls, value):
        if value is None or type(value) is bool:
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return None
            if normalized in {"true", "false"}:
                return normalized == "true"
        raise ValueError("LLM_ENABLE_THINKING must be true, false or unset")

    @field_validator("llm_proxy_url", mode="before")
    @classmethod
    def validate_proxy_url(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        message = "LLM_PROXY_URL must be an HTTP loopback origin without credentials or URL extras"
        if not isinstance(value, str):
            raise ValueError(message)
        try:
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError(message)
            normalized = Settings.validate_api_url(value)
            parts = urlsplit(value)
            if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
                    or parts.path not in {"", "/"}):
                raise ValueError(message)
        except ValueError:
            raise ValueError(message) from None
        return normalized.rstrip("/")

    @field_validator("llm_base_url", "llm_model", "llm_api_key", mode="before")
    @classmethod
    def blank_is_unconfigured(cls, value):
        text = value.get_secret_value() if isinstance(value, SecretStr) else value
        return None if isinstance(text, str) and not text.strip() else value

    @field_validator("llm_base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError
            return Settings.validate_api_url(value)
        except ValueError:
            raise ValueError("LLM_BASE_URL must be an HTTP(S) URL without credentials, query, or fragment") from None

    @field_validator("llm_model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) > 256 or any(character.isspace() or ord(character) < 32 or ord(character) == 127
                                    for character in value)
        ):
            raise ValueError("LLM_MODEL must be at most 256 characters without whitespace or controls")
        return value

    @field_validator("llm_api_key")
    @classmethod
    def validate_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and any(not 33 <= ord(character) <= 126
                                     for character in value.get_secret_value()):
            raise ValueError("LLM_API_KEY must contain visible ASCII characters only")
        return value

    @field_validator("llm_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_type(cls, value):
        if isinstance(value, bool):
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive finite number")
        return value

    @property
    def configured(self) -> bool:
        return all(value is not None for value in (self.llm_base_url, self.llm_model, self.llm_api_key))
