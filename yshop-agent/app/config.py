from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVICE_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    yshop_api_url: str = "http://127.0.0.1:8008/api/"
    http_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 10.0
    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("yshop_api_url")
    @classmethod
    def validate_api_url(cls, value: str) -> str:
        message = "YSHOP_API_URL must be an HTTP(S) URL without credentials, query, or fragment"
        try:
            parts = urlsplit(value)
            valid_port = parts.port is None or 0 < parts.port <= 65535
            if (
                parts.scheme not in {"http", "https"}
                or not parts.hostname
                or parts.username is not None
                or parts.password is not None
                or parts.query
                or parts.fragment
                or not valid_port
                or any(character.isspace() for character in value)
                or "\\" in value
            ):
                raise ValueError(message)
        except ValueError:
            raise ValueError(message) from None
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        normalized = []
        for origin in origins:
            try:
                url = cls.validate_api_url(origin)
                if urlsplit(url).path != "/":
                    raise ValueError
            except ValueError:
                raise ValueError("CORS_ORIGINS must contain actual HTTP(S) origins") from None
            normalized.append(url.rstrip("/"))
        return normalized
