"""Application settings loaded from environment variables and `.env` files.

Precedence (highest first): real environment variables, `apps/api/.env`
(or a `.env` in the current working directory), then the repo-root `.env`.
Missing `.env` files are ignored, so CI can run on defaults alone.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents: core, app, api, apps, <repo root>
REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Typed application configuration."""

    model_config = SettingsConfigDict(
        # Later files take priority over earlier ones.
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        # The shared .env also holds values for docker-compose and later phases.
        extra="ignore",
    )

    app_name: str = "AIWealth API"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
