"""Application settings loaded from environment variables and `.env` files.

Precedence (highest first): real environment variables, `apps/api/.env`
(or a `.env` in the current working directory), then the repo-root `.env`.
Missing `.env` files are ignored, so CI can run on defaults alone.

Secrets (DATABASE_URL, JWT_SECRET, GROQ_API_KEY) are SecretStr: they never appear in a repr,
a log line or an error message. Read them with `.get_secret_value()` only where they are used.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents: core, app, api, apps, <repo root>
REPO_ROOT = Path(__file__).resolve().parents[4]


def sqlalchemy_url(url: str) -> str:
    """Use the psycopg 3 driver for a plain postgres URL (Neon hands out `postgresql://`)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


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

    # Data layer: universe.json lives here; the Parquet cache in <data_dir>/raw (gitignored).
    data_dir: Path = REPO_ROOT / "data"

    # --- database (Neon Postgres in dev and prod; never logged) ---
    database_url: SecretStr | None = None
    # Optional non-pooled URL for migrations; falls back to database_url.
    database_url_direct: SecretStr | None = None

    # Where request paths read market data from. "auto" = Postgres when DATABASE_URL is set,
    # otherwise the Parquet cache. Either way, never a provider (CLAUDE.md rule 4).
    market_data_backend: Literal["auto", "parquet", "postgres"] = "auto"
    # Load market data in a background thread at startup, so the first request is fast.
    warm_market_data: bool = True

    # --- auth ---
    # Required in production. In development a random per-process secret is used (tokens stop
    # working after a restart) and a warning is logged.
    jwt_secret: SecretStr | None = None
    jwt_expire_minutes: int = 60 * 24

    # --- LLM (any OpenAI-compatible chat API; Groq free tier by default) ---
    # Swapping provider = changing these settings (and the key), not code.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    # Read from LLM_API_KEY, or GROQ_API_KEY (the default provider's variable name).
    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY")
    )
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 4
    # Chat rate limit per client IP (requests per window) and a global daily cap, because LLM
    # calls cost quota. In-process, so per API instance.
    chat_rate_limit: int = 8
    chat_rate_window_seconds: int = 60
    chat_daily_limit: int = 300
    # Where the AI trace (prompts, tool calls, retrieved chunks) is written as JSON lines.
    ai_trace_dir: Path | None = REPO_ROOT / "logs"

    # --- knowledge base (RAG) ---
    # dense = embeddings only (baseline); hybrid = BM25 + dense with reciprocal rank fusion;
    # hybrid_rerank = hybrid then a cross-encoder (better ordering, ~1-4 s more per query on CPU).
    rag_mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid"
    rag_backend: Literal["auto", "local", "postgres"] = "auto"
    rag_index_dir: Path | None = None  # default <repo>/data/docs/index
    rag_model_dir: Path | None = None  # default <repo>/.cache/models

    # --- CORS: comma-separated origins allowed to call the API from a browser ---
    cors_origins: str = "http://localhost:3000"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def universe_path(self) -> Path:
        return self.data_dir / "universe.json"

    @property
    def scenarios_dir(self) -> Path:
        return self.data_dir / "scenarios"

    @property
    def portfolios_dir(self) -> Path:
        return self.data_dir / "portfolios"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolved_market_data_backend(self) -> Literal["parquet", "postgres"]:
        if self.market_data_backend == "auto":
            return "postgres" if self.database_url else "parquet"
        return self.market_data_backend


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
