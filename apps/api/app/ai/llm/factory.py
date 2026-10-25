"""Build the configured LLMClient from settings (no provider names in calling code)."""

from functools import lru_cache

from app.ai.llm.base import LLMClient, LLMError
from app.ai.llm.openai_compat import OpenAICompatibleClient
from app.core.config import Settings, get_settings


class LLMNotConfigured(LLMError):
    """No API key: the assistant is unavailable (the rest of the API works)."""


def make_client(settings: Settings) -> LLMClient:
    if settings.llm_api_key is None:
        raise LLMNotConfigured("the AI assistant is not configured (no LLM API key)")
    return OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache
def get_llm_client() -> LLMClient:
    """Process-wide client (FastAPI dependency; tests override it with a FakeLLMClient)."""
    return make_client(get_settings())
