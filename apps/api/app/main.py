"""FastAPI application factory.

Run locally with: uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import __version__
from app.ai.llm.base import LLMError, QuotaExhausted
from app.ai.llm.factory import LLMNotConfigured
from app.ai.rag.kb import KnowledgeBase
from app.ai.scenario_parser import ScenarioParseError
from app.api import health
from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.core.ratelimit import RateLimited, RateLimiter
from app.core.security import jwt_secret
from app.data.errors import NotInCacheError
from app.db.session import DatabaseNotConfigured
from app.schemas.chat import ChatEventEnvelope
from app.services.errors import AnalyticsInputError, MarketDataUnavailable, NotFound
from app.services.market import MarketData, MarketDataStore, load_market_data

logger = logging.getLogger(__name__)


def _error(code: int) -> Callable[[Request, Exception], JSONResponse]:
    def handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=code, content={"detail": str(exc)})

    return handler


def _rate_limited(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RateLimited)
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after)},
    )


def _with_extra_schemas(app: FastAPI, models: list[type[BaseModel]]) -> None:
    """Add models that only appear in non-JSON responses (the chat SSE stream) to the OpenAPI
    components, so the web client generates their types too."""
    original = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        spec = original()
        schemas = spec.setdefault("components", {}).setdefault("schemas", {})
        for model in models:
            js = model.model_json_schema(ref_template="#/components/schemas/{model}")
            for name, sub in js.pop("$defs", {}).items():
                schemas.setdefault(name, sub)
            schemas[model.__name__] = js
        app.openapi_schema = spec
        return spec

    app.openapi = openapi  # type: ignore[method-assign]


def create_app(
    settings: Settings | None = None,
    *,
    market_loader: Callable[[], MarketData] | None = None,
    knowledge: KnowledgeBase | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application.

    `market_loader` replaces the configured market-data source (tests pass synthetic data).
    """
    settings = settings or get_settings()
    jwt_secret(settings)  # fail fast in production without JWT_SECRET
    store = MarketDataStore(
        market_loader or (lambda: load_market_data(settings)),
        backend=settings.resolved_market_data_backend(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.warm_market_data:
            store.warm_in_background()
        yield

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.market_store = store
    app.state.knowledge = knowledge or KnowledgeBase(settings)
    app.state.chat_limiter = RateLimiter(
        limit=settings.chat_rate_limit,
        window_seconds=settings.chat_rate_window_seconds,
        daily_limit=settings.chat_daily_limit,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.add_exception_handler(AnalyticsInputError, _error(status.HTTP_422_UNPROCESSABLE_CONTENT))
    app.add_exception_handler(NotFound, _error(status.HTTP_404_NOT_FOUND))
    app.add_exception_handler(MarketDataUnavailable, _error(status.HTTP_503_SERVICE_UNAVAILABLE))
    app.add_exception_handler(NotInCacheError, _error(status.HTTP_503_SERVICE_UNAVAILABLE))
    app.add_exception_handler(DatabaseNotConfigured, _error(status.HTTP_503_SERVICE_UNAVAILABLE))
    app.add_exception_handler(RateLimited, _rate_limited)
    app.add_exception_handler(LLMNotConfigured, _error(status.HTTP_503_SERVICE_UNAVAILABLE))
    app.add_exception_handler(QuotaExhausted, _error(status.HTTP_503_SERVICE_UNAVAILABLE))
    app.add_exception_handler(ScenarioParseError, _error(status.HTTP_422_UNPROCESSABLE_CONTENT))
    app.add_exception_handler(LLMError, _error(status.HTTP_502_BAD_GATEWAY))

    # Bare /health for container/load-balancer probes; versioned API under /api/v1.
    app.include_router(health.router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    _with_extra_schemas(app, [ChatEventEnvelope])

    return app


app = create_app()
