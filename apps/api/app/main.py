"""FastAPI application factory.

Run locally with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app import __version__
from app.api import health
from app.api.v1 import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)

    # Bare /health for container/load-balancer probes; versioned API under /api/v1.
    app.include_router(health.router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
