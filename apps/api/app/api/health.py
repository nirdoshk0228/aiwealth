"""Health router, mounted at both `/health` and `/api/v1/health`."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Liveness probe: the process is up and serving requests."""
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        env=settings.app_env,
    )
