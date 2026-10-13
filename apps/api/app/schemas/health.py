"""Health check response schema."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness response. Does not yet check Postgres or Redis."""

    status: Literal["ok"]
    app: str
    version: str
    env: str
