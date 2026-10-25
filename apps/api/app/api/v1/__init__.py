"""Version 1 API router aggregator."""

from fastapi import APIRouter

from app.api import health
from app.api.v1 import accounts, ai, analytics, planning, universe

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(universe.router)
api_router.include_router(analytics.router)
api_router.include_router(planning.router)
api_router.include_router(accounts.router)
api_router.include_router(ai.router)
