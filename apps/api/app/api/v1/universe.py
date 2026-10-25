"""Universe, market-data status and demo routes (public)."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import MarketDataDep, get_store
from app.schemas.analytics import MarketDataStatus, UniverseResponse
from app.schemas.portfolio import SamplePortfolio
from app.services import universe
from app.services.market import MarketDataStore

router = APIRouter()


@router.get("/universe", response_model=UniverseResponse, tags=["universe"])
def get_universe(md: MarketDataDep) -> UniverseResponse:
    """Funds, stocks and the benchmark covered by the platform, with the data window."""
    return universe.universe_report(md)


@router.get("/status", response_model=MarketDataStatus, tags=["universe"])
def get_status(store: Annotated[MarketDataStore, Depends(get_store)]) -> MarketDataStatus:
    """Whether market data is loaded yet (never blocks; for the UI's cold-start state)."""
    return universe.market_status(store)


@router.get("/demo/portfolios", response_model=list[SamplePortfolio], tags=["demo"])
def demo_portfolios() -> list[SamplePortfolio]:
    """Sample portfolios for the no-login demo. Illustrative, not recommendations."""
    return universe.sample_portfolios()
