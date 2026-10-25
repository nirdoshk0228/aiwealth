"""Universe listing, sample (demo) portfolios and market-data status."""

from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.schemas.analytics import AssetOut, MarketDataStatus, UniverseResponse
from app.schemas.portfolio import SamplePortfolio
from app.services.market import MarketData, MarketDataStore


def universe_report(md: MarketData) -> UniverseResponse:
    u = md.universe
    available = set(md.prices.columns)
    assets = [
        AssetOut(
            asset_id=f.asset().id,
            kind="mutual_fund",
            symbol=f.code,
            name=f.name,
            category=f.category,
            amc=f.amc,
            available=f.asset().id in available,
        )
        for f in u.funds
    ]
    assets += [
        AssetOut(
            asset_id=s.asset().id,
            kind="equity",
            symbol=s.symbol,
            name=s.name,
            industry=s.industry,
            available=s.asset().id in available,
        )
        for s in u.stocks
    ]
    b = u.benchmark.asset()
    assets.append(
        AssetOut(
            asset_id=b.id, kind="index", symbol=b.symbol, name=b.name, available=b.id in available
        )
    )
    factors = md.factors
    factor_end = None
    if factors is not None:
        valid = factors.factors.dropna()
        factor_end = valid.index[-1].date() if len(valid) else None
    return UniverseResponse(
        as_of=u.as_of,
        benchmark_id=md.benchmark_id,
        data_start=md.prices.index[0].date(),
        data_end=md.data_end,
        n_rows=len(md.prices),
        factor_data_end=factor_end,
        assets=assets,
    )


@lru_cache
def _load_samples(directory: Path) -> tuple[SamplePortfolio, ...]:
    files = sorted(directory.glob("*.json"))
    samples = tuple(
        SamplePortfolio.model_validate_json(p.read_text(encoding="utf-8")) for p in files
    )
    for p, s in zip(files, samples, strict=True):
        if s.id != p.stem:
            raise ValueError(f"{p.name}: id {s.id!r} must equal the file name")
    return samples


def sample_portfolios(directory: Path | None = None) -> list[SamplePortfolio]:
    """The committed sample portfolios (data/portfolios/*.json), validated."""
    return list(_load_samples(directory or get_settings().portfolios_dir))


def market_status(store: MarketDataStore) -> MarketDataStatus:
    state, message = store.state()
    md = store.data
    return MarketDataStatus(
        state=state,
        backend=store.backend,
        loaded_at=md.loaded_at if md is not None else None,
        data_end=md.data_end if md is not None else None,
        message=message,
    )
