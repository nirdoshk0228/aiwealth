"""Download assets from providers into the cache. Used only by scripts/seed_data.py.

This is the single code path allowed to call providers (CLAUDE.md rule 4). It is a module
rather than script-only code so it can be tested with fake providers and a temp cache.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date

from app.data.errors import ProviderError
from app.data.models import AssetKind, AssetRef
from app.data.providers.base import PriceProvider
from app.data.repository import SeriesRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssetResult:
    asset: AssetRef
    error: str | None = None
    rows: int = 0
    first_date: date | None = None
    last_date: date | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class SeedReport:
    results: tuple[AssetResult, ...]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failed(self) -> list[AssetResult]:
        return [r for r in self.results if not r.ok]

    def format_table(self) -> str:
        lines = [f"{'asset':<24} {'status':<7} {'rows':>6}  {'first':<10}  {'last':<10}  notes"]
        for r in self.results:
            status = "ok" if r.ok else "FAILED"
            first = str(r.first_date or "")
            last = str(r.last_date or "")
            notes = r.error or "; ".join(r.warnings)
            lines.append(
                f"{r.asset.id:<24} {status:<7} {r.rows:>6}  {first:<10}  {last:<10}  {notes}"
            )
        n_ok = sum(r.ok for r in self.results)
        lines.append(f"\n{n_ok}/{len(self.results)} assets cached, {len(self.failed)} failed")
        return "\n".join(lines)


def run_seed(
    assets: Sequence[AssetRef],
    providers: Mapping[AssetKind, PriceProvider],
    repo: SeriesRepository,
    *,
    pause: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> SeedReport:
    """Fetch each asset's full history and save it, one asset at a time.

    - A ProviderError (not found, bad payload, retries exhausted) marks that asset failed
      and the run continues; anything else is a bug and propagates.
    - For mutual funds, the scheme name returned by the source is compared with the name in
      universe.json; a mismatch (wrong code, or a scheme renamed since) is saved as a warning.
    - `pause` seconds between requests keeps us polite to free APIs.
    """
    results: list[AssetResult] = []
    for i, asset in enumerate(assets):
        if i:
            sleep(pause)
        provider = providers[asset.kind]
        try:
            history = provider.fetch_history(asset)
        except ProviderError as exc:
            logger.error("%s: %s", asset.id, exc)
            results.append(AssetResult(asset=asset, error=f"{type(exc).__name__}: {exc}"))
            continue

        if asset.kind == "mutual_fund" and history.source_name != asset.name:
            msg = f"name mismatch: universe {asset.name!r} vs source {history.source_name!r}"
            logger.warning("%s: %s", asset.id, msg)
            history = replace(history, warnings=(*history.warnings, msg))

        meta = repo.save(history)
        results.append(
            AssetResult(
                asset=asset,
                rows=meta.rows,
                first_date=meta.first_date,
                last_date=meta.last_date,
                warnings=history.warnings,
            )
        )
        logger.info(
            "%s: cached %d rows (%s to %s)", asset.id, meta.rows, meta.first_date, meta.last_date
        )
    return SeedReport(results=tuple(results))
