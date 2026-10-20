"""run_seed with fake providers and a temp Parquet cache (no network)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.data.errors import BadResponseError, NotFoundError
from app.data.models import AssetRef, PriceHistory
from app.data.repository import ParquetRepository
from app.data.seed import run_seed

FUND = AssetRef(
    kind="mutual_fund", symbol="122639", name="Parag Parikh Flexi Cap Fund - Direct Plan - Growth"
)
STOCK = AssetRef(kind="equity", symbol="TCS", name="Tata Consultancy Services Ltd.")
BAD = AssetRef(kind="equity", symbol="NOPE", name="Missing")
NIFTY = AssetRef(kind="index", symbol="^NSEI", name="Nifty 50")


class FakeProvider:
    source = "fake"

    def __init__(
        self, source_names: dict[str, str] | None = None, fail: dict[str, Exception] | None = None
    ):
        self.source_names = source_names or {}
        self.fail = fail or {}
        self.calls: list[str] = []

    def fetch_history(self, asset: AssetRef, start: date | None = None) -> PriceHistory:
        self.calls.append(asset.id)
        if asset.id in self.fail:
            raise self.fail[asset.id]
        idx = pd.bdate_range("2026-01-05", periods=3, name="date")
        return PriceHistory(
            asset=asset,
            prices=pd.Series([10.0, 11.0, 12.0], index=idx),
            source=self.source,
            source_name=self.source_names.get(asset.id, asset.name),
        )


@pytest.fixture
def repo(tmp_path: Path) -> ParquetRepository:
    return ParquetRepository(tmp_path)


def test_one_failure_does_not_stop_the_run(repo: ParquetRepository) -> None:
    provider = FakeProvider(fail={BAD.id: NotFoundError("no such ticker")})
    sleeps: list[float] = []

    report = run_seed(
        [FUND, BAD, STOCK, NIFTY],
        {"mutual_fund": provider, "equity": provider, "index": provider},
        repo,
        pause=0.5,
        sleep=sleeps.append,
    )

    assert provider.calls == [FUND.id, BAD.id, STOCK.id, NIFTY.id]
    assert not report.ok
    assert [r.asset for r in report.failed] == [BAD]
    assert "NotFoundError: no such ticker" in report.failed[0].error
    assert repo.exists(FUND) and repo.exists(STOCK) and repo.exists(NIFTY)
    assert not repo.exists(BAD)
    assert sleeps == [0.5, 0.5, 0.5]  # between requests, not after the last one


def test_successful_run_reports_rows_and_dates(repo: ParquetRepository) -> None:
    provider = FakeProvider()
    report = run_seed([FUND], {"mutual_fund": provider}, repo, sleep=lambda s: None)
    assert report.ok
    (r,) = report.results
    assert (r.rows, r.first_date, r.last_date) == (3, date(2026, 1, 5), date(2026, 1, 7))
    table = report.format_table()
    assert "mutual_fund:122639" in table and "1/1 assets cached, 0 failed" in table


def test_fund_name_mismatch_is_warned_and_persisted(repo: ParquetRepository) -> None:
    provider = FakeProvider(source_names={FUND.id: "Some Other Fund - Regular Plan - IDCW"})
    report = run_seed([FUND], {"mutual_fund": provider}, repo, sleep=lambda s: None)
    assert report.ok  # still cached, but flagged
    assert any("name mismatch" in w for w in report.results[0].warnings)
    assert any("name mismatch" in w for w in repo.metadata(FUND).warnings)


def test_name_check_applies_to_funds_only(repo: ParquetRepository) -> None:
    # Yahoo returns the ticker as source_name, which never equals the company name.
    provider = FakeProvider(source_names={STOCK.id: "TCS.NS"})
    report = run_seed([STOCK], {"equity": provider}, repo, sleep=lambda s: None)
    assert report.results[0].warnings == ()


def test_bugs_are_not_swallowed(repo: ParquetRepository) -> None:
    provider = FakeProvider(fail={FUND.id: KeyError("bug")})
    with pytest.raises(KeyError):
        run_seed([FUND], {"mutual_fund": provider}, repo, sleep=lambda s: None)


def test_bad_response_is_reported_as_failure(repo: ParquetRepository) -> None:
    provider = FakeProvider(fail={FUND.id: BadResponseError("garbled")})
    report = run_seed([FUND], {"mutual_fund": provider}, repo, sleep=lambda s: None)
    assert report.failed[0].error == "BadResponseError: garbled"
