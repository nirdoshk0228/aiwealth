import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import REPO_ROOT
from app.data.universe import Universe, load_universe

UNIVERSE_JSON = REPO_ROOT / "data" / "universe.json"


def raw() -> dict:
    return json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))


def test_committed_universe_is_valid_and_complete() -> None:
    u = load_universe(UNIVERSE_JSON)
    assert len(u.funds) == 15
    assert len(u.stocks) == 20
    assert u.benchmark.symbol == "^NSEI"
    assert u.data_start == date(2015, 1, 1)
    assets = u.assets()
    assert len(assets) == 36
    assert len({a.id for a in assets}) == 36


def test_universe_covers_the_agreed_categories() -> None:
    u = load_universe(UNIVERSE_JSON)
    assert {f.category for f in u.funds} >= {
        "large_cap",
        "flexi_cap",
        "mid_cap",
        "small_cap",
        "gold",
        "hybrid_balanced_advantage",
    }
    assert any(f.category.startswith("debt_") for f in u.funds)
    # Sector spread rather than top-by-weight: no single NSE industry dominates.
    industries = [s.industry for s in u.stocks]
    assert len(set(industries)) >= 10
    assert max(industries.count(i) for i in set(industries)) <= 4


def test_excluded_stocks_are_absent() -> None:
    symbols = {s.symbol for s in load_universe(UNIVERSE_JSON).stocks}
    assert "ITC" not in symbols  # unadjusted ITC Hotels demerger, see docs/data.md
    assert "TMPV" not in symbols


def test_find_by_code_or_symbol() -> None:
    u = load_universe(UNIVERSE_JSON)
    assert u.find("118663").kind == "mutual_fund"
    assert u.find("m&m").symbol == "M&M"
    assert u.find("^NSEI").kind == "index"
    with pytest.raises(KeyError):
        u.find("ITC")


def test_duplicate_fund_code_rejected() -> None:
    data = raw()
    data["funds"].append(dict(data["funds"][0]))
    with pytest.raises(ValidationError, match="duplicate fund code"):
        Universe.model_validate(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["funds"][0].update(code="12O586"),  # letter O
        lambda d: d["funds"][0].update(category="thematic"),
        lambda d: d["stocks"][0].update(symbol="HDFCBANK.NS"),
        lambda d: d["stocks"][0].update(isin="XX040A01034"),
        lambda d: d["stocks"][0].update(sybmol="typo"),  # unknown key
        lambda d: d["benchmark"].update(symbol="NSEI"),
    ],
    ids=[
        "non-numeric-code",
        "unknown-category",
        "suffix-in-symbol",
        "bad-isin",
        "extra-key",
        "benchmark-no-caret",
    ],
)
def test_invalid_entries_rejected(mutate) -> None:
    data = raw()
    mutate(data)
    with pytest.raises(ValidationError):
        Universe.model_validate(data)


def test_load_universe_from_custom_path(tmp_path: Path) -> None:
    p = tmp_path / "u.json"
    p.write_text(UNIVERSE_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    assert len(load_universe(p).funds) == 15
