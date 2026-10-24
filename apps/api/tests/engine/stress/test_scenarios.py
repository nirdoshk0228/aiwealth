"""Tests for the engine's scenario dataclasses and the shared StressResult helpers."""

from datetime import date

import pytest

from app.engine.stress.result import AssetImpact, StressResult
from app.engine.stress.scenarios import (
    DirectShock,
    HistoricalScenario,
    HypotheticalScenario,
    check_shock,
)


@pytest.mark.parametrize("value", [-1.0, -1.5, 1.01, 20, float("nan"), float("inf"), True, "0.1"])
def test_invalid_shock_values_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="shock must"):
        check_shock(value, "x")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.999, 0.0, 0.1, 1.0, -1 + 1e-9])
def test_valid_shock_values_accepted(value: float) -> None:
    check_shock(value, "x")


def test_hypothetical_needs_a_shock() -> None:
    with pytest.raises(ValueError, match="at least one shock"):
        HypotheticalScenario(id="h", name="h")


def test_unknown_factor_rejected() -> None:
    with pytest.raises(ValueError, match="unknown factor 'RF'"):
        HypotheticalScenario(id="h", name="h", factor_shocks={"RF": -0.01})


def test_factor_shock_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="factor MKT_RF"):
        HypotheticalScenario(id="h", name="h", factor_shocks={"MKT_RF": -20})


def test_duplicate_direct_targets_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate direct shock targets"):
        HypotheticalScenario(
            id="h",
            name="h",
            direct_shocks=(
                DirectShock("category", "gold", 0.1),
                DirectShock("category", "gold", 0.2),
            ),
        )


def test_same_target_name_different_kind_is_not_a_duplicate() -> None:
    HypotheticalScenario(
        id="h",
        name="h",
        direct_shocks=(DirectShock("category", "x", 0.1), DirectShock("sector", "x", 0.2)),
    )


@pytest.mark.parametrize(
    ("kind", "target", "match"),
    [("fund", "gold", "unknown target kind"), ("asset", "", "must not be empty")],
)
def test_invalid_direct_shock_rejected(kind: str, target: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DirectShock(kind, target, 0.1)  # type: ignore[arg-type]


def test_historical_scenario_validation() -> None:
    with pytest.raises(ValueError, match="must be before"):
        HistoricalScenario(id="x", name="x", start=date(2020, 3, 23), end=date(2020, 1, 14))
    with pytest.raises(ValueError, match="id must not be empty"):
        HistoricalScenario(id="", name="x", start=date(2020, 1, 1), end=date(2020, 1, 2))


def impact(a: str, c: float) -> AssetImpact:
    return AssetImpact(a, 0.25, c * 4, c, c * 100, "historical")


def test_worst_contributors_lists_losses_worst_first() -> None:
    r = StressResult(
        scenario_id="s",
        scenario_name="s",
        scenario_type="historical",
        portfolio_value=100,
        pnl=-0.04,
        pnl_rupees=-4,
        impacts=(impact("a", -0.01), impact("b", 0.02), impact("c", -0.03), impact("d", -0.01)),
        notes=(),
        warnings=(),
    )
    assert [i.asset_id for i in r.worst_contributors()] == ["c", "a", "d"]  # tie a/d by id
    assert [i.asset_id for i in r.worst_contributors(1)] == ["c"]
    with pytest.raises(ValueError):
        r.worst_contributors(0)
    frame = r.to_frame()
    assert list(frame.index) == ["a", "b", "c", "d"]
    assert frame["contribution"].sum() == pytest.approx(-0.03)
