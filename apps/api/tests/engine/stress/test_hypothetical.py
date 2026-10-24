"""Tests for app.engine.stress.hypothetical: factor shocks computed by hand as
weight x beta x shock, direct-shock precedence, category/sector shocks and warnings."""

import pytest

from app.engine.stress.hypothetical import hypothetical_shock
from app.engine.stress.scenarios import (
    AssetExposure,
    AssetGroups,
    DirectShock,
    HypotheticalScenario,
)

EXPOSURES = {
    "equity:A": AssetExposure(
        betas={"MKT_RF": 1.2, "SMB": 0.5, "HML": 0.3, "WML": 0.0}, r_squared=0.6
    ),
    "equity:B": AssetExposure(
        betas={"MKT_RF": 0.8, "SMB": -0.2, "HML": 0.1, "WML": 0.1}, r_squared=0.5
    ),
    "mutual_fund:G": AssetExposure(
        betas={"MKT_RF": 0.02, "SMB": 0.0, "HML": 0.0, "WML": 0.0}, r_squared=0.01
    ),
    "mutual_fund:D": AssetExposure(
        betas={"MKT_RF": 0.01, "SMB": 0.01, "HML": 0.0, "WML": 0.0}, r_squared=0.02
    ),
}
GROUPS = {
    "equity:A": AssetGroups(sector="Information Technology"),
    "equity:B": AssetGroups(sector="Financial Services"),
    "mutual_fund:G": AssetGroups(category="gold"),
    "mutual_fund:D": AssetGroups(category="debt_gilt"),
}
BEAR = {"MKT_RF": -0.20, "SMB": -0.10}


def scen(factors: dict | None = None, *direct: DirectShock) -> HypotheticalScenario:
    return HypotheticalScenario(id="h", name="h", factor_shocks=factors or {}, direct_shocks=direct)


def by_id(result) -> dict:
    return {i.asset_id: i for i in result.impacts}


def test_factor_shock_by_hand() -> None:
    # A: 1.2(-0.20) + 0.5(-0.10) = -0.29     B: 0.8(-0.20) - 0.2(-0.10) = -0.14
    # P&L = 0.6(-0.29) + 0.4(-0.14) = -0.174 - 0.056 = -0.23
    r = hypothetical_shock(
        {"equity:A": 0.6, "equity:B": 0.4}, scen(BEAR), portfolio_value=500_000, exposures=EXPOSURES
    )
    imp = by_id(r)
    assert imp["equity:A"].asset_return == pytest.approx(-0.29)
    assert imp["equity:B"].asset_return == pytest.approx(-0.14)
    assert imp["equity:A"].contribution == pytest.approx(-0.174)
    assert imp["equity:B"].contribution == pytest.approx(-0.056)
    assert r.pnl == pytest.approx(-0.23)
    assert r.pnl_rupees == pytest.approx(-115_000)
    assert imp["equity:A"].contribution_rupees == pytest.approx(-87_000)
    assert {i.source for i in r.impacts} == {"factor"}
    assert r.warnings == ()
    assert [i.asset_id for i in r.worst_contributors()] == ["equity:A", "equity:B"]
    assert r.drawdown is None and r.sessions is None


def test_unshocked_factors_contribute_nothing() -> None:
    # HML and WML betas exist but are not shocked: only MKT_RF counts.
    r = hypothetical_shock(
        {"equity:A": 1.0}, scen({"MKT_RF": -0.1}), portfolio_value=1, exposures=EXPOSURES
    )
    assert r.pnl == pytest.approx(-0.12)


def test_direct_asset_shock_overrides_factor_and_says_so() -> None:
    r = hypothetical_shock(
        {"equity:A": 0.6, "equity:B": 0.4},
        scen(BEAR, DirectShock("asset", "equity:A", -0.05)),
        portfolio_value=1,
        exposures=EXPOSURES,
    )
    a = by_id(r)["equity:A"]
    assert a.asset_return == -0.05
    assert a.source == "direct_asset"
    assert a.factor_implied == pytest.approx(-0.29)  # kept for display
    assert r.pnl == pytest.approx(0.6 * -0.05 + 0.4 * -0.14)
    assert any("equity:A" in n and "overrides the factor-implied -29.00%" in n for n in r.notes)


def test_precedence_asset_over_sector_over_category() -> None:
    groups = {
        "x": AssetGroups(category="large_cap", sector="IT"),
        "y": AssetGroups(category="large_cap", sector="IT"),
        "z": AssetGroups(category="large_cap"),
    }
    s = scen(
        None,
        DirectShock("category", "large_cap", -0.01),
        DirectShock("sector", "IT", -0.02),
        DirectShock("asset", "x", -0.03),
    )
    r = hypothetical_shock({"x": 0.2, "y": 0.3, "z": 0.5}, s, portfolio_value=1, groups=groups)
    imp = by_id(r)
    assert (imp["x"].asset_return, imp["x"].source) == (-0.03, "direct_asset")
    assert (imp["y"].asset_return, imp["y"].source) == (-0.02, "direct_sector")
    assert (imp["z"].asset_return, imp["z"].source) == (-0.01, "direct_category")
    assert r.pnl == pytest.approx(0.2 * -0.03 + 0.3 * -0.02 + 0.5 * -0.01)


def test_category_shock_hits_every_fund_in_the_category() -> None:
    groups = {
        "g1": AssetGroups(category="gold"),
        "g2": AssetGroups(category="gold"),
        "d": AssetGroups(category="debt_gilt"),
    }
    s = scen(
        None, DirectShock("category", "gold", 0.10), DirectShock("category", "debt_gilt", -0.02)
    )
    r = hypothetical_shock(
        {"g1": 0.25, "g2": 0.25, "d": 0.5}, s, portfolio_value=200_000, groups=groups
    )
    # 0.25(0.10) + 0.25(0.10) + 0.5(-0.02) = 0.04
    assert r.pnl == pytest.approx(0.04)
    assert r.pnl_rupees == pytest.approx(8_000)
    assert [i.asset_id for i in r.worst_contributors()] == ["d"]


def test_mixed_factor_and_direct_contributions_sum_to_total() -> None:
    w = {"equity:A": 0.3, "equity:B": 0.3, "mutual_fund:G": 0.2, "mutual_fund:D": 0.2}
    s = scen(
        BEAR, DirectShock("category", "gold", 0.10), DirectShock("category", "debt_gilt", -0.02)
    )
    r = hypothetical_shock(w, s, portfolio_value=1_000_000, exposures=EXPOSURES, groups=GROUPS)
    # 0.3(-0.29) + 0.3(-0.14) + 0.2(0.10) + 0.2(-0.02) = -0.087 - 0.042 + 0.02 - 0.004 = -0.113
    assert r.pnl == pytest.approx(-0.113)
    assert sum(i.contribution for i in r.impacts) == pytest.approx(r.pnl, abs=1e-15)
    assert sum(i.contribution_rupees for i in r.impacts) == pytest.approx(r.pnl_rupees)
    assert sum("overrides" in n for n in r.notes) == 2  # G and D had factor-implied values
    assert r.warnings == ()  # low-R^2 funds are covered by direct shocks


def test_low_r_squared_factor_shock_warns() -> None:
    r = hypothetical_shock(
        {"equity:A": 0.5, "mutual_fund:G": 0.5}, scen(BEAR), portfolio_value=1, exposures=EXPOSURES
    )
    assert by_id(r)["mutual_fund:G"].asset_return == pytest.approx(0.02 * -0.20)
    assert len(r.warnings) == 1
    assert "mutual_fund:G" in r.warnings[0] and "R^2 = 0.01" in r.warnings[0]


def test_unshocked_asset_is_zero() -> None:
    r = hypothetical_shock(
        {"equity:A": 0.5, "mutual_fund:G": 0.5},
        scen(None, DirectShock("category", "gold", 0.10)),
        portfolio_value=1,
        groups=GROUPS,
    )
    a = by_id(r)["equity:A"]
    assert (a.asset_return, a.source, a.factor_implied) == (0.0, "unshocked", None)
    assert r.pnl == pytest.approx(0.05)


def test_unmatched_direct_shock_warns() -> None:
    r = hypothetical_shock(
        {"equity:A": 1.0},
        scen(None, DirectShock("category", "gold", 0.1), DirectShock("asset", "equity:A", -0.1)),
        portfolio_value=1,
        groups=GROUPS,
    )
    assert any("category shock (gold) was not applied" in w for w in r.warnings)


def test_direct_shock_fully_overridden_by_more_specific_one_warns() -> None:
    r = hypothetical_shock(
        {"equity:A": 1.0},
        scen(
            None,
            DirectShock("sector", "Information Technology", -0.1),
            DirectShock("asset", "equity:A", -0.2),
        ),
        portfolio_value=1,
        groups=GROUPS,
    )
    assert r.pnl == pytest.approx(-0.2)
    assert any("sector shock (Information Technology) was not applied" in w for w in r.warnings)


# --- rejection ---


def test_missing_betas_rejected_when_factor_shocks_apply() -> None:
    with pytest.raises(ValueError, match="equity:Z has no factor betas"):
        hypothetical_shock({"equity:Z": 1.0}, scen(BEAR), portfolio_value=1, exposures=EXPOSURES)


def test_missing_betas_fine_when_a_direct_shock_covers_the_asset() -> None:
    r = hypothetical_shock(
        {"equity:Z": 1.0},
        scen(BEAR, DirectShock("asset", "equity:Z", -0.3)),
        portfolio_value=1,
        exposures=EXPOSURES,
    )
    assert r.pnl == pytest.approx(-0.3)


def test_missing_beta_for_a_shocked_factor_rejected() -> None:
    with pytest.raises(ValueError, match=r"no beta for shocked factor\(s\) \['WML'\]"):
        hypothetical_shock(
            {"x": 1.0},
            scen({"WML": -0.1}),
            portfolio_value=1,
            exposures={"x": AssetExposure(betas={"MKT_RF": 1.0})},
        )


def test_group_shocks_without_groups_rejected() -> None:
    with pytest.raises(ValueError, match="need groups for every asset"):
        hypothetical_shock(
            {"equity:A": 1.0}, scen(None, DirectShock("category", "gold", 0.1)), portfolio_value=1
        )


def test_invalid_weights_and_value_rejected() -> None:
    s = scen(None, DirectShock("asset", "a", 0.1))
    with pytest.raises(ValueError, match="sum to 1"):
        hypothetical_shock({"a": 0.9}, s, portfolio_value=1)
    with pytest.raises(ValueError, match="portfolio_value"):
        hypothetical_shock({"a": 1.0}, s, portfolio_value=0)
    with pytest.raises(ValueError, match="min_r_squared"):
        hypothetical_shock({"a": 1.0}, s, portfolio_value=1, min_r_squared=1.5)
