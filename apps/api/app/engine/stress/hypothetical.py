"""Hypothetical shocks: instantaneous moves in factors or in specific assets/sectors/categories.

Pure Python/pandas: no I/O, no network, no imports from app.data or app.ai (enforced by
tests/test_architecture.py). Betas come from app/engine/factors/regression.py via the service.

Model
-----
For asset i with weight w_i:

    factor-implied   f_i = sum_k beta_ik * s_k           (k over the shocked factors only)
    applied shock    r_i = direct shock if one matches i, else f_i, else 0
    contribution     c_i = w_i * r_i,     portfolio P&L = sum_i c_i  (exact)

Direct shocks match by precedence asset > sector > category (most specific wins), and a
direct shock **overrides** the factor-implied value for that asset. The result keeps f_i and
adds a note for every override, so the replaced figure is visible.

Assumptions and limits
----------------------
- Instantaneous: the shock happens at once, with no horizon, no drift, no rebalancing.
- Linear factor model: r_i = sum_k beta_ik s_k, using betas estimated on **daily** excess
  returns. Alpha and RF are ignored (negligible for an instantaneous shock) and the residual
  e_i is set to 0, so an asset's own news is ignored. A -20% market shock is about 15 to 20
  daily standard deviations: betas from normal days need not hold in a crash (correlations
  rise), and daily betas of NAV-based funds are biased toward 0 by stale pricing.
- Factor-implied shocks mean little where the regression explains little: debt, liquid and
  gold funds have R^2 near 0 against equity factors. A warning is added when an asset's
  factor-implied shock rests on R^2 < `min_r_squared` (default 0.2); use a direct shock.
- Unshocked factors contribute 0: shocking MKT_RF alone means SMB, HML and WML are unchanged,
  which is itself a choice (in real sell-offs small caps usually fall more, i.e. SMB < 0).
"""

from collections.abc import Mapping

import pandas as pd

from app.engine.stress.result import AssetImpact, ShockSource, StressResult
from app.engine.stress.scenarios import (
    TARGET_PRECEDENCE,
    AssetExposure,
    AssetGroups,
    DirectShock,
    HypotheticalScenario,
    check_portfolio_value,
    check_weights,
)

DEFAULT_MIN_R_SQUARED = 0.2

NOTE_LINEAR = (
    "Hypothetical shock: instantaneous. Factor-implied returns = sum of beta x factor shock "
    "(linear, daily betas; alpha and the asset-specific residual are ignored)."
)


_DIRECT_SOURCE: dict[str, ShockSource] = {
    "asset": "direct_asset",
    "sector": "direct_sector",
    "category": "direct_category",
}


def _direct_index(shocks: tuple[DirectShock, ...]) -> dict[tuple[str, str], DirectShock]:
    return {(d.kind, d.target): d for d in shocks}


def _match(
    asset_id: str,
    groups: AssetGroups | None,
    direct: Mapping[tuple[str, str], DirectShock],
) -> DirectShock | None:
    """The most specific direct shock that applies to the asset, or None."""
    keys = {
        "asset": asset_id,
        "sector": groups.sector if groups else None,
        "category": groups.category if groups else None,
    }
    for kind in TARGET_PRECEDENCE:
        key = keys[kind]
        if key is not None and (kind, key) in direct:
            return direct[(kind, key)]
    return None


def _factor_implied(
    asset_id: str, exposure: AssetExposure | None, factor_shocks: Mapping[str, float]
) -> float | None:
    if not factor_shocks or exposure is None:
        return None
    missing = [k for k in factor_shocks if k not in exposure.betas]
    if missing:
        raise ValueError(f"{asset_id}: no beta for shocked factor(s) {missing}")
    return float(sum(exposure.betas[k] * s for k, s in factor_shocks.items()))


def hypothetical_shock(
    weights: Mapping[str, float] | pd.Series,
    scenario: HypotheticalScenario,
    *,
    portfolio_value: float,
    exposures: Mapping[str, AssetExposure] | None = None,
    groups: Mapping[str, AssetGroups] | None = None,
    min_r_squared: float = DEFAULT_MIN_R_SQUARED,
) -> StressResult:
    """Apply factor and direct shocks to a portfolio (see the module docstring).

    Args:
        weights: asset id -> weight, finite and summing to 1.
        scenario: factor shocks and/or direct shocks (decimal returns).
        portfolio_value: current value in rupees (> 0), for rupee P&L.
        exposures: asset id -> betas (and R^2). Needed, when the scenario has factor shocks,
            for every asset that no direct shock covers.
        groups: asset id -> category/sector. Needed for every asset when the scenario has
            sector or category shocks.
        min_r_squared: warn when a factor-implied shock rests on a fit below this R^2.

    Returns:
        StressResult with r_i, c_i = w_i r_i (fraction and rupees), the source of each r_i,
        the factor-implied value where available, notes (assumptions, every override) and
        warnings (low R^2, direct shocks that matched no asset).

    Raises:
        ValueError: invalid weights or value; missing betas or groups where they are needed.
    """
    w = check_weights(weights)
    check_portfolio_value(portfolio_value)
    if not 0 <= min_r_squared <= 1:
        raise ValueError(f"min_r_squared must be in [0, 1], got {min_r_squared}")
    exposures = exposures or {}
    direct = _direct_index(scenario.direct_shocks)
    if any(d.kind != "asset" for d in scenario.direct_shocks):
        missing = [a for a in w.index if groups is None or a not in groups]
        if missing:
            raise ValueError(
                f"{scenario.id}: sector/category shocks need groups for every asset; "
                f"missing {missing}"
            )

    notes = [NOTE_LINEAR]
    warnings: list[str] = []
    applied: set[DirectShock] = set()
    impacts: list[AssetImpact] = []
    for a in w.index:
        exposure = exposures.get(a)
        shock = _match(a, groups.get(a) if groups else None, direct)
        if shock is None and scenario.factor_shocks and exposure is None:
            raise ValueError(
                f"{scenario.id}: {a} has no factor betas and no direct shock; the factor "
                "shocks cannot be mapped to it"
            )
        implied = _factor_implied(a, exposure, scenario.factor_shocks)
        r2 = exposure.r_squared if exposure else None
        source: ShockSource
        if shock is not None:
            applied.add(shock)
            r, source = shock.shock, _DIRECT_SOURCE[shock.kind]
            if implied is not None:
                notes.append(
                    f"{a}: direct {shock.kind} shock ({shock.target}) {shock.shock:+.2%} "
                    f"overrides the factor-implied {implied:+.2%}"
                )
        elif implied is not None:
            r, source = implied, "factor"
            if r2 is not None and not r2 >= min_r_squared:
                warnings.append(
                    f"{a}: factor-implied shock {implied:+.2%} rests on R^2 = {r2:.2f} "
                    f"(< {min_r_squared:.2f}); the factors explain little of this asset, "
                    "consider a direct shock"
                )
        else:
            r, source = 0.0, "unshocked"
        c = float(w[a]) * r
        impacts.append(
            AssetImpact(
                asset_id=a,
                weight=float(w[a]),
                asset_return=r,
                contribution=c,
                contribution_rupees=c * portfolio_value,
                source=source,
                factor_implied=implied,
                r_squared=r2,
            )
        )

    for d in scenario.direct_shocks:
        if d not in applied:
            warnings.append(
                f"direct {d.kind} shock ({d.target}) was not applied: no asset in the portfolio "
                "matches it, or a more specific shock wins for every asset that does"
            )

    pnl = float(sum(i.contribution for i in impacts))
    return StressResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        scenario_type="hypothetical",
        portfolio_value=float(portfolio_value),
        pnl=pnl,
        pnl_rupees=pnl * portfolio_value,
        impacts=tuple(impacts),
        notes=tuple(notes),
        warnings=tuple(warnings),
    )
