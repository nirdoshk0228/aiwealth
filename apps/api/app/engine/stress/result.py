"""Result type shared by historical replays and hypothetical shocks.

Sign convention: P&L is signed, as a return (-0.15 = the portfolio loses 15% of its value).
This differs on purpose from VaR and max drawdown, which are positive loss numbers: a stress
result can be a gain (gold in a crash), and the sign must survive into the AI layer. The
in-window max drawdown of a replay follows the metrics.py convention (positive).

Every per-asset contribution is w_i * r_i, and the portfolio P&L is sum_i w_i r_i, so the
contributions add up to the total exactly (floating-point precision), for both scenario types.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

ShockSource = Literal[
    "historical",  # realised window return (replay)
    "direct_asset",
    "direct_sector",
    "direct_category",
    "factor",  # sum_k beta_k * shock_k
    "unshocked",  # no direct shock matched and no factor shock applies: 0
]


@dataclass(frozen=True)
class AssetImpact:
    asset_id: str
    weight: float
    asset_return: float  # simple return applied to the asset (decimal)
    contribution: float  # weight * asset_return, fraction of portfolio value
    contribution_rupees: float  # contribution * portfolio_value
    source: ShockSource
    # Hypothetical only: the factor-implied return, kept even when a direct shock overrode it
    # (None if there were no factor shocks or the asset had no betas), and the fit's R^2.
    factor_implied: float | None = None
    r_squared: float | None = None


@dataclass(frozen=True)
class WindowDrawdown:
    """Largest peak-to-trough fall of the buy-and-hold portfolio value inside a replay window.

    `max_drawdown` is positive (0.2 = 20% below the running peak). The starting value counts as
    a peak. Dates are None when the value never fell below a previous peak."""

    max_drawdown: float
    peak_date: date | None
    trough_date: date | None


@dataclass(frozen=True)
class StressResult:
    scenario_id: str
    scenario_name: str
    scenario_type: Literal["historical", "hypothetical"]
    portfolio_value: float  # rupees
    pnl: float  # signed fraction of portfolio_value; = sum of contributions
    pnl_rupees: float  # pnl * portfolio_value
    impacts: tuple[AssetImpact, ...]  # in the order of the weights
    notes: tuple[str, ...]  # assumptions and overrides, always shown with the result
    warnings: tuple[str, ...]  # things that make a figure less reliable
    # Historical only:
    start: date | None = None
    end: date | None = None
    sessions: int | None = None  # return days in the window (rows - 1)
    drawdown: WindowDrawdown | None = None

    def worst_contributors(self, n: int = 5) -> tuple[AssetImpact, ...]:
        """Up to n assets with the most negative contributions, worst first.

        Only losses are listed (contribution < 0), so a scenario where everything gains returns
        an empty tuple. Ties are broken by asset id, so the order is deterministic."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        losers = [i for i in self.impacts if i.contribution < 0]
        return tuple(sorted(losers, key=lambda i: (i.contribution, i.asset_id))[:n])

    def to_frame(self) -> pd.DataFrame:
        """One row per asset (index = asset id)."""
        return pd.DataFrame([vars(i) for i in self.impacts]).set_index("asset_id")
