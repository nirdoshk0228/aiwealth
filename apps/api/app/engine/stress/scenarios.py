"""Scenario definitions as the engine sees them: plain frozen dataclasses, validated on creation.

Pure Python: no I/O, no imports from app.data or app.ai (enforced by tests/test_architecture.py).
Scenario files are JSON validated by Pydantic in app/data/scenarios.py; app/services/stress.py
converts them into these types, so the engine never depends on the data layer.

Two scenario types
------------------
- HistoricalScenario: replay the realised returns of [start, end] (both trading days, close
  to close) on today's weights. See app/engine/stress/historical.py.
- HypotheticalScenario: instantaneous shocks, either to factors (mapped to assets through
  their betas) or directly to an asset, an NSE sector or a fund category. See
  app/engine/stress/hypothetical.py.

Units: every shock is a **simple return in decimals** (-0.20 = a 20% fall). A shock must be
in (-1, 1]: a fall of 100% or more is impossible for a long position, and a move above +100%
is far more likely a unit error (10 meant as 10%) than a scenario. The same bound is enforced
by the Pydantic schema, so an AI-generated scenario fails there first.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from app.engine.risk.var import WEIGHT_SUM_TOL

# Factors a shock may name; the same names as app.engine.factors.regression.DEFAULT_FACTORS.
SHOCKABLE_FACTORS = ("MKT_RF", "SMB", "HML", "WML")

TargetKind = Literal["asset", "sector", "category"]
# Most specific first: an asset shock beats a sector shock, which beats a category shock.
TARGET_PRECEDENCE: tuple[TargetKind, ...] = ("asset", "sector", "category")

SHOCK_MIN = -1.0  # exclusive
SHOCK_MAX = 1.0  # inclusive


def check_shock(value: float, label: str) -> None:
    """Raise ValueError unless `value` is a finite simple return in (-1, 1]."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label}: shock must be a finite number, got {value!r}")
    if not SHOCK_MIN < value <= SHOCK_MAX:
        raise ValueError(
            f"{label}: shock must be a decimal return in (-1, 1], got {value} "
            "(-0.2 = -20%; a value like -20 is probably in percent)"
        )


# --- scenarios ---


@dataclass(frozen=True)
class HistoricalScenario:
    """Replay window. `start` is the base close (weights are set there), `end` the last close;
    both must be rows of the price frame."""

    id: str
    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario id must not be empty")
        if self.start >= self.end:
            raise ValueError(f"{self.id}: start {self.start} must be before end {self.end}")


@dataclass(frozen=True)
class DirectShock:
    """A shock applied as-is to every matching asset.

    kind = "asset": `target` is an asset id ("equity:TCS"); "sector": an NSE industry
    ("Information Technology"); "category": a fund category ("gold", "debt_gilt").
    """

    kind: TargetKind
    target: str
    shock: float

    def __post_init__(self) -> None:
        if self.kind not in TARGET_PRECEDENCE:
            raise ValueError(f"unknown target kind {self.kind!r}; use one of {TARGET_PRECEDENCE}")
        if not self.target:
            raise ValueError("direct shock target must not be empty")
        check_shock(self.shock, f"{self.kind} {self.target!r}")


@dataclass(frozen=True)
class HypotheticalScenario:
    """Factor shocks and/or direct shocks; at least one shock in total."""

    id: str
    name: str
    factor_shocks: Mapping[str, float] = field(default_factory=dict)
    direct_shocks: tuple[DirectShock, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario id must not be empty")
        if not self.factor_shocks and not self.direct_shocks:
            raise ValueError(f"{self.id}: a hypothetical scenario needs at least one shock")
        for name, value in self.factor_shocks.items():
            if name not in SHOCKABLE_FACTORS:
                raise ValueError(f"{self.id}: unknown factor {name!r}; use {SHOCKABLE_FACTORS}")
            check_shock(value, f"factor {name}")
        keys = [(d.kind, d.target) for d in self.direct_shocks]
        dups = sorted({k for k in keys if keys.count(k) > 1})
        if dups:
            raise ValueError(f"{self.id}: duplicate direct shock targets {dups}")


# --- per-asset inputs for hypothetical scenarios ---


@dataclass(frozen=True)
class AssetExposure:
    """Factor betas of one asset (from engine/factors/regression.py) and the fit's R^2.

    R^2 is used only to warn when a factor-implied shock rests on a poor fit."""

    betas: Mapping[str, float]
    r_squared: float | None = None


@dataclass(frozen=True)
class AssetGroups:
    """What sector/category shocks can match: a fund has a category, a stock a sector (its NSE
    industry), the index neither."""

    category: str | None = None
    sector: str | None = None


# --- shared validation ---


def check_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    """Weights as a float Series keyed by asset id: non-empty, unique, finite, summing to 1
    (the rule in app.engine.risk.var). Negative weights (shorts) are allowed."""
    w = pd.Series(weights, dtype=float)
    if w.empty:
        raise ValueError("weights is empty")
    if not w.index.is_unique:
        raise ValueError(f"duplicate asset ids in weights: {sorted(w.index[w.index.duplicated()])}")
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("weights must be finite")
    total = float(w.sum())
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"weights must sum to 1, got {total:.10f}")
    w.index = w.index.map(str)
    return w


def check_portfolio_value(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"portfolio_value must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"portfolio_value must be finite and > 0 (rupees), got {value}")
