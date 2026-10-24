"""Stress scenario files: schema, loader and universe checks.

One JSON file per scenario in data/scenarios/ (committed), named <id>.json. Each file is a
`historical` or a `hypothetical` scenario, distinguished by its "type" field.

The hypothetical schema is also the target format of the Phase 7 AI scenario parser: flat
fields, closed enums for factors, target kinds and fund categories, decimal units stated in
every description, and `extra="forbid"`, so `HypotheticalScenarioSpec.model_json_schema()` can
be used as the structured-output schema and anything malformed fails validation instead of
being half-applied. Values that depend on the universe (asset ids, sectors) cannot be enums
in a static schema; `check_against_universe` validates them.

The engine never sees these models: app/services/stress.py converts them to the dataclasses
in app/engine/stress/scenarios.py.
"""

from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from app.core.config import get_settings
from app.data.universe import FundCategory, Universe

FactorName = Literal["MKT_RF", "SMB", "HML", "WML"]
TargetKind = Literal["asset", "sector", "category"]

# (-1, 1]: a fall of 100% or more is impossible for a long position; a rise above 100% is far
# more likely a percent/decimal mix-up than a scenario. Same bound as the engine.
Shock = Annotated[
    float,
    Field(
        gt=-1.0,
        le=1.0,
        description="Simple return in decimals: -0.2 means -20%, 0.1 means +10%.",
    ),
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- shock targets ---


class AssetTarget(_Strict):
    kind: Literal["asset"]
    asset_id: str = Field(
        pattern=r"^(mutual_fund|equity|index):\S+$",
        description='Universe asset id, e.g. "equity:TCS" or "mutual_fund:118663".',
    )


class SectorTarget(_Strict):
    kind: Literal["sector"]
    sector: str = Field(
        min_length=1,
        description='NSE industry of a stock in the universe, e.g. "Information Technology".',
    )


class CategoryTarget(_Strict):
    kind: Literal["category"]
    category: FundCategory = Field(description="Fund category from the universe.")


ShockTarget = Annotated[AssetTarget | SectorTarget | CategoryTarget, Field(discriminator="kind")]


def target_key(t: AssetTarget | SectorTarget | CategoryTarget) -> tuple[TargetKind, str]:
    match t:
        case AssetTarget():
            return ("asset", t.asset_id)
        case SectorTarget():
            return ("sector", t.sector)
        case CategoryTarget():
            return ("category", t.category)


class DirectShockSpec(_Strict):
    target: ShockTarget
    shock: Shock


# --- scenarios ---


class _ScenarioBase(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_]+$", description="Unique id; also the file name.")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1, description="Why this scenario and these numbers.")
    caveats: list[str] = Field(default_factory=list)


class BenchmarkCheck(_Strict):
    """Nifty 50 closes from our cache at the window ends, recorded when the window was chosen.
    tests/services/test_stress.py re-reads the cache and fails if they no longer match."""

    symbol: Literal["^NSEI"] = "^NSEI"
    start_close: float = Field(gt=0)
    end_close: float = Field(gt=0)
    change: float = Field(description="end_close / start_close - 1, decimals, rounded to 4 dp")


class HistoricalScenarioSpec(_ScenarioBase):
    type: Literal["historical"]
    start: date = Field(description="Base close: weights are invested at this day's close.")
    end: date = Field(description="Last close of the window (inclusive).")
    benchmark_check: BenchmarkCheck

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must be before end {self.end}")
        bc = self.benchmark_check
        if abs(bc.end_close / bc.start_close - 1 - bc.change) > 1e-4:  # change is 4 dp
            raise ValueError("benchmark_check.change does not match its closes")
        return self


class HypotheticalScenarioSpec(_ScenarioBase):
    type: Literal["hypothetical"]
    factor_shocks: dict[FactorName, Shock] = Field(
        default_factory=dict,
        description="Shocks to Carhart factors, mapped to assets through their betas.",
    )
    direct_shocks: list[DirectShockSpec] = Field(
        default_factory=list,
        description="Shocks applied as-is; override the factor-implied shock of an asset. "
        "Precedence when several match: asset > sector > category.",
    )

    @model_validator(mode="after")
    def _shocks(self) -> Self:
        if not self.factor_shocks and not self.direct_shocks:
            raise ValueError("a hypothetical scenario needs at least one shock")
        keys = [target_key(d.target) for d in self.direct_shocks]
        dups = sorted({k for k in keys if keys.count(k) > 1})
        if dups:
            raise ValueError(f"duplicate direct shock targets: {dups}")
        return self


ScenarioSpec = Annotated[
    HistoricalScenarioSpec | HypotheticalScenarioSpec, Field(discriminator="type")
]
_ADAPTER: TypeAdapter[HistoricalScenarioSpec | HypotheticalScenarioSpec] = TypeAdapter(ScenarioSpec)


def parse_scenario(text: str) -> HistoricalScenarioSpec | HypotheticalScenarioSpec:
    """Validate one scenario from JSON text (raises pydantic.ValidationError)."""
    return _ADAPTER.validate_json(text)


def load_scenarios(
    directory: Path | None = None,
) -> dict[str, HistoricalScenarioSpec | HypotheticalScenarioSpec]:
    """Load every *.json in `directory` (default <DATA_DIR>/scenarios), keyed by id.

    Raises ValueError if a file's id differs from its file name (which also makes ids unique)
    or the directory has no scenarios; pydantic.ValidationError for an invalid file.
    """
    directory = directory or get_settings().scenarios_dir
    out: dict[str, HistoricalScenarioSpec | HypotheticalScenarioSpec] = {}
    for path in sorted(directory.glob("*.json")):
        spec = parse_scenario(path.read_text(encoding="utf-8"))
        if spec.id != path.stem:
            raise ValueError(f"{path.name}: id {spec.id!r} must equal the file name")
        out[spec.id] = spec
    if not out:
        raise ValueError(f"no scenario files in {directory}")
    return out


def check_against_universe(
    spec: HistoricalScenarioSpec | HypotheticalScenarioSpec, universe: Universe
) -> None:
    """Raise ValueError if a hypothetical scenario targets an asset, sector or category that is
    not in the universe (a typo would otherwise silently match nothing)."""
    if isinstance(spec, HistoricalScenarioSpec):
        return
    asset_ids = {a.id for a in universe.assets()}
    sectors = {s.industry for s in universe.stocks}
    categories = {f.category for f in universe.funds}
    known = {"asset": asset_ids, "sector": sectors, "category": categories}
    bad = [
        f"{kind} {value!r}"
        for kind, value in (target_key(d.target) for d in spec.direct_shocks)
        if value not in known[kind]
    ]
    if bad:
        raise ValueError(f"{spec.id}: targets not in the universe: {bad}")
