"""PortfolioIn enforces the product rules the engine deliberately does not (long-only etc.)."""

import pytest
from pydantic import ValidationError

from app.schemas.portfolio import PortfolioIn


def make(*weights: float, ids: list[str] | None = None, **extra: object) -> PortfolioIn:
    ids = ids or [f"equity:S{i}" for i in range(len(weights))]
    holdings = [{"asset_id": a, "weight": w} for a, w in zip(ids, weights, strict=True)]
    return PortfolioIn.model_validate({"holdings": holdings, **extra})


def test_valid_portfolio_keeps_order_and_default_value() -> None:
    p = make(0.6, 0.4)
    assert p.weights() == {"equity:S0": 0.6, "equity:S1": 0.4}
    assert p.portfolio_value == 1_000_000.0


def test_short_position_is_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        make(1.2, -0.2)


def test_weight_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        make(1.5, 0.0)


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match=r"sum to 1 \(100%\), got 1.200000"):
        make(0.6, 0.6)


def test_small_rounding_is_renormalised_exactly() -> None:
    # Three thirds typed as 0.3333333 sum to 0.9999999: inside the 1e-6 tolerance, then
    # renormalised so the engine's 1e-8 check passes.
    p = make(0.3333333, 0.3333333, 0.3333333)
    assert sum(p.weights().values()) == pytest.approx(1.0, abs=1e-15)
    assert p.weights()["equity:S0"] == pytest.approx(1 / 3, abs=1e-15)


def test_duplicate_assets_are_rejected() -> None:
    with pytest.raises(ValidationError, match="appear once"):
        make(0.5, 0.5, ids=["equity:TCS", "equity:TCS"])


@pytest.mark.parametrize("bad_id", ["TCS", "crypto:BTC", "equity:"])
def test_asset_id_format(bad_id: str) -> None:
    with pytest.raises(ValidationError, match="asset_id"):
        make(1.0, ids=[bad_id])


def test_unknown_fields_and_bad_value_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        make(1.0, currency="USD")
    with pytest.raises(ValidationError, match="greater than 0"):
        make(1.0, portfolio_value=0)


def test_empty_portfolio_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        PortfolioIn.model_validate({"holdings": []})
