import pytest

from app.engine.goals.profile import BANDS, QUESTIONS, score_answers


def answers(points: int) -> dict[str, str]:
    """Pick the option worth `points` in every question."""
    return {q.id: next(o.id for o in q.options if o.points == points) for q in QUESTIONS}


def test_bands_cover_every_score_once() -> None:
    covered = [s for b in BANDS for s in range(b.min_score, b.max_score + 1)]
    assert covered == list(range(0, 21))
    for lo, hi in zip(BANDS, BANDS[1:], strict=False):
        assert lo.equity_max == hi.equity_min


def test_extremes_and_middle() -> None:
    assert score_answers(answers(0)).band.id == "conservative"
    top = score_answers(answers(4))
    assert (top.score, top.max_score, top.band.id) == (20, 20, "aggressive")
    assert score_answers(answers(2)).band.id == "balanced"  # 10 points
    assert "not a suitability assessment" in top.disclaimer


def test_short_horizon_caps_the_band() -> None:
    a = answers(4) | {"horizon": "lt3"}  # 16 points -> growth, capped
    r = score_answers(a)
    assert r.score == 16
    assert r.band.id == "moderately_conservative" and r.capped_by_horizon
    low = answers(0)  # already below the cap: unchanged
    assert not score_answers(low).capped_by_horizon


def test_invalid_answers() -> None:
    with pytest.raises(ValueError, match="unanswered"):
        score_answers({"horizon": "gt10"})
    with pytest.raises(ValueError, match="unknown option"):
        score_answers(answers(2) | {"horizon": "forever"})
    with pytest.raises(ValueError, match="unknown questions"):
        score_answers(answers(2) | {"age": "30"})
