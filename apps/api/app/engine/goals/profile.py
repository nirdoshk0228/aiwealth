"""A simple risk-profile questionnaire mapped to an illustrative equity allocation band.

Pure data and scoring: no I/O. This is a teaching tool, **not** a suitability assessment:
SEBI-registered advisers must follow their own risk-profiling process, and nothing here says
what anyone should buy or sell. Every result carries that statement.

Scoring: five questions, each answer worth 0-4 points, total 0-20, mapped to five bands of
"equity-oriented share of the portfolio". One override: a horizon under three years caps the
band at "moderately conservative", because a short horizon leaves little time to recover from
an equity drawdown (the dashboard shows drawdowns of 25-35% in 2020 for equity baskets).
The band boundaries are this tool's illustrative choice, stated as such.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Option:
    id: str
    label: str
    points: int


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: tuple[Option, ...]


@dataclass(frozen=True)
class Band:
    id: str
    label: str
    min_score: int
    max_score: int
    equity_min: float  # decimal share of the portfolio in equity-oriented assets
    equity_max: float
    description: str


QUESTIONS: tuple[Question, ...] = (
    Question(
        "horizon",
        "When will you need most of this money?",
        (
            Option("lt3", "In less than 3 years", 0),
            Option("3to5", "In 3 to 5 years", 1),
            Option("5to7", "In 5 to 7 years", 2),
            Option("7to10", "In 7 to 10 years", 3),
            Option("gt10", "In more than 10 years", 4),
        ),
    ),
    Question(
        "drawdown_reaction",
        "If your investments fell 20% in a few months, what would you most likely do?",
        (
            Option("sell_all", "Sell everything to stop further losses", 0),
            Option("sell_some", "Sell part of it", 1),
            Option("wait", "Wait and do nothing", 2),
            Option("hold_review", "Hold and review the plan calmly", 3),
            Option("add", "Invest more while prices are lower", 4),
        ),
    ),
    Question(
        "income_stability",
        "How stable is your income?",
        (
            Option("none", "No regular income", 0),
            Option("unstable", "Irregular", 1),
            Option("somewhat", "Somewhat stable", 2),
            Option("stable", "Stable", 3),
            Option("very_stable", "Very stable, with savings for emergencies", 4),
        ),
    ),
    Question(
        "experience",
        "How much experience do you have with equity investments?",
        (
            Option("none", "None", 0),
            Option("little", "A little (deposits, maybe one fund)", 1),
            Option("some", "Some (a few years of funds)", 2),
            Option("good", "Good (funds and stocks through a downturn)", 3),
            Option("extensive", "Extensive", 4),
        ),
    ),
    Question(
        "goal_priority",
        "How important is reaching this goal at exactly the target amount?",
        (
            Option("essential", "Essential, it cannot be reduced or delayed", 0),
            Option("very", "Very important", 1),
            Option("important", "Important, some flexibility", 2),
            Option("flexible", "Flexible on amount or timing", 3),
            Option("aspirational", "Nice to have", 4),
        ),
    ),
)

BANDS: tuple[Band, ...] = (
    Band(
        "conservative",
        "Conservative",
        0,
        4,
        0.0,
        0.2,
        "Mostly debt and liquid funds; little exposure to equity drawdowns.",
    ),
    Band(
        "moderately_conservative",
        "Moderately conservative",
        5,
        8,
        0.2,
        0.4,
        "Debt-led, with a minority in equity-oriented assets.",
    ),
    Band(
        "balanced",
        "Balanced",
        9,
        12,
        0.4,
        0.6,
        "Roughly even between equity-oriented and other assets.",
    ),
    Band("growth", "Growth", 13, 16, 0.6, 0.8, "Equity-led, accepting deeper interim falls."),
    Band(
        "aggressive",
        "Aggressive",
        17,
        20,
        0.8,
        1.0,
        "Almost entirely equity-oriented; large drawdowns are expected along the way.",
    ),
)

SHORT_HORIZON_OPTION = "lt3"
SHORT_HORIZON_MAX_BAND = "moderately_conservative"
DISCLAIMER = (
    "Illustrative mapping used by this tool; not a suitability assessment, a SEBI risk profile "
    "or advice to buy or sell anything."
)


@dataclass(frozen=True)
class ProfileResult:
    score: int
    max_score: int
    band: Band
    capped_by_horizon: bool
    disclaimer: str


def score_answers(answers: dict[str, str]) -> ProfileResult:
    """Total points of the chosen options -> band (with the short-horizon cap).

    `answers` maps question id -> option id; every question must be answered exactly once.
    """
    by_id = {q.id: q for q in QUESTIONS}
    unknown = sorted(set(answers) - set(by_id))
    if unknown:
        raise ValueError(f"unknown questions: {unknown}")
    missing = [q.id for q in QUESTIONS if q.id not in answers]
    if missing:
        raise ValueError(f"unanswered questions: {missing}")
    total = 0
    for qid, oid in answers.items():
        options = {o.id: o for o in by_id[qid].options}
        if oid not in options:
            raise ValueError(f"{qid}: unknown option {oid!r}")
        total += options[oid].points
    band = next(b for b in BANDS if b.min_score <= total <= b.max_score)
    capped = False
    if answers["horizon"] == SHORT_HORIZON_OPTION:
        cap_index = [b.id for b in BANDS].index(SHORT_HORIZON_MAX_BAND)
        if BANDS.index(band) > cap_index:
            band, capped = BANDS[cap_index], True
    return ProfileResult(
        score=total,
        max_score=sum(max(o.points for o in q.options) for q in QUESTIONS),
        band=band,
        capped_by_horizon=capped,
        disclaimer=DISCLAIMER,
    )
