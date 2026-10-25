"""Optimizer and goal-planner routes (public, like the other analytics)."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import MarketDataDep, get_store
from app.schemas.planning import (
    GoalRequest,
    GoalResponse,
    OptimizeRequest,
    OptimizeResponse,
    ProfileRequest,
    ProfileResponse,
    QuestionnaireOut,
)
from app.services import planning
from app.services.market import MarketDataStore

router = APIRouter()


@router.post("/optimize", response_model=OptimizeResponse, tags=["optimizer"])
def optimize(body: OptimizeRequest, md: MarketDataDep) -> OptimizeResponse:
    """Long-only efficient frontier, minimum variance, maximum Sharpe and risk parity."""
    return planning.optimize(md, body)


@router.post("/goals/simulate", response_model=GoalResponse, tags=["goals"])
def simulate_goal(body: GoalRequest, md: MarketDataDep) -> GoalResponse:
    """Monte Carlo SIP simulation: probability of reaching a goal and the range of outcomes."""
    return planning.simulate_goal(md, body)


@router.get("/goals/questionnaire", response_model=QuestionnaireOut, tags=["goals"])
def questionnaire() -> QuestionnaireOut:
    """The risk-profile questions and the illustrative allocation bands."""
    return planning.questionnaire()


@router.post("/goals/profile", response_model=ProfileResponse, tags=["goals"])
def profile(
    body: ProfileRequest, store: Annotated[MarketDataStore, Depends(get_store)]
) -> ProfileResponse:
    """Score the answers; with a portfolio, compare its equity-oriented share with the band."""
    md = store.get() if body.portfolio is not None else None
    return planning.profile(md, body)
