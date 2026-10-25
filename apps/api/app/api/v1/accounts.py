"""Auth (register, login, me) and saved-portfolio CRUD routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.deps import DbDep, UserDep
from app.core.config import Settings, get_settings
from app.schemas.auth import (
    Credentials,
    SavedPortfolioIn,
    SavedPortfolioOut,
    TokenResponse,
    UserOut,
)
from app.services import accounts

router = APIRouter()


KnownIds = Annotated[frozenset[str], Depends(accounts.known_asset_ids)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _user_out(user: object) -> UserOut:
    return UserOut.model_validate(user, from_attributes=True)


@router.post(
    "/auth/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
def register(body: Credentials, db: DbDep) -> UserOut:
    try:
        return _user_out(accounts.register(db, body))
    except accounts.EmailTaken as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: Credentials, db: DbDep, settings: SettingsDep) -> TokenResponse:
    try:
        user = accounts.authenticate(db, body)
    except accounts.BadCredentials as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    return accounts.issue_token(user, settings)


@router.get("/auth/me", response_model=UserOut, tags=["auth"])
def me(user: UserDep) -> UserOut:
    return _user_out(user)


@router.get("/portfolios", response_model=list[SavedPortfolioOut], tags=["portfolios"])
def list_portfolios(db: DbDep, user: UserDep) -> list[SavedPortfolioOut]:
    return accounts.list_portfolios(db, user)


@router.post(
    "/portfolios",
    response_model=SavedPortfolioOut,
    status_code=status.HTTP_201_CREATED,
    tags=["portfolios"],
)
def create_portfolio(
    body: SavedPortfolioIn, db: DbDep, user: UserDep, known: KnownIds
) -> SavedPortfolioOut:
    return accounts.create_portfolio(db, user, body, known)


@router.get("/portfolios/{portfolio_id}", response_model=SavedPortfolioOut, tags=["portfolios"])
def get_portfolio(portfolio_id: int, db: DbDep, user: UserDep) -> SavedPortfolioOut:
    return accounts.get_portfolio(db, user, portfolio_id)


@router.put("/portfolios/{portfolio_id}", response_model=SavedPortfolioOut, tags=["portfolios"])
def update_portfolio(
    portfolio_id: int, body: SavedPortfolioIn, db: DbDep, user: UserDep, known: KnownIds
) -> SavedPortfolioOut:
    return accounts.update_portfolio(db, user, portfolio_id, body, known)


@router.delete(
    "/portfolios/{portfolio_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["portfolios"],
)
def delete_portfolio(portfolio_id: int, db: DbDep, user: UserDep) -> Response:
    accounts.delete_portfolio(db, user, portfolio_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
