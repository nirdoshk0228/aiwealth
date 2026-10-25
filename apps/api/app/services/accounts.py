"""Users and saved portfolios (SQLAlchemy). A user can only see and change their own rows."""

from collections.abc import Collection
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import create_access_token, hash_password, verify_password
from app.data.universe import load_universe
from app.db.models import Portfolio, PortfolioHolding, User
from app.schemas.auth import Credentials, SavedPortfolioIn, SavedPortfolioOut, TokenResponse
from app.schemas.portfolio import Holding, PortfolioIn
from app.services.errors import AnalyticsInputError, NotFound


class EmailTaken(ValueError):
    pass


class BadCredentials(ValueError):
    pass


# Computed once so a login for an unknown email costs one scrypt too (no timing oracle).
_DUMMY_HASH = hash_password("not-a-real-password")


def register(db: Session, creds: Credentials) -> User:
    user = User(email=creds.email, password_hash=hash_password(creds.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise EmailTaken("an account with this email already exists") from exc
    db.refresh(user)
    return user


def authenticate(db: Session, creds: Credentials) -> User:
    user = db.scalar(select(User).where(User.email == creds.email))
    if user is None:
        verify_password(creds.password, _DUMMY_HASH)
        raise BadCredentials("incorrect email or password")
    if not verify_password(creds.password, user.password_hash):
        raise BadCredentials("incorrect email or password")
    return user


def issue_token(user: User, settings: Settings) -> TokenResponse:
    token, expires_in = create_access_token(user.id, settings)
    return TokenResponse(access_token=token, expires_in=expires_in)


# --- saved portfolios ---


@lru_cache
def known_asset_ids() -> frozenset[str]:
    """Every asset id in data/universe.json (saved portfolios may only use these)."""
    return frozenset(a.id for a in load_universe().assets())


def _check_ids(portfolio: PortfolioIn, known_ids: Collection[str]) -> None:
    unknown = [h.asset_id for h in portfolio.holdings if h.asset_id not in known_ids]
    if unknown:
        raise AnalyticsInputError(f"assets not in the universe: {unknown}")


def _out(p: Portfolio) -> SavedPortfolioOut:
    return SavedPortfolioOut(
        id=p.id,
        name=p.name,
        portfolio=PortfolioIn(
            portfolio_value=p.portfolio_value,
            holdings=[Holding(asset_id=h.asset_id, weight=h.weight) for h in p.holdings],
        ),
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


def _holdings(portfolio: PortfolioIn) -> list[PortfolioHolding]:
    return [
        PortfolioHolding(asset_id=h.asset_id, weight=h.weight, position=i)
        for i, h in enumerate(portfolio.holdings)
    ]


def _get_owned(db: Session, user: User, portfolio_id: int) -> Portfolio:
    p = db.get(Portfolio, portfolio_id)
    if p is None or p.user_id != user.id:  # someone else's portfolio looks like a missing one
        raise NotFound(f"portfolio {portfolio_id} not found")
    return p


def list_portfolios(db: Session, user: User) -> list[SavedPortfolioOut]:
    rows = db.scalars(
        select(Portfolio).where(Portfolio.user_id == user.id).order_by(Portfolio.id)
    ).all()
    return [_out(p) for p in rows]


def create_portfolio(
    db: Session, user: User, body: SavedPortfolioIn, known_ids: Collection[str]
) -> SavedPortfolioOut:
    _check_ids(body.portfolio, known_ids)
    p = Portfolio(
        user_id=user.id,
        name=body.name,
        portfolio_value=body.portfolio.portfolio_value,
        holdings=_holdings(body.portfolio),
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _out(p)


def get_portfolio(db: Session, user: User, portfolio_id: int) -> SavedPortfolioOut:
    return _out(_get_owned(db, user, portfolio_id))


def update_portfolio(
    db: Session, user: User, portfolio_id: int, body: SavedPortfolioIn, known_ids: Collection[str]
) -> SavedPortfolioOut:
    _check_ids(body.portfolio, known_ids)
    p = _get_owned(db, user, portfolio_id)
    p.name = body.name
    p.portfolio_value = body.portfolio.portfolio_value
    p.holdings.clear()
    db.flush()  # delete the old rows before inserting new ones (unique portfolio_id, asset_id)
    p.holdings.extend(_holdings(body.portfolio))
    db.commit()
    db.refresh(p)
    return _out(p)


def delete_portfolio(db: Session, user: User, portfolio_id: int) -> None:
    p = _get_owned(db, user, portfolio_id)
    db.delete(p)
    db.commit()
