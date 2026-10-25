"""FastAPI dependencies: market data, database session, current user."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import InvalidToken, decode_access_token
from app.db.models import User
from app.db.session import get_db
from app.services.market import MarketData, MarketDataStore


def get_store(request: Request) -> MarketDataStore:
    return request.app.state.market_store


def get_market_data(store: Annotated[MarketDataStore, Depends(get_store)]) -> MarketData:
    """Blocks until market data is loaded; raises MarketDataUnavailable (-> 503) on failure."""
    return store.get()


_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    unauthorised = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorised
    try:
        user_id = decode_access_token(creds.credentials, settings)
    except InvalidToken:
        raise unauthorised from None
    user = db.get(User, user_id)
    if user is None:
        raise unauthorised
    return user


MarketDataDep = Annotated[MarketData, Depends(get_market_data)]
DbDep = Annotated[Session, Depends(get_db)]
UserDep = Annotated[User, Depends(get_current_user)]
