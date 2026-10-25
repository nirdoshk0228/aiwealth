"""Auth and saved-portfolio schemas."""

from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from app.schemas.portfolio import PortfolioIn

PASSWORD_MIN = 8
PASSWORD_MAX = 128  # bounds the work per scrypt call


def _normalise_email(v: str) -> str:
    return v.strip().lower()


Email = Annotated[
    str,
    Field(
        min_length=3,
        max_length=254,
        pattern=r"^\s*[^@\s]+@[^@\s]+\.[^@\s]+\s*$",
        examples=["investor@example.com"],
    ),
    AfterValidator(_normalise_email),
]


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: Email
    password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until the token expires.")


class UserOut(BaseModel):
    id: int
    email: str
    created_at: datetime


class SavedPortfolioIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    portfolio: PortfolioIn


class SavedPortfolioOut(BaseModel):
    id: int
    name: str
    portfolio: PortfolioIn
    created_at: datetime
    updated_at: datetime
