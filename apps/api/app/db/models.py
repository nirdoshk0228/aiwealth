"""Database tables. Generic column types only, so unit tests can run on SQLite in memory;
Postgres-only features (pgvector) live in their own module and migration (Phase 8).

Two groups:
- users, portfolios, portfolio_holdings: saved portfolios behind JWT auth;
- market_assets, market_prices, factor_datasets, factor_returns: the market-data cache in
  Postgres, copied from the Parquet cache by scripts/seed_db.py. Only derived factor
  *returns* are stored, never the raw IIMA file (no stated licence, see docs/data.md).
"""

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# --- users and portfolios ---


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    portfolios: Mapped[list["Portfolio"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    portfolio_value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship(back_populates="portfolios")
    holdings: Mapped[list["PortfolioHolding"]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        order_by="PortfolioHolding.position",
    )


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"
    __table_args__ = (UniqueConstraint("portfolio_id", "asset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[str] = mapped_column(String(64))
    weight: Mapped[float] = mapped_column(Float)
    position: Mapped[int] = mapped_column(Integer)  # input order

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")


# --- market data cache ---


class MarketAsset(Base):
    """One cached price series' metadata (mirrors app.data.repository.CacheMeta)."""

    __tablename__ = "market_assets"

    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(100))
    source_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    first_date: Mapped[date] = mapped_column(Date)
    last_date: Mapped[date] = mapped_column(Date)
    rows: Mapped[int] = mapped_column(Integer)
    warnings: Mapped[list[str]] = mapped_column(JSON)


class MarketPrice(Base):
    __tablename__ = "market_prices"

    asset_id: Mapped[str] = mapped_column(
        ForeignKey("market_assets.asset_id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    price: Mapped[float] = mapped_column(Float)


class FactorDataset(Base):
    """Provenance of a factor dataset (mirrors FactorCacheMeta: release, sha256, citation)."""

    __tablename__ = "factor_datasets"

    dataset: Mapped[str] = mapped_column(String(64), primary_key=True)
    meta: Mapped[dict[str, object]] = mapped_column(JSON)


class FactorReturn(Base):
    """Daily factor returns, decimals. NULL = NA in the source (kept, never dropped)."""

    __tablename__ = "factor_returns"

    dataset: Mapped[str] = mapped_column(
        ForeignKey("factor_datasets.dataset", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    mkt_rf: Mapped[float | None] = mapped_column(Float, nullable=True)
    smb: Mapped[float | None] = mapped_column(Float, nullable=True)
    hml: Mapped[float | None] = mapped_column(Float, nullable=True)
    wml: Mapped[float | None] = mapped_column(Float, nullable=True)
    rf: Mapped[float | None] = mapped_column(Float, nullable=True)
