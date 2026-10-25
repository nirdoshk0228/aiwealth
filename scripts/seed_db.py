"""Copy the Parquet cache (data/raw/) into Postgres, the market-data source of the deployed API.

The Parquet cache stays the seed source: run scripts/seed_data.py first to download, then this
script to copy. Metadata (download time, warnings, provenance) is copied unchanged. Only derived
factor returns are copied, never the raw IIMA file (no stated licence; docs/data.md).

Usage (from the repo root, venv active, DATABASE_URL set, migrations applied):
    python scripts/seed_db.py
    python scripts/seed_db.py --no-factors

Exit code 0 on success. Never prints the database URL.
"""

import argparse
import sys
import time

from app.data.factor_repository import ParquetFactorRepository
from app.data.models import DEFAULT_FACTOR_DATASET
from app.data.repository import ParquetRepository
from app.data.sql_repository import SqlFactorRepository, SqlSeriesRepository, copy_parquet_to_sql
from app.db.session import DatabaseNotConfigured, get_engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-factors", action="store_true", help="copy price series only")
    args = parser.parse_args(argv)

    source = ParquetRepository()
    if not source.list_cached():
        print(f"no cached series under {source.root}; run scripts/seed_data.py first")
        return 1
    try:
        engine = get_engine()
    except DatabaseNotConfigured:
        print("DATABASE_URL is not set")
        return 1

    factor_source = None if args.no_factors else ParquetFactorRepository(DEFAULT_FACTOR_DATASET)
    factor_target = None if args.no_factors else SqlFactorRepository(engine, DEFAULT_FACTOR_DATASET)
    started = time.perf_counter()
    report = copy_parquet_to_sql(
        source,
        SqlSeriesRepository(engine),
        factor_source,
        factor_target,
        progress=lambda msg: print(f"  {msg}", flush=True),
    )
    print(
        f"copied {report.assets} series ({report.price_rows:,} price rows) and "
        f"{report.factor_rows:,} factor rows in {time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
