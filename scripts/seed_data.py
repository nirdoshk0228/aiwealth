"""Download the universe (data/universe.json) into the Parquet cache (data/raw/).

This is the only entry point that calls external market data APIs; the API reads the cache.

Usage (from the repo root, venv active):
    python scripts/seed_data.py                         # everything (~36 assets, ~1 min)
    python scripts/seed_data.py --only 118663 --only RELIANCE
    python scripts/seed_data.py --kind mutual_fund

Exit code 0 if every requested asset was cached, 1 otherwise.
"""

import argparse
import logging
import sys

from app.data.providers.mfapi import MfapiProvider
from app.data.providers.yfinance_provider import YFinanceProvider
from app.data.repository import ParquetRepository
from app.data.seed import run_seed
from app.data.universe import load_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", action="append", metavar="CODE_OR_SYMBOL", help="repeatable; default: all"
    )
    parser.add_argument("--kind", choices=["mutual_fund", "equity", "index"])
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    universe = load_universe()
    try:
        assets = [universe.find(k) for k in args.only] if args.only else universe.assets()
    except KeyError as exc:
        parser.error(str(exc))
    if args.kind:
        assets = [a for a in assets if a.kind == args.kind]

    repo = ParquetRepository()
    print(f"Seeding {len(assets)} asset(s) into {repo.root}")
    yf_provider = YFinanceProvider()
    with MfapiProvider() as mf_provider:
        report = run_seed(
            assets,
            {"mutual_fund": mf_provider, "equity": yf_provider, "index": yf_provider},
            repo,
            pause=args.pause,
        )
    print()
    print(report.format_table())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
