"""Download the universe (data/universe.json) and the factor file into the cache (data/raw/).

This is the only entry point that calls external market data APIs; the API reads the cache.

Usage (from the repo root, venv active):
    python scripts/seed_data.py                         # everything (~36 assets + factors)
    python scripts/seed_data.py --only 118663 --only RELIANCE
    python scripts/seed_data.py --kind mutual_fund
    python scripts/seed_data.py --kind factors          # IIMA factors only (release 2025-12)
    python scripts/seed_data.py --kind factors --factor-release 2026-06

Exit code 0 if everything requested was cached, 1 otherwise.
"""

import argparse
import logging
import sys

from app.data.factor_repository import ParquetFactorRepository
from app.data.providers.iima_factors import (
    DATASET,
    DEFAULT_RELEASE,
    IimaFactorProvider,
    daily_file_name,
)
from app.data.providers.mfapi import MfapiProvider
from app.data.providers.yfinance_provider import YFinanceProvider
from app.data.repository import ParquetRepository
from app.data.seed import run_factor_seed, run_seed
from app.data.universe import load_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", action="append", metavar="CODE_OR_SYMBOL", help="repeatable; default: all"
    )
    parser.add_argument("--kind", choices=["mutual_fund", "equity", "index", "factors"])
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    parser.add_argument(
        "--factor-release",
        default=DEFAULT_RELEASE,
        metavar="YYYY-MM",
        help=f"IIMA factor release to download (default {DEFAULT_RELEASE})",
    )
    args = parser.parse_args(argv)
    if args.only and args.kind == "factors":
        parser.error("--only selects assets; it cannot be combined with --kind factors")
    try:
        daily_file_name(args.factor_release)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    ok = True
    if args.kind != "factors":
        ok = seed_assets(parser, args)
    if args.kind in (None, "factors") and not args.only:
        ok = seed_factors(args.factor_release) and ok
    return 0 if ok else 1


def seed_assets(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
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
    return report.ok


def seed_factors(release: str) -> bool:
    repo = ParquetFactorRepository(DATASET)
    print(f"\nSeeding factors {DATASET} (release {release}) into {repo.directory}")
    with IimaFactorProvider(release=release) as provider:
        result = run_factor_seed(provider, repo, dataset=DATASET, release=release)
    print(result.format_line())
    return result.ok


if __name__ == "__main__":
    sys.exit(main())
