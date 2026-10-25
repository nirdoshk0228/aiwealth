"""Write the FastAPI OpenAPI spec to apps/web/openapi.json (input of the typed web client).

The web app's API types are generated from this file (`npm run gen:api` in apps/web), never
hand-written. Run this after changing any schema or route, then regenerate the client:

    python scripts/export_openapi.py
    cd apps/web && npm run gen:api

`--check` exits 1 if the committed file is out of date (used in CI).
No market data, database or network is needed: only the app's routes and schemas are read.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("WARM_MARKET_DATA", "false")

from app.main import create_app  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "apps" / "web" / "openapi.json"


def render() -> str:
    spec = create_app().openapi()
    return json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is out of date; run python scripts/export_openapi.py")
            return 1
        print(f"{OUT} is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
