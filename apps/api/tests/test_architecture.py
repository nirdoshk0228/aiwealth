"""Import-boundary checks that turn CLAUDE.md architecture rules 1, 3 and 4 into CI failures.

Static (AST) scan of every module under app/, so a forbidden import fails even if the code
path is never executed in tests. Relative imports are resolved to absolute module names.
"""

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"

NETWORK_LIBS = ("httpx", "yfinance", "requests", "curl_cffi", "urllib.request", "socket")
# Only these modules may import providers (rule 4: providers run from the seed script only).
PROVIDER_IMPORTERS_ALLOWED = ("app.data.providers", "app.data.seed")


def module_name(path: Path) -> str:
    rel = path.relative_to(APP_DIR.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules(path: Path, mod: str | None = None) -> set[str]:
    """Absolute names imported by `path`; `mod` overrides its dotted name (for tests)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mod = mod or module_name(path)
    package = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against the current package
                base_parts = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join(base_parts)
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


ALL_MODULES = sorted(APP_DIR.rglob("*.py"))


def violations(files: list[Path], forbidden: tuple[str, ...]) -> list[str]:
    return [
        f"{module_name(f)} imports {name}"
        for f in files
        for name in sorted(imported_modules(f))
        if matches(name, forbidden)
    ]


def files_in(*packages: str) -> list[Path]:
    return [f for f in ALL_MODULES if matches(module_name(f), packages)]


def test_engine_has_no_ai_data_or_network_imports() -> None:
    # The engine works on pandas objects only; loading data is the service layer's job.
    forbidden = ("app.ai", "app.data", *NETWORK_LIBS)
    assert violations(files_in("app.engine"), forbidden) == []


def test_engine_imports_no_other_app_layer_or_framework() -> None:
    # Pure functions: no web framework, ORM, schemas, settings or services.
    forbidden = (
        "app.api",
        "app.services",
        "app.db",
        "app.schemas",
        "app.core",
        "fastapi",
        "sqlalchemy",
        "pydantic",
    )
    assert violations(files_in("app.engine"), forbidden) == []


def test_routers_go_through_services() -> None:
    # Rule 3: router -> service -> engine/data. Routers never call the engine or the data
    # layer directly. (The chat router calls app.ai, which sits above the services.)
    assert violations(files_in("app.api"), ("app.engine", "app.data")) == []


def test_schemas_are_plain_models() -> None:
    # Schemas describe I/O; they may reuse the scenario spec models in app.data.scenarios but
    # never call the engine, services, routers or the database.
    forbidden = ("app.engine", "app.services", "app.api", "app.db", "app.ai")
    assert violations(files_in("app.schemas"), forbidden) == []


def test_services_do_not_import_routers() -> None:
    assert violations(files_in("app.services", "app.db", "app.data"), ("app.api",)) == []


def test_ai_layer_reaches_the_engine_only_through_tools() -> None:
    # Rule 2: app/ai/agent/tools.py is the only bridge from the AI layer to the engine (and
    # it goes through services). No AI module calls a provider.
    others = [f for f in files_in("app.ai") if module_name(f) != "app.ai.agent.tools"]
    assert violations(others, ("app.engine",)) == []
    assert violations(files_in("app.ai"), ("app.data.providers", "app.data.seed")) == []


def test_lower_layers_never_import_the_ai_layer() -> None:
    # Layering: api -> ai -> services -> engine/data. Nothing below the AI layer imports it.
    lower = files_in("app.engine", "app.data", "app.db", "app.schemas", "app.services")
    assert violations(lower, ("app.ai",)) == []


def test_only_the_seed_path_imports_providers() -> None:
    others = [f for f in ALL_MODULES if not matches(module_name(f), PROVIDER_IMPORTERS_ALLOWED)]
    assert violations(others, ("app.data.providers", "app.data.seed")) == []


def test_api_layer_does_not_import_network_libraries() -> None:
    api_files = [f for f in ALL_MODULES if matches(module_name(f), ("app.api",))]
    assert violations(api_files, NETWORK_LIBS) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from app.ai import agent", "app.ai"),
        ("import yfinance as yf", "yfinance"),
        ("from ..data.providers import mfapi", "app.data.providers"),
        ("from app.data.repository import ParquetRepository", "app.data"),
    ],
)
def test_scanner_detects_forbidden_imports(tmp_path: Path, source: str, expected: str) -> None:
    # Guard against a scanner bug silently passing everything: scan a fake engine module
    # (written to tmp_path, never into the source tree).
    fake = tmp_path / "planted.py"
    fake.write_text(source + "\n", encoding="utf-8")
    found = imported_modules(fake, mod="app.engine.planted")
    assert any(matches(name, (expected,)) for name in found)
