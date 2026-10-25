# Run from the repo root with the virtual environment activated.
# Windows users without make: see "Development" in README.md for the raw commands.

API_DIR := apps/api

.PHONY: install lint format test api seed migrate seed-db ingest-docs infra-up infra-down eval

install:
	cd $(API_DIR) && python -m pip install -e ".[dev]"

lint:
	cd $(API_DIR) && ruff check . && ruff format --check .

format:
	cd $(API_DIR) && ruff check --fix . && ruff format .

test:
	cd $(API_DIR) && pytest

api:
	cd $(API_DIR) && uvicorn app.main:app --reload

seed:
	python scripts/seed_data.py

migrate:
	cd $(API_DIR) && alembic upgrade head

seed-db:
	python scripts/seed_db.py

# Download, verify, chunk and embed the knowledge-base documents (add --db for Postgres).
ingest-docs:
	python scripts/ingest_docs.py --db

# Optional local Postgres (Neon is the default; see README).
infra-up:
	docker compose up -d postgres

infra-down:
	docker compose down

# Retrieval, answers, tool calls and VaR backtest -> eval/results/ (LLM calls cached in .cache/llm).
eval:
	python eval/build_golden.py
	python eval/run_eval.py --answers-mode hybrid_rerank --judge
