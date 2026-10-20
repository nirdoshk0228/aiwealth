# Run from the repo root with the virtual environment activated.
# Windows users without make: see "Development" in README.md for the raw commands.

API_DIR := apps/api

.PHONY: install lint format test api seed infra-up infra-down eval

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

infra-up:
	docker compose up -d postgres redis

infra-down:
	docker compose down

eval:
	@echo "Evaluation suite not implemented yet (Phase 9)."
