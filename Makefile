# ═══════════════════════════════════════════════════════════════
#  Makefile — Data-Driven Customer Retention Pipeline
# ═══════════════════════════════════════════════════════════════
#
#  Automates common tasks: setup, linting, formatting, testing,
#  data preprocessing, model training, and cleanup.
#
#  Usage:
#    make help     — show all available targets
#    make all      — run full quality + test pipeline
#    make pipeline — run the ML pipeline (data → model)
#
# ═══════════════════════════════════════════════════════════════

PYTHON := py -3.11

# ── Phony targets (not real files) ────────────────────────────
.PHONY: help setup format check lint isort test test-unit test-integration pipeline clean all

# Default target
help:
	@echo ============================================================
	@echo   Data-Driven Customer Retention — Makefile Targets
	@echo ============================================================
	@echo
	@echo   SETUP
	@echo     make setup          Install all dependencies via Poetry
	@echo
	@echo   CODE QUALITY
	@echo     make format         Auto-format code with Black
	@echo     make check          Check formatting without changes
	@echo     make lint           Run Flake8 linter
	@echo     make isort          Sort imports with isort
	@echo
	@echo   TESTING
	@echo     make test           Run full test suite (unit + integration)
	@echo     make test-unit      Run unit tests only
	@echo     make test-integration  Run integration tests only
	@echo
	@echo   PIPELINE
	@echo     make pipeline       Run the full ML pipeline (main.py)
	@echo
	@echo   UTILITIES
	@echo     make clean          Delete generated files (cache, plots, logs)
	@echo     make all            Run everything: format, lint, test

# ── Setup ─────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install poetry
	$(PYTHON) -m poetry install --with dev

# ── Code Quality ──────────────────────────────────────────────
format:
	$(PYTHON) -m black .

check:
	$(PYTHON) -m black --check .

lint:
	$(PYTHON) -m flake8 .

isort:
	$(PYTHON) -m isort .

# ── Testing ───────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-unit:
	$(PYTHON) -m pytest tests/ -v --tb=short --ignore=tests/test_integration.py

test-integration:
	$(PYTHON) -m pytest tests/test_integration.py -v --tb=short

# ── Pipeline ──────────────────────────────────────────────────
pipeline:
	$(PYTHON) main.py

# ── Full Run ──────────────────────────────────────────────────
all: isort format check lint test

# ── Cleanup ───────────────────────────────────────────────────
clean:
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	if exist plots rmdir /s /q plots
	if exist mlruns rmdir /s /q mlruns
	if exist pipeline.log del pipeline.log
	if exist churn_data_processed.csv del churn_data_processed.csv
	if exist model_results.csv del model_results.csv
	if exist mlflow.db del mlflow.db
