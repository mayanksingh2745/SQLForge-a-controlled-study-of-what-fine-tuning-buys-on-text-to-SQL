.PHONY: help install install-dev lint format typecheck test check clean

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
RUFF ?= ruff
MYPY ?= mypy

help:
	@echo "SQLForge Development Commands"
	@echo "=============================="
	@echo "make install       : Install base package"
	@echo "make install-dev   : Install package with development dependencies"
	@echo "make lint          : Run Ruff lint checks"
	@echo "make format        : Run Ruff code formatting"
	@echo "make typecheck     : Run MyPy static type checking"
	@echo "make test          : Run unit and integration tests"
	@echo "make check         : Run format, lint, typecheck, and test"
	@echo "make clean         : Clean build, cache, and test artifacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

typecheck:
	$(MYPY) src/sqlforge

test:
	$(PYTEST) -v tests/

check: format lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
