#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Running Ruff Code Formatting Check ==="
ruff format --check .

echo "=== [2/4] Running Ruff Linter ==="
ruff check .

echo "=== [3/4] Running MyPy Static Type Checking ==="
mypy src/sqlforge

echo "=== [4/4] Running Pytest Suite ==="
pytest -v tests/

echo "=== All checks passed successfully! ==="
