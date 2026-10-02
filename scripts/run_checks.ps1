# PowerShell script to run code quality and test checks on Windows
$ErrorActionPreference = "Stop"

Write-Host "=== [1/4] Running Ruff Code Formatting Check ===" -ForegroundColor Cyan
ruff format --check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Formatting failed. Run 'ruff format .' to fix formatting." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "=== [2/4] Running Ruff Linter ===" -ForegroundColor Cyan
ruff check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Linting failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "=== [3/4] Running MyPy Static Type Checking ===" -ForegroundColor Cyan
mypy src/sqlforge
if ($LASTEXITCODE -ne 0) {
    Write-Host "Type checking failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "=== [4/4] Running Pytest Suite ===" -ForegroundColor Cyan
pytest -v tests/
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "=== All checks passed successfully! ===" -ForegroundColor Green
