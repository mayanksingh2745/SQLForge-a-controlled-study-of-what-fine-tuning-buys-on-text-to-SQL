# Setup development environment on Windows
$ErrorActionPreference = "Stop"

Write-Host "Setting up SQLForge development environment..." -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..."
    python -m venv .venv
}

Write-Host "Activating virtual environment and installing editable package..."
& .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

Write-Host "Verifying installation with sqlforge CLI..."
sqlforge --version
sqlforge env

Write-Host "Setup complete!" -ForegroundColor Green
