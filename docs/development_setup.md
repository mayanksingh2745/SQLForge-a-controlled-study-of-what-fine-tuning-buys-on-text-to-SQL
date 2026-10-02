# SQLForge Developer & Environment Setup Guide

This guide walks you through configuring your local development environment for SQLForge on Windows, Linux, or macOS.

---

## 1. System Requirements

* **Operating System:** Windows 10/11 (PowerShell), Ubuntu 22.04+ (or WSL2), or macOS (Sonoma+)
* **Python Runtime:** Python 3.11, 3.12, or 3.13 (64-bit)
* **Storage:** $\ge 5 \text{ GB}$ free disk space for base repository, virtual environment, and local unit test caches.
* **Memory (RAM):** $\ge 8 \text{ GB}$ physical memory recommended for local development and CLI operations.
* **GPU (Optional):** NVIDIA GPU with CUDA 12.1+ and $\ge 16 \text{ GB}$ VRAM is required for local model fine-tuning; Step 0 tooling and unit tests run completely on standard CPU hardware.

---

## 2. Quickstart Installation

### Step A: Clone the Repository
```bash
git clone https://github.com/mayanksingh2745/SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL.git
cd SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL
```

### Step B: Create and Activate Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step C: Install Package in Editable Mode
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

---

## 3. Environment Configuration

Copy the example environment template:
```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` to configure optional tokens (e.g., `WANDB_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`). **All base CLI tooling, configuration validation, and unit tests function fully offline without any API keys.**

---

## 4. CLI Commands & Verification

Verify your installation using the built-in CLI commands:

### Check Version & System Diagnostics
```bash
# Print installed version
sqlforge --version

# Display rich diagnostic table of OS, CPU, RAM, GPU, Git status, and toolchain
sqlforge env
```

### Validate YAML Configurations
```bash
# Validate all standard configs in configs/ directory
sqlforge config validate

# Validate a specific custom YAML file
sqlforge config validate --config configs/defaults.yaml
```

### Initialize a Local Experiment Run Record
```bash
# Creates artifacts/runs/<run_id>/ with frozen config snapshot & hardware metadata
sqlforge experiment init --name baseline_test --seed 42
```

---

## 5. Development Quality Checks

Before committing changes, execute the quality toolchain:

### On Windows (PowerShell):
```powershell
# Run the automated checks script
.\scripts\run_checks.ps1

# Or run individual tools:
ruff format --check .
ruff check .
mypy src/sqlforge
pytest -v tests/
```

### On Linux / macOS:
```bash
# Run via Makefile
make check

# Or individual targets:
make format
make lint
make typecheck
make test
```

---

## 6. Platform-Specific Notes & Troubleshooting

### Windows Console Character Encoding
* Legacy Windows console codepages (e.g. `cp1252`) can fail when printing unicode symbols (such as checkmarks `\u2713`). SQLForge's CLI uses ASCII-safe indicators (`[OK]`, `[FAILED]`) to guarantee crash-free terminal rendering on Windows.
* For optimal Unicode support in PowerShell, run: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`.

### Linux/WSL2 Requirement for DeepSpeed & BitsAndBytes
* While Step 0 development and CPU inference testing work seamlessly on Windows, quantized 4-bit fine-tuning using `bitsandbytes` and distributed multi-GPU training with `deepspeed` or `flash-attn` are natively supported on Linux.
* If developing on Windows for future model training steps (Steps 5–6), execute the training pipeline inside **WSL2 (Windows Subsystem for Linux)** with the NVIDIA Container Toolkit / CUDA WSL drivers enabled.
