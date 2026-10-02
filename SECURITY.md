# Security Policy

## Overview

SQLForge is an empirical research framework studying fine-tuning and inference for text-to-SQL systems. By nature, text-to-SQL models generate arbitrary SQL queries from natural language. **Generated SQL queries must always be treated as untrusted, potentially malicious input.**

## Core Security Boundaries

### 1. SQL Execution Isolation
* **Sandbox Execution Only:** Never execute generated SQL against production, sensitive, or mission-critical databases.
* **Ephemeral and Read-Only:** Execution environments must run with read-only credentials or against isolated, ephemeral SQLite / PostgreSQL container instances.
* **Strict Timeouts and Resource Quotas:** Every query execution must have hard wall-clock timeouts (default: 10 seconds) and memory limits to prevent denial-of-service via runaway queries (e.g., accidental Cartesian products).
* **Single-Statement Policy:** Multi-statement queries separated by semicolons (e.g., `SELECT ...; DROP TABLE ...;`) are rejected before execution.
* **No Simplistic Regex Claims:** Text filtering alone does not guarantee security. Sandboxing at the engine/permission level is the primary defense.

### 2. Secrets and Credential Protection
* **No Hardcoded Secrets:** API keys, database credentials, and Hugging Face tokens must never be committed to Git.
* **Environment-Based Config:** Use `.env` (gitignored) and environment variables for local credentials.
* **Pre-commit Scanning:** Pre-commit hooks check for committed private keys and credential patterns.

### 3. Data Governance and Privacy
* Datasets used in benchmarks (Spider, BIRD, custom schemas) must be verified for license compliance and absence of personally identifiable information (PII).

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within SQLForge (such as a sandbox escape, unsafe query execution bypass, or credential leak):

1. **Do not open a public issue.**
2. Email the maintainer at `mayanksingh2745@gmail.com` with:
   - Description of the vulnerability
   - Proof-of-concept steps or code
   - Potential impact
3. We will acknowledge receipt within 48 hours and work on a resolution.
