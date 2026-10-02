# ADR-003: Defensive SQL Execution and Sandboxing Architecture

## Status
Accepted

## Context
Text-to-SQL language models generate arbitrary SQL strings. When evaluating execution accuracy (EX), these queries must be run against a database engine. If executed carelessly, malicious or hallucinated queries (e.g. `DROP TABLE`, Cartesian explosion joins, infinite recursive CTEs, filesystem reads) can compromise host security or exhaust compute resources.

## Decision
1. Treat all generated SQL as completely untrusted input.
2. Explicitly reject simplistic regex filtering (e.g. `re.search("DROP|DELETE", sql)`) as inadequate for security guarantees.
3. Enforce structural sandboxing at multiple layers:
   - **Connection Mode:** Database connections must be explicitly opened in read-only mode (`mode=ro` in SQLite URI) or connect via an unprivileged database role without write grants.
   - **Single-Statement Rule:** Multi-statement queries separated by semicolons are blocked before execution to prevent stacked SQL injection payloads.
   - **Hard Timeouts:** Execution is bounded by an OS thread interrupt / wall-clock timeout (default: 10.0 seconds).
   - **Resource Quotas:** Memory allocations are capped at 1024 MB and returned result sets are limited to 5000 rows.
   - **Database Isolation:** Benchmark databases must be isolated copies; execution against production databases is strictly prohibited.

## Consequences
* **Positive:** Prevents data corruption, DoS attacks via runaway queries, and privilege escalation.
* **Negative:** Queries legitimately requiring more than 10 seconds or generating massive result sets are categorized as timeouts or truncated.
