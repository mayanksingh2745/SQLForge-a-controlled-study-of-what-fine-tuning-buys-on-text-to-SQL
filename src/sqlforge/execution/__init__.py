"""Execution layer: Isolated sandboxes, read-only connections, timeouts, and normalization.

Planned for future steps:
- SQLite safe runner with connection-level read-only URI and query timeout interrupts.
- PostgreSQL sandbox container runner with unprivileged user role.
- Canonical result set normalizer (type casting, NULL alignment, tuple sorting).
"""

from typing import Protocol

from sqlforge.schemas.execution import ExecutionResult


class SQLExecutor(Protocol):
    """Protocol for secure, isolated database query execution."""

    def execute(self, query: str, db_id: str, timeout_seconds: float = 10.0) -> ExecutionResult:
        """Execute a single query in an isolated, read-only sandbox."""
        ...
