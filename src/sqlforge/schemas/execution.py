"""Data contracts for SQL query execution and database sandbox outcomes."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStatus(StrEnum):
    """Execution status outcomes from the database sandbox."""

    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_ERROR = "schema_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    PERMISSION_DENIED = "permission_denied"
    RUNTIME_ERROR = "runtime_error"


class ExecutionResult(BaseModel):
    """Outcome of safely executing a single SQL query in the isolated database sandbox."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="The exact SQL statement that was executed")
    status: ExecutionStatus = Field(..., description="Execution outcome classification")
    error_message: str | None = Field(
        default=None, description="Database engine error trace or reason for failure"
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Elapsed wall-clock execution time in milliseconds"
    )
    row_count: int = Field(default=0, ge=0, description="Number of rows returned by query")
    columns: list[str] = Field(
        default_factory=list, description="Column names returned in the result set"
    )
    result_fingerprint: str | None = Field(
        default=None,
        description="Deterministic cryptographic hash of the canonical sorted result set for fast equivalence checks",
    )
    truncated: bool = Field(default=False, description="True if output exceeded max row limits")
