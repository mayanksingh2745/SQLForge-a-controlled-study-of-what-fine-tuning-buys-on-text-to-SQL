"""Serving layer: Production inference contracts and service interfaces.

Planned for future steps:
- FastAPI / vLLM inference server.
- Model-agnostic request/response data contracts.
- Health checks and metric endpoints.
"""

from pydantic import BaseModel, Field


class TextToSQLRequest(BaseModel):
    """Inference API request contract."""

    question: str = Field(..., description="Natural language query")
    db_id: str = Field(..., description="Target database identifier")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)


class TextToSQLResponse(BaseModel):
    """Inference API response contract."""

    sql: str = Field(..., description="Generated candidate SQL query")
    db_id: str = Field(..., description="Target database identifier")
    model_id: str = Field(..., description="Model identifier that served the request")
    latency_ms: float = Field(..., description="Execution latency in milliseconds")
