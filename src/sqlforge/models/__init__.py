"""Model inference layer for SQLForge.

Exports:
- ModelRunner: Protocol definition for model inference engines.
- MockModelRunner: Deterministic offline mock runner for software pipeline verification.
- OpenAIRunner: Frontier API runner for gpt-4o-mini with rate limiting, retries, and budget guards.
- LocalHFModelRunner: Local Hugging Face causal LM runner with lazy-loaded dependencies.
- Exceptions: APIBudgetExceededError, APIOptInRequiredError, OpenAIPermanentError, OpenAITransientError, LocalModelDependencyError.
"""

from sqlforge.models.base import ModelRunner
from sqlforge.models.local_hf import LocalHFModelRunner, LocalModelDependencyError
from sqlforge.models.mock import MockModelError, MockModelRunner
from sqlforge.models.openai_runner import (
    APIBudgetExceededError,
    APIOptInRequiredError,
    OpenAIPermanentError,
    OpenAIRunner,
    OpenAIRunnerError,
    OpenAITransientError,
)

__all__ = [
    "APIBudgetExceededError",
    "APIOptInRequiredError",
    "LocalHFModelRunner",
    "LocalModelDependencyError",
    "MockModelError",
    "MockModelRunner",
    "ModelRunner",
    "OpenAIPermanentError",
    "OpenAIRunner",
    "OpenAIRunnerError",
    "OpenAITransientError",
]
