"""Prompting layer: Schema serialization, prompt templates, few-shot retrieval, and engine.

Exports:
- SchemaSerializer, DDLSerializer, CompactPipeSerializer, JSONSchemaSerializer, get_serializer
- BM25Retriever, DemonstrationRecord
- PromptEngine, AssembledPrompt, PromptBudgetExceededError, estimate_token_count
- PromptBuilder (foundational baseline builder)
"""

from sqlforge.prompting.builder import PromptBuilder
from sqlforge.prompting.engine import (
    AssembledPrompt,
    PromptBudgetExceededError,
    PromptEngine,
    estimate_token_count,
)
from sqlforge.prompting.retriever import BM25Retriever, DemonstrationRecord
from sqlforge.prompting.serializers import (
    CompactPipeSerializer,
    DDLSerializer,
    JSONSchemaSerializer,
    SchemaSerializer,
    get_serializer,
)

__all__ = [
    "AssembledPrompt",
    "BM25Retriever",
    "CompactPipeSerializer",
    "DDLSerializer",
    "DemonstrationRecord",
    "JSONSchemaSerializer",
    "PromptBudgetExceededError",
    "PromptBuilder",
    "PromptEngine",
    "SchemaSerializer",
    "estimate_token_count",
    "get_serializer",
]
