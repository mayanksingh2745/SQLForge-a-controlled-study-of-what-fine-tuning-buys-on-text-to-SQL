"""BIRD (Benchmark for Large-scale Database Grounded Text-to-SQL) ingestion adapter.

Resolves research protocol ambiguity between:
1. mini_dev_500: Pinned 500-instance SELECT-only subset with external domain evidence.
2. mini_dev_780: Official BIRD Mini-Dev V2 release containing 780 instances.

Preserves natural language questions, evidence text, Gold SQL, database schemas,
and strict non-commercial licensing under CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from sqlforge.data.normalization import compute_file_sha256
from sqlforge.data.spider import parse_spider_tables
from sqlforge.schemas.dataset import DatasetProvenance, LicenseIdentifier
from sqlforge.schemas.examples import DatasetSplit, DifficultyLevel, TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata


class BirdSubsetVariant(StrEnum):
    """Supported BIRD Mini-Dev evaluation subsets."""

    MINI_DEV_500 = "mini_dev_500"
    MINI_DEV_780 = "mini_dev_780"


# Pinned reference metadata for reproducibility
BIRD_VARIANT_METADATA: dict[BirdSubsetVariant, dict[str, str | int]] = {
    BirdSubsetVariant.MINI_DEV_500: {
        "expected_count": 500,
        "description": "SQLForge canonical 500-instance SELECT-only subset with external evidence grounding.",
        "query_filter": "SELECT-only",
    },
    BirdSubsetVariant.MINI_DEV_780: {
        "expected_count": 780,
        "description": "Upstream BIRD Mini-Dev V2 release across 11 complex databases.",
        "query_filter": "All query types",
    },
}

_BIRD_DIFFICULTY_MAP: dict[str, DifficultyLevel] = {
    "simple": DifficultyLevel.EASY,
    "moderate": DifficultyLevel.MEDIUM,
    "challenging": DifficultyLevel.HARD,
}


class BIRDAdapterError(ValueError):
    """Raised when BIRD dataset files fail validation or structural contracts."""


def parse_bird_tables(tables_file: Path | str) -> dict[str, SchemaMetadata]:
    """Parse BIRD tables metadata (which follows the standard tables.json format)."""
    return parse_spider_tables(tables_file)


def load_bird_mini_dev(
    examples_file: Path | str,
    schemas: dict[str, SchemaMetadata],
    variant: BirdSubsetVariant = BirdSubsetVariant.MINI_DEV_500,
    enforce_expected_count: bool = False,
) -> list[TextToSQLExample]:
    """Ingest and strictly validate BIRD Mini-Dev examples.

    Args:
        examples_file: Path to mini_dev_sqlite.json or dev.json.
        schemas: Ingested schemas dictionary to validate db_id references.
        variant: BIRD subset variant (mini_dev_500 or mini_dev_780).
        enforce_expected_count: If True, asserts the ingested count matches variant specification.

    Returns:
        List of validated TextToSQLExample instances.

    Raises:
        FileNotFoundError: If examples_file does not exist.
        BIRDAdapterError: If JSON is malformed, required fields missing, or count mismatch.
    """
    path = Path(examples_file)
    if not path.is_file():
        raise FileNotFoundError(f"BIRD examples file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_examples = json.load(f)
    except json.JSONDecodeError as exc:
        raise BIRDAdapterError(f"Malformed JSON in BIRD examples file '{path}': {exc}") from exc

    if not isinstance(raw_examples, list):
        raise BIRDAdapterError(f"BIRD examples file '{path}' must contain a JSON list.")

    examples: list[TextToSQLExample] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_examples):
        if not isinstance(item, dict):
            raise BIRDAdapterError(f"BIRD example at index {idx} in '{path}' must be a dictionary.")

        question = item.get("question")
        gold_sql = item.get("SQL") or item.get("query")
        db_id = item.get("db_id")
        evidence = item.get("evidence")
        raw_diff = str(item.get("difficulty", "")).strip().lower()
        question_id = item.get("question_id", idx)

        if not question or not isinstance(question, str) or not question.strip():
            raise BIRDAdapterError(f"Missing or empty 'question' at index {idx} in '{path}'.")

        if not gold_sql or not isinstance(gold_sql, str) or not gold_sql.strip():
            raise BIRDAdapterError(f"Missing or empty 'SQL' at index {idx} in '{path}'.")

        if not db_id or not isinstance(db_id, str) or not db_id.strip():
            raise BIRDAdapterError(f"Missing or empty 'db_id' at index {idx} in '{path}'.")

        if db_id not in schemas:
            raise BIRDAdapterError(
                f"Unknown db_id '{db_id}' referenced at index {idx} in '{path}'. Database missing from BIRD tables."
            )

        example_id = f"bird_{variant.value}_{question_id}"
        if example_id in seen_ids:
            raise BIRDAdapterError(f"Duplicate BIRD example ID detected: '{example_id}'.")
        seen_ids.add(example_id)

        difficulty = _BIRD_DIFFICULTY_MAP.get(raw_diff, DifficultyLevel.UNCLASSIFIED)

        try:
            example = TextToSQLExample(
                id=example_id,
                question=question.strip(),
                db_id=db_id.strip(),
                gold_sql=gold_sql.strip(),
                dataset_name="bird_mini",
                split=DatasetSplit.DEV,
                difficulty=difficulty,
                evidence=evidence.strip()
                if isinstance(evidence, str) and evidence.strip()
                else None,
                schema_version="bird_mini_1.0",
            )
        except ValidationError as exc:
            raise BIRDAdapterError(
                f"Validation failed for BIRD example '{example_id}': {exc}"
            ) from exc

        examples.append(example)

    if enforce_expected_count:
        expected = BIRD_VARIANT_METADATA[variant]["expected_count"]
        if len(examples) != expected:
            raise BIRDAdapterError(
                f"BIRD {variant.value} count mismatch: expected {expected} examples, found {len(examples)}."
            )

    return examples


def create_bird_provenance(
    tables_file: Path | str,
    dev_file: Path | str,
    variant: BirdSubsetVariant = BirdSubsetVariant.MINI_DEV_500,
    revision: str | None = "v2.0",
) -> DatasetProvenance:
    """Generate a provenance record for BIRD Mini-Dev with raw file checksums."""
    checksums: dict[str, str] = {}

    for label, filepath in [
        ("dev_tables.json", tables_file),
        (f"{variant.value}.json", dev_file),
    ]:
        p = Path(filepath)
        if p.is_file():
            checksums[label] = compute_file_sha256(p)

    meta = BIRD_VARIANT_METADATA[variant]

    return DatasetProvenance(
        dataset_id="bird_mini",
        source_name="BIRD: Can Language Models Solve Extreme Complex Text-to-SQL with Large and Dirty Databases? (Li et al., NeurIPS 2023)",
        release_version=variant.value,
        revision=revision,
        license_id=LicenseIdentifier.CC_BY_NC_SA_4_0,
        license_notes=(
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). "
            "HKU & Alibaba DAMO Academy. Strict Non-Commercial research use only."
        ),
        dialect="sqlite",
        download_url="https://bird-bench.github.io/",
        raw_files_checksums=checksums,
        normalization_version="1.0.0",
        created_at=datetime.now(UTC).isoformat(),
        description=str(meta["description"]),
    )
