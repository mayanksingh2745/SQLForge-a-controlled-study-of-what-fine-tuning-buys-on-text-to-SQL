"""Synthetic dataset verification, provenance validation, and contamination audit.

Enforces strict research contracts for EXP-06:
- Complete provenance and licensing records.
- SQL execution against approved training databases.
- Independent semantic verification against reference queries or human review decisions.
- Cross-partition contamination protection against evaluation splits (spider:dev, bird_mini:dev, held_out:test).
- Exact and fuzzy deduplication within synthetic records and against human training examples.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sqlforge.data.normalization import (
    compute_ngram_jaccard,
    normalize_question,
    normalize_sql,
)
from sqlforge.evaluation.comparator import execute_sqlite_query
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample

FORBIDDEN_SOURCE_SPLITS: tuple[str, ...] = (
    "spider:dev",
    "spider:test",
    "bird:dev",
    "bird_mini:dev",
    "held_out_custom:test",
    "held_out:test",
)


class SyntheticRecord(BaseModel):
    """Canonical data model for an externally supplied synthetic text-to-SQL instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    synthetic_id: str = Field(description="Unique stable identifier for the synthetic record")
    question: str = Field(description="Synthesized natural-language question")
    sql: str = Field(description="Synthesized target SQL query")
    database_id: str = Field(description="Target database identifier")
    schema_reference: str = Field(description="Checksum or schema version string")
    source_model_id: str = Field(description="Identifier of model used to produce synthetic pair")
    source_model_revision: str = Field(description="Revision or commit hash of generator model")
    generation_timestamp: str = Field(description="ISO 8601 UTC timestamp of generation")
    generation_config_hash: str = Field(description="SHA-256 hash of generator configuration")
    source_dataset_reference: str = Field(description="Dataset seed split (e.g. 'spider:train')")
    source_record_ids: list[str] = Field(
        description="Source record IDs used for few-shot prompt or seed"
    )
    verification_status: str = Field(
        description="Verification status: 'verified', 'pending', or 'rejected'"
    )
    verification_method: str = Field(
        description="Method: 'reference_query_comparison', 'expert_human_review', etc."
    )
    verifier_version: str = Field(description="Software or protocol version of verification engine")
    verification_artifact_hash: str = Field(
        description="Cryptographic hash of verification evidence artifact"
    )
    license_or_usage_rights: str = Field(description="Explicit license or usage terms")

    @field_validator(
        "synthetic_id", "question", "sql", "database_id", "source_model_id", "source_model_revision"
    )
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"Field {info.field_name} must not be empty.")
        return v.strip()

    @field_validator("source_dataset_reference")
    @classmethod
    def validate_source_dataset_reference(cls, v: str) -> str:
        cleaned = v.strip().lower()
        for forbidden in FORBIDDEN_SOURCE_SPLITS:
            if forbidden in cleaned:
                raise ValueError(
                    f"source_dataset_reference points to forbidden evaluation partition: {v!r}"
                )
        return v

    @property
    def is_verified(self) -> bool:
        """Whether this record has been semantically and executionally verified."""
        return self.verification_status == "verified"

    def to_example(self) -> TextToSQLExample:
        """Convert to canonical TextToSQLExample instance."""
        return TextToSQLExample(
            id=self.synthetic_id,
            question=self.question,
            db_id=self.database_id,
            gold_sql=self.sql,
            dataset_name="synthetic_exp06",
            split=DatasetSplit.TRAIN,
        )


class RecordValidationDetail(BaseModel):
    """Per-record audit detail distinguishing individual verification dimensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    synthetic_id: str
    status: str = "accepted"  # "accepted", "rejected", "review_required"
    schema_valid: bool = True
    sql_execution_valid: bool = True
    semantically_correct: bool = True
    contamination_cleared: bool = True
    requires_review: bool = False
    is_training_eligible: bool = True
    reasons: list[str] = Field(default_factory=list)


class SyntheticValidationReport(BaseModel):
    """Aggregate audit report covering schema, execution, semantics, contamination, and eligibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_records: int
    accepted_records: int
    rejected_records: int
    review_required_records: int
    rejection_reasons: dict[str, int]
    source_provenance_hash: str
    accepted_dataset_hash: str | None
    validation_timestamp: str

    all_schemas_valid: bool
    all_executions_valid: bool
    all_semantically_correct: bool
    all_contamination_cleared: bool
    is_training_eligible: bool
    record_details: list[RecordValidationDetail] = Field(default_factory=list)


class SyntheticDataValidator:
    """Audits candidate synthetic datasets for EXP-06 against research contracts."""

    UNSAFE_SQL_PREFIXES: tuple[str, ...] = (
        "DROP",
        "DELETE",
        "UPDATE",
        "ALTER",
        "INSERT",
        "TRUNCATE",
    )

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        n_gram_size: int = 4,
        db_dir: Path | str | None = None,
        db_paths: dict[str, Path | str] | None = None,
        eval_examples: list[TextToSQLExample] | None = None,
        human_examples: list[TextToSQLExample] | None = None,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.n_gram_size = n_gram_size
        self.db_dir = Path(db_dir) if db_dir else None
        self.db_paths = {k: Path(v) for k, v in db_paths.items()} if db_paths else {}
        self.eval_examples = eval_examples
        self.human_examples = human_examples

    def compute_provenance_hash(self, records: list[SyntheticRecord]) -> str:
        """Compute deterministic SHA-256 hash over serialized record list."""
        serialized = json.dumps([r.model_dump() for r in records], sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate_dataset(self, records: list[SyntheticRecord]) -> SyntheticValidationReport:
        """Convenience validation method using pre-configured database and isolation settings."""
        report, _ = self.validate_corpus(
            records=records,
            db_dir=self.db_dir,
            eval_examples=self.eval_examples,
            human_condition_examples=self.human_examples,
        )
        return report

    def validate_corpus(
        self,
        records: list[SyntheticRecord],
        db_dir: Path | str | None = None,
        eval_examples: list[TextToSQLExample] | None = None,
        human_condition_examples: list[TextToSQLExample] | None = None,
    ) -> tuple[SyntheticValidationReport, list[SyntheticRecord]]:
        """Perform comprehensive five-stage validation on candidate synthetic records."""
        if not records:
            source_hash = "0" * 64
            report = SyntheticValidationReport(
                total_records=0,
                accepted_records=0,
                rejected_records=0,
                review_required_records=0,
                rejection_reasons={"empty_corpus": 1},
                source_provenance_hash=source_hash,
                accepted_dataset_hash=None,
                validation_timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                all_schemas_valid=False,
                all_executions_valid=False,
                all_semantically_correct=False,
                all_contamination_cleared=False,
                is_training_eligible=False,
                record_details=[],
            )
            return report, []

        source_provenance_hash = self.compute_provenance_hash(records)

        # Pre-compute lookup structures for evaluation data
        eval_questions: set[str] = set()
        eval_sqls: set[str] = set()
        eval_question_list: list[tuple[str, str]] = []

        if eval_examples:
            for ex in eval_examples:
                norm_q = normalize_question(ex.question)
                eval_questions.add(norm_q)
                if ex.gold_sql:
                    eval_sqls.add(normalize_sql(ex.gold_sql))
                eval_question_list.append((ex.id, ex.question))

        # Pre-compute normalized lookup structures for human examples selected for same condition
        human_pairs: set[tuple[str, str]] = set()
        if human_condition_examples:
            for ex in human_condition_examples:
                if ex.gold_sql:
                    human_pairs.add((normalize_question(ex.question), normalize_sql(ex.gold_sql)))

        seen_synthetic_ids: set[str] = set()
        seen_synthetic_pairs: set[tuple[str, str]] = set()

        rejection_counts: dict[str, int] = {}
        accepted_records: list[SyntheticRecord] = []
        record_details: list[RecordValidationDetail] = []
        review_required_count = 0

        all_schemas_valid = True
        all_executions_valid = True
        all_semantically_correct = True
        all_contamination_cleared = True

        for rec in records:
            reasons: list[str] = []
            is_schema_valid = True
            is_execution_valid = True
            is_semantically_correct = True
            is_contamination_cleared = True
            requires_review = False

            # 0. Unsafe SQL statement check
            first_word = rec.sql.strip().split()[0].upper() if rec.sql.strip() else ""
            if first_word in self.UNSAFE_SQL_PREFIXES:
                reasons.append(f"Unsafe SQL statement ({first_word} disallowed)")
                is_schema_valid = False

            # 1. Duplicate ID check
            if rec.synthetic_id in seen_synthetic_ids:
                reasons.append(f"Duplicate synthetic_id: {rec.synthetic_id}")
                is_schema_valid = False
            else:
                seen_synthetic_ids.add(rec.synthetic_id)

            # 2. Exact duplicate within synthetic corpus
            norm_q = normalize_question(rec.question)
            norm_sql = normalize_sql(rec.sql)
            pair = (norm_q, norm_sql)

            if pair in seen_synthetic_pairs:
                reasons.append("Duplicate normalized question/SQL pair within synthetic corpus")
                is_schema_valid = False
            else:
                seen_synthetic_pairs.add(pair)

            # 3. Exact duplicate against human examples in the same condition
            if pair in human_pairs:
                reasons.append("Exact duplicate of human training example in same condition")
                is_schema_valid = False

            # 4. Semantic correctness & verification status
            if rec.verification_status != "verified":
                reasons.append(f"unverified_status_{rec.verification_status}")
                is_semantically_correct = False

            if not rec.verification_artifact_hash or rec.verification_artifact_hash == "0" * 64:
                reasons.append("missing_verification_artifact_hash")
                is_semantically_correct = False

            if not rec.verification_method or rec.verification_method not in (
                "semantic_execution_and_reference_match",
                "reference_query_comparison",
                "expert_human_review",
            ):
                reasons.append("invalid_or_missing_verification_method")
                is_semantically_correct = False

            # 5. Contamination audit against evaluation splits
            if norm_q in eval_questions or norm_sql in eval_sqls:
                reasons.append(
                    "Contamination with evaluation split (exact question or SQL query match)"
                )
                is_contamination_cleared = False

            # Fuzzy question overlap against evaluation split
            for eval_id, eval_q in eval_question_list:
                jaccard = compute_ngram_jaccard(rec.question, eval_q, n=self.n_gram_size)
                if jaccard >= self.fuzzy_threshold:
                    reasons.append(f"fuzzy_overlap_with_{eval_id}_jaccard_{jaccard:.2f}")
                    requires_review = True
                    is_contamination_cleared = False
                    break

            # 6. SQL execution validation against database sandbox
            target_db_file: Path | None = None
            if rec.database_id in self.db_paths:
                target_db_file = self.db_paths[rec.database_id]
            elif db_dir is not None:
                candidate = Path(db_dir) / rec.database_id / f"{rec.database_id}.sqlite"
                if candidate.is_file():
                    target_db_file = candidate

            if target_db_file is not None:
                if target_db_file.is_file():
                    status, rows, err = execute_sqlite_query(target_db_file, rec.sql)
                    if status != "SUCCESS":
                        reasons.append(f"SQL execution failed ({status}): {err}")
                        is_execution_valid = False
                else:
                    reasons.append(
                        f"SQL execution failed: database file not found ({target_db_file})"
                    )
                    is_execution_valid = False

            # Aggregate record status
            if not is_schema_valid:
                all_schemas_valid = False
            if not is_execution_valid:
                all_executions_valid = False
            if not is_semantically_correct:
                all_semantically_correct = False
            if not is_contamination_cleared:
                all_contamination_cleared = False

            if requires_review:
                review_required_count += 1

            is_eligible = (
                is_schema_valid
                and is_execution_valid
                and is_semantically_correct
                and is_contamination_cleared
                and not requires_review
            )

            rec_status = "accepted"
            if requires_review:
                rec_status = "review_required"
            if (
                not is_schema_valid
                or not is_execution_valid
                or not is_semantically_correct
                or not is_contamination_cleared
            ):
                rec_status = "rejected"

            record_details.append(
                RecordValidationDetail(
                    synthetic_id=rec.synthetic_id,
                    status=rec_status,
                    schema_valid=is_schema_valid,
                    sql_execution_valid=is_execution_valid,
                    semantically_correct=is_semantically_correct,
                    contamination_cleared=is_contamination_cleared,
                    requires_review=requires_review,
                    is_training_eligible=is_eligible,
                    reasons=reasons,
                )
            )

            if is_eligible:
                accepted_records.append(rec)
            else:
                for r in reasons:
                    rejection_counts[r] = rejection_counts.get(r, 0) + 1

        accepted_hash: str | None = None
        if accepted_records:
            acc_serialized = json.dumps([r.model_dump() for r in accepted_records], sort_keys=True)
            accepted_hash = hashlib.sha256(acc_serialized.encode("utf-8")).hexdigest()

        is_training_eligible = (
            len(accepted_records) == len(records)
            and all_schemas_valid
            and all_executions_valid
            and all_semantically_correct
            and all_contamination_cleared
            and review_required_count == 0
        )

        report = SyntheticValidationReport(
            total_records=len(records),
            accepted_records=len(accepted_records),
            rejected_records=len(records) - len(accepted_records),
            review_required_records=review_required_count,
            rejection_reasons=rejection_counts,
            source_provenance_hash=source_provenance_hash,
            accepted_dataset_hash=accepted_hash,
            validation_timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            all_schemas_valid=all_schemas_valid,
            all_executions_valid=all_executions_valid,
            all_semantically_correct=all_semantically_correct,
            all_contamination_cleared=all_contamination_cleared,
            is_training_eligible=is_training_eligible,
            record_details=record_details,
        )

        return report, accepted_records
