"""Contamination audit engine and partition isolation guards.

Performs cross-partition audits:
1. Normalized exact question string matching.
2. Normalized exact SQL query matching.
3. Configurable fuzzy word n-gram Jaccard similarity.
4. Schema and database disjointness verification (D_train ∩ D_eval = ∅).
5. Split leakage guards preventing evaluation data from entering training/retrieval corpora.

Guiding principle:
Findings are recorded and surfaced in machine-readable reports.
Official dev/test splits are NEVER silently modified or filtered.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime

from sqlforge.data.normalization import (
    compute_ngram_jaccard,
    normalize_question,
    normalize_sql,
)
from sqlforge.schemas.dataset import (
    ContaminationReport,
    DuplicateDetail,
    FuzzyOverlapDetail,
    SchemaDisjointnessViolation,
    SplitLeakageViolation,
)
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample


class LeakageContaminationError(ValueError):
    """Raised when an isolation guard detects evaluation examples inside training/retrieval data."""


class ContaminationAuditor:
    """Audits TextToSQLExample datasets across partition splits for data contamination and leakage."""

    def __init__(
        self,
        n_gram_size: int = 4,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        """Initialize the auditor with configurable n-gram parameters.

        Args:
            n_gram_size: Word n-gram size for lexical overlap calculation.
            fuzzy_threshold: Jaccard similarity threshold above which pairs are flagged.
        """
        if n_gram_size < 1:
            raise ValueError(f"n_gram_size must be >= 1, got {n_gram_size}")
        if not (0.0 <= fuzzy_threshold <= 1.0):
            raise ValueError(f"fuzzy_threshold must be in [0.0, 1.0], got {fuzzy_threshold}")

        self.n_gram_size = n_gram_size
        self.fuzzy_threshold = fuzzy_threshold

    def find_exact_question_duplicates(
        self,
        partition_a: list[TextToSQLExample],
        partition_b: list[TextToSQLExample],
    ) -> list[DuplicateDetail]:
        """Find exact normalized question collisions between two partitions."""
        duplicates: list[DuplicateDetail] = []
        # Build lookup table for partition_b: normalized_q -> list[example]
        lookup_b: dict[str, list[TextToSQLExample]] = defaultdict(list)
        for ex in partition_b:
            norm_q = normalize_question(ex.question)
            if norm_q:
                lookup_b[norm_q].append(ex)

        for ex_a in partition_a:
            norm_q = normalize_question(ex_a.question)
            if norm_q in lookup_b:
                for ex_b in lookup_b[norm_q]:
                    # Skip self-comparisons if partitions happen to overlap
                    if ex_a.id == ex_b.id:
                        continue
                    duplicates.append(
                        DuplicateDetail(
                            example_id_1=ex_a.id,
                            split_1=str(
                                ex_a.split.value
                                if isinstance(ex_a.split, DatasetSplit)
                                else ex_a.split
                            ),
                            example_id_2=ex_b.id,
                            split_2=str(
                                ex_b.split.value
                                if isinstance(ex_b.split, DatasetSplit)
                                else ex_b.split
                            ),
                            matched_content=norm_q,
                            db_id_1=ex_a.db_id,
                            db_id_2=ex_b.db_id,
                        )
                    )

        return duplicates

    def find_exact_sql_duplicates(
        self,
        partition_a: list[TextToSQLExample],
        partition_b: list[TextToSQLExample],
    ) -> list[DuplicateDetail]:
        """Find exact normalized SQL query collisions between two partitions."""
        duplicates: list[DuplicateDetail] = []
        lookup_b: dict[str, list[TextToSQLExample]] = defaultdict(list)
        for ex in partition_b:
            if ex.gold_sql:
                norm_sql = normalize_sql(ex.gold_sql)
                if norm_sql:
                    lookup_b[norm_sql].append(ex)

        for ex_a in partition_a:
            if not ex_a.gold_sql:
                continue
            norm_sql = normalize_sql(ex_a.gold_sql)
            if norm_sql in lookup_b:
                for ex_b in lookup_b[norm_sql]:
                    if ex_a.id == ex_b.id:
                        continue
                    duplicates.append(
                        DuplicateDetail(
                            example_id_1=ex_a.id,
                            split_1=str(
                                ex_a.split.value
                                if isinstance(ex_a.split, DatasetSplit)
                                else ex_a.split
                            ),
                            example_id_2=ex_b.id,
                            split_2=str(
                                ex_b.split.value
                                if isinstance(ex_b.split, DatasetSplit)
                                else ex_b.split
                            ),
                            matched_content=norm_sql,
                            db_id_1=ex_a.db_id,
                            db_id_2=ex_b.db_id,
                        )
                    )

        return duplicates

    def find_fuzzy_question_overlaps(
        self,
        partition_a: list[TextToSQLExample],
        partition_b: list[TextToSQLExample],
    ) -> list[FuzzyOverlapDetail]:
        """Find pairs between partition_a and partition_b whose word n-gram Jaccard >= fuzzy_threshold."""
        overlaps: list[FuzzyOverlapDetail] = []

        for ex_a in partition_a:
            for ex_b in partition_b:
                if ex_a.id == ex_b.id:
                    continue
                score = compute_ngram_jaccard(
                    ex_a.question,
                    ex_b.question,
                    n=self.n_gram_size,
                    char_level=False,
                )
                if score >= self.fuzzy_threshold:
                    overlaps.append(
                        FuzzyOverlapDetail(
                            example_id_1=ex_a.id,
                            split_1=str(
                                ex_a.split.value
                                if isinstance(ex_a.split, DatasetSplit)
                                else ex_a.split
                            ),
                            example_id_2=ex_b.id,
                            split_2=str(
                                ex_b.split.value
                                if isinstance(ex_b.split, DatasetSplit)
                                else ex_b.split
                            ),
                            question_1=ex_a.question,
                            question_2=ex_b.question,
                            similarity_score=round(score, 4),
                            db_id_1=ex_a.db_id,
                            db_id_2=ex_b.db_id,
                        )
                    )

        return overlaps

    def check_schema_disjointness(
        self,
        train_examples: list[TextToSQLExample],
        eval_examples: list[TextToSQLExample],
        eval_split_name: str = "eval",
    ) -> list[SchemaDisjointnessViolation]:
        """Verify D_train ∩ D_eval = ∅ (eval databases must never appear in cross-domain train)."""
        train_db_counts: dict[str, int] = defaultdict(int)
        for ex in train_examples:
            train_db_counts[ex.db_id] += 1

        eval_db_counts: dict[str, int] = defaultdict(int)
        for ex in eval_examples:
            eval_db_counts[ex.db_id] += 1

        overlapping_dbs = set(train_db_counts.keys()).intersection(eval_db_counts.keys())
        violations: list[SchemaDisjointnessViolation] = []

        for db_id in sorted(overlapping_dbs):
            violations.append(
                SchemaDisjointnessViolation(
                    db_id=db_id,
                    split_1="train",
                    split_2=eval_split_name,
                    count_split_1=train_db_counts[db_id],
                    count_split_2=eval_db_counts[db_id],
                )
            )

        return violations

    def check_split_leakage(
        self,
        train_examples: list[TextToSQLExample],
    ) -> list[SplitLeakageViolation]:
        """Detect examples tagged with non-train splits mistakenly placed inside training pool."""
        violations: list[SplitLeakageViolation] = []
        for ex in train_examples:
            split_val = ex.split.value if isinstance(ex.split, DatasetSplit) else str(ex.split)
            if split_val != DatasetSplit.TRAIN.value:
                violations.append(
                    SplitLeakageViolation(
                        example_id=ex.id,
                        intended_split=split_val,
                        found_in_split="train",
                    )
                )
        return violations

    def audit_partitions(
        self,
        train_examples: list[TextToSQLExample],
        eval_partitions: dict[str, list[TextToSQLExample]],
    ) -> ContaminationReport:
        """Run full contamination audit suite comparing train against evaluation partitions.

        Args:
            train_examples: Ingested training partition examples.
            eval_partitions: Dict mapping evaluation split name (e.g. 'dev', 'held_out') to examples.

        Returns:
            Structured ContaminationReport model.
        """
        audit_id = f"audit_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(UTC).isoformat()

        all_exact_q_dups: list[DuplicateDetail] = []
        all_exact_sql_dups: list[DuplicateDetail] = []
        all_fuzzy_overlaps: list[FuzzyOverlapDetail] = []
        all_schema_violations: list[SchemaDisjointnessViolation] = []
        total_pairs = 0

        # Check train split leakage first
        split_leakage_violations = self.check_split_leakage(train_examples)

        # Audit against each evaluation partition
        for eval_name, eval_list in eval_partitions.items():
            total_pairs += len(train_examples) * len(eval_list)

            # Exact question deduplication
            q_dups = self.find_exact_question_duplicates(train_examples, eval_list)
            all_exact_q_dups.extend(q_dups)

            # Exact SQL deduplication
            sql_dups = self.find_exact_sql_duplicates(train_examples, eval_list)
            all_exact_sql_dups.extend(sql_dups)

            # Fuzzy lexical overlap
            fuzzy = self.find_fuzzy_question_overlaps(train_examples, eval_list)
            all_fuzzy_overlaps.extend(fuzzy)

            # Schema disjointness
            schema_viols = self.check_schema_disjointness(
                train_examples, eval_list, eval_split_name=eval_name
            )
            all_schema_violations.extend(schema_viols)

        # Passed condition: no split leakage, no schema disjointness violations, no exact question duplicates
        has_critical_failures = bool(
            split_leakage_violations or all_schema_violations or all_exact_q_dups
        )
        passed = not has_critical_failures

        summary_parts: list[str] = []
        if passed:
            summary_parts.append(
                f"Contamination audit PASSED. Evaluated {total_pairs} cross-partition pairs with 0 critical violations."
            )
        else:
            summary_parts.append(
                f"Contamination audit FAILED with critical findings: "
                f"{len(split_leakage_violations)} split leakage violations, "
                f"{len(all_schema_violations)} schema disjointness violations, "
                f"{len(all_exact_q_dups)} exact question duplicates."
            )

        if all_fuzzy_overlaps:
            summary_parts.append(
                f"Notice: {len(all_fuzzy_overlaps)} fuzzy question overlaps detected (threshold >= {self.fuzzy_threshold})."
            )
        if all_exact_sql_dups:
            summary_parts.append(
                f"Notice: {len(all_exact_sql_dups)} exact SQL template matches across different databases."
            )

        return ContaminationReport(
            audit_id=audit_id,
            timestamp=timestamp,
            exact_question_duplicates=all_exact_q_dups,
            exact_sql_duplicates=all_exact_sql_dups,
            fuzzy_question_overlaps=all_fuzzy_overlaps,
            schema_disjointness_violations=all_schema_violations,
            split_leakage_violations=split_leakage_violations,
            n_gram_size=self.n_gram_size,
            fuzzy_threshold=self.fuzzy_threshold,
            total_pairs_evaluated=total_pairs,
            passed=passed,
            summary=" ".join(summary_parts),
        )


class IsolationGuard:
    """Pre-flight and runtime quarantine assertion guards to prevent data leakage."""

    FORBIDDEN_EVAL_SPLITS: frozenset[DatasetSplit] = frozenset(
        {DatasetSplit.DEV, DatasetSplit.TEST, DatasetSplit.HELD_OUT}
    )

    @classmethod
    def assert_training_isolation(
        cls,
        examples: Iterable[TextToSQLExample],
        forbidden_splits: Iterable[DatasetSplit] | None = None,
        forbidden_db_ids: Iterable[str] | None = None,
    ) -> None:
        """Enforce that no evaluation examples or evaluation databases enter the training pool.

        Args:
            examples: Iterable of training candidate examples.
            forbidden_splits: Splits that must never enter training (default: DEV, TEST, HELD_OUT).
            forbidden_db_ids: Database IDs that must never be trained on (e.g. eval-only schemas).

        Raises:
            LeakageContaminationError: If any example violates isolation constraints.
        """
        disallowed_splits = set(forbidden_splits or cls.FORBIDDEN_EVAL_SPLITS)
        disallowed_dbs = set(forbidden_db_ids or ())

        for ex in examples:
            if ex.split in disallowed_splits:
                raise LeakageContaminationError(
                    f"CRITICAL DATA LEAKAGE: Example '{ex.id}' has forbidden split '{ex.split}' "
                    f"in training partition."
                )

            if ex.db_id in disallowed_dbs:
                raise LeakageContaminationError(
                    f"CRITICAL SCHEMA LEAKAGE: Example '{ex.id}' references quarantined evaluation db_id "
                    f"'{ex.db_id}' inside training partition."
                )

    @classmethod
    def assert_retrieval_isolation(
        cls,
        retrieval_examples: Iterable[TextToSQLExample],
        forbidden_splits: Iterable[DatasetSplit] | None = None,
        forbidden_db_ids: Iterable[str] | None = None,
    ) -> None:
        """Enforce that retrieval demonstration indexes contain strictly training instances."""
        try:
            cls.assert_training_isolation(
                retrieval_examples,
                forbidden_splits=forbidden_splits,
                forbidden_db_ids=forbidden_db_ids,
            )
        except LeakageContaminationError as exc:
            raise LeakageContaminationError(
                f"RETRIEVAL INDEX CONTAMINATION: Evaluation examples detected in demonstration retrieval index: {exc}"
            ) from exc
