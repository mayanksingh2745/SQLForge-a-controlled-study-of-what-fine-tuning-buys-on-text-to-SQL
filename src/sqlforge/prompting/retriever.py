"""Training-only BM25 demonstration retriever for few-shot prompt construction.

Enforces strict partition isolation:
1. Rejects evaluation (DEV, TEST) and OOD HELD_OUT instances before indexing.
2. Asserts IsolationGuard.assert_retrieval_isolation() across all candidate pools.
3. Provides deterministic tie-breaking and traceable demonstration records.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.data.audit import IsolationGuard
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample


class DemonstrationRecord(BaseModel):
    """Immutable audit record of a retrieved few-shot demonstration."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Unique training example identifier")
    db_id: str = Field(..., description="Database identifier of the demonstration")
    question: str = Field(..., description="Natural language question")
    gold_sql: str = Field(default="", description="Ground truth SQL query")
    score: float = Field(..., description="BM25 similarity score")
    evidence: str | None = Field(
        default=None, description="External domain knowledge/evidence if present"
    )


def tokenize_text(text: str) -> list[str]:
    """Deterministic word tokenizer for BM25 indexing and retrieval."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class BM25Retriever:
    """In-memory BM25 retrieval index strictly restricted to training partition examples."""

    def __init__(
        self,
        examples: Iterable[TextToSQLExample] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        forbidden_splits: Iterable[DatasetSplit] | None = None,
        forbidden_db_ids: Iterable[str] | None = None,
    ) -> None:
        """Initialize and index candidate training demonstrations.

        Args:
            examples: Candidate demonstration examples. Must strictly belong to DatasetSplit.TRAIN.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            forbidden_splits: Explicit partition splits forbidden from retrieval indexing.
            forbidden_db_ids: Database IDs forbidden from retrieval indexing.

        Raises:
            LeakageContaminationError: If any candidate example is from an evaluation partition
                                       or belongs to a quarantined database.
        """
        self.k1 = k1
        self.b = b
        self._examples: list[TextToSQLExample] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_term_frequencies: list[Counter[str]] = []
        self._inverted_index: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}

        if examples is not None:
            self.index(
                examples,
                forbidden_splits=forbidden_splits,
                forbidden_db_ids=forbidden_db_ids,
            )

    @property
    def corpus_size(self) -> int:
        """Total number of indexed training demonstrations."""
        return len(self._examples)

    def index(
        self,
        examples: Iterable[TextToSQLExample],
        forbidden_splits: Iterable[DatasetSplit] | None = None,
        forbidden_db_ids: Iterable[str] | None = None,
    ) -> None:
        """Index training examples into the BM25 search corpus.

        Enforces runtime isolation checks before storing any documents.
        """
        candidate_list = list(examples)

        # Enforce quarantine isolation assertions
        IsolationGuard.assert_retrieval_isolation(
            candidate_list,
            forbidden_splits=forbidden_splits,
            forbidden_db_ids=forbidden_db_ids,
        )

        self._examples = candidate_list
        num_docs = len(self._examples)
        if num_docs == 0:
            self._doc_lengths = []
            self._avg_doc_length = 0.0
            self._doc_term_frequencies = []
            self._inverted_index = {}
            self._idf = {}
            return

        doc_frequencies: Counter[str] = Counter()
        self._doc_term_frequencies = []
        self._doc_lengths = []

        for _doc_idx, ex in enumerate(self._examples):
            # Tokenize question
            tokens = tokenize_text(ex.question)
            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)

            tf = Counter(tokens)
            self._doc_term_frequencies.append(tf)

            for term in tf:
                doc_frequencies[term] += 1

        self._avg_doc_length = sum(self._doc_lengths) / num_docs if num_docs > 0 else 0.0

        # Compute Inverted Index and Okapi BM25 IDF
        self._inverted_index = {}
        self._idf = {}

        for term, df in doc_frequencies.items():
            # Standard Okapi BM25 IDF with smoothing
            idf_val = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[term] = max(idf_val, 0.0)

            # Build postings list: list of (doc_idx, count)
            postings = []
            for doc_idx, tf in enumerate(self._doc_term_frequencies):
                if term in tf:
                    postings.append((doc_idx, tf[term]))
            self._inverted_index[term] = postings

    def retrieve_demonstrations(
        self,
        query: str,
        k: int = 3,
        exclude_example_ids: set[str] | list[str] | None = None,
        same_db_only: bool = False,
        target_db_id: str | None = None,
    ) -> list[DemonstrationRecord]:
        """Retrieve top-k demonstration records scored by BM25 relevance.

        Args:
            query: Query question text.
            k: Number of demonstrations to retrieve. Must be >= 0.
            exclude_example_ids: Optional set of example IDs to exclude (prevents self-retrieval).
            same_db_only: If True, restricts retrieval to instances matching target_db_id.
            target_db_id: Target database ID for schema-scoped few-shot selection.

        Returns:
            List of DemonstrationRecord instances sorted by (-score, example_id).
        """
        if k <= 0 or self.corpus_size == 0:
            return []

        excluded = set(exclude_example_ids or ())
        query_tokens = tokenize_text(query)
        if not query_tokens:
            # Deterministic fallback when query tokens are empty: take first k non-excluded examples
            fallback_records: list[DemonstrationRecord] = []
            sorted_candidates = sorted(self._examples, key=lambda x: x.id)
            for ex in sorted_candidates:
                if ex.id in excluded:
                    continue
                if same_db_only and target_db_id and ex.db_id != target_db_id:
                    continue
                fallback_records.append(
                    DemonstrationRecord(
                        example_id=ex.id,
                        db_id=ex.db_id,
                        question=ex.question,
                        gold_sql=ex.gold_sql or "",
                        score=0.0,
                        evidence=ex.evidence,
                    )
                )
                if len(fallback_records) == k:
                    break
            return fallback_records

        scores: list[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self._inverted_index:
                continue

            idf = self._idf.get(term, 0.0)
            postings = self._inverted_index[term]

            for doc_idx, freq in postings:
                doc_len = self._doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self._avg_doc_length)
                )
                scores[doc_idx] += idf * (numerator / denominator)

        # Build candidate tuples: (-score, example_id, doc_idx) for deterministic tie-breaking
        scored_candidates: list[tuple[float, str, int]] = []
        for doc_idx, ex in enumerate(self._examples):
            if ex.id in excluded:
                continue
            if same_db_only and target_db_id and ex.db_id != target_db_id:
                continue

            score = scores[doc_idx]
            # Always allow candidates, sort by score descending, then tie-break by ID alphabetically
            scored_candidates.append((-round(score, 6), ex.id, doc_idx))

        scored_candidates.sort()

        results: list[DemonstrationRecord] = []
        for neg_score, _ex_id, doc_idx in scored_candidates[:k]:
            ex = self._examples[doc_idx]
            results.append(
                DemonstrationRecord(
                    example_id=ex.id,
                    db_id=ex.db_id,
                    question=ex.question,
                    gold_sql=ex.gold_sql or "",
                    score=round(-neg_score, 6),
                    evidence=ex.evidence,
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        k: int = 3,
        exclude_example_ids: set[str] | list[str] | None = None,
        same_db_only: bool = False,
        target_db_id: str | None = None,
    ) -> list[TextToSQLExample]:
        """Convenience wrapper returning raw TextToSQLExample instances."""
        records = self.retrieve_demonstrations(
            query=query,
            k=k,
            exclude_example_ids=exclude_example_ids,
            same_db_only=same_db_only,
            target_db_id=target_db_id,
        )
        record_ids = {r.example_id for r in records}
        ex_map = {ex.id: ex for ex in self._examples if ex.id in record_ids}
        return [ex_map[r.example_id] for r in records if r.example_id in ex_map]
