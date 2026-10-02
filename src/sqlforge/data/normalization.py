"""Text and SQL normalization, n-gram extraction, and cryptographic hashing utilities."""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections.abc import Iterable
from pathlib import Path

from sqlforge.schemas.examples import TextToSQLExample

# Pre-compiled regexes
_WHITESPACE_RE = re.compile(r"\s+")
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)


def normalize_question(text: str) -> str:
    """Normalize a natural language question for exact deduplication.

    Steps:
    1. Strip leading and trailing whitespace.
    2. Convert to lowercase.
    3. Remove ASCII punctuation.
    4. Collapse contiguous whitespace into a single space.
    """
    cleaned = text.strip().lower()
    cleaned = cleaned.translate(_PUNCTUATION_TRANS)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_sql(sql: str) -> str:
    """Normalize a SQL query string for exact and structural deduplication.

    Steps:
    1. Strip block comments (/* ... */) and line comments (-- ...).
    2. Strip trailing semicolons and whitespace.
    3. Collapse contiguous whitespace into single spaces.
    4. Convert to lowercase for case-insensitive matching across dialect casing.
    """
    no_comments = _SQL_BLOCK_COMMENT_RE.sub(" ", sql)
    no_comments = _SQL_LINE_COMMENT_RE.sub(" ", no_comments)
    collapsed = _WHITESPACE_RE.sub(" ", no_comments).strip()
    # Strip terminal semicolons
    while collapsed.endswith(";"):
        collapsed = collapsed[:-1].strip()
    return collapsed.lower()


def extract_ngrams(tokens: list[str], n: int = 4) -> set[tuple[str, ...]]:
    """Extract token n-grams from a token sequence."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def compute_ngram_jaccard(
    text_a: str,
    text_b: str,
    n: int = 4,
    char_level: bool = False,
) -> float:
    """Calculate the n-gram Jaccard similarity coefficient between two texts.

    Args:
        text_a: First input text.
        text_b: Second input text.
        n: Size of n-grams (default 4).
        char_level: If True, uses character n-grams; otherwise uses word tokens.

    Returns:
        Float similarity score in [0.0, 1.0]. Returns 1.0 if both are identical,
        and 0.0 if either is empty or union is empty.
    """
    norm_a = normalize_question(text_a)
    norm_b = normalize_question(text_b)

    if norm_a == norm_b:
        return 1.0 if norm_a else 0.0

    if char_level:
        tokens_a = list(norm_a)
        tokens_b = list(norm_b)
    else:
        tokens_a = norm_a.split()
        tokens_b = norm_b.split()

    if not tokens_a or not tokens_b:
        return 0.0

    ngrams_a = extract_ngrams(tokens_a, n)
    ngrams_b = extract_ngrams(tokens_b, n)

    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = len(ngrams_a.intersection(ngrams_b))
    union = len(ngrams_a.union(ngrams_b))

    return float(intersection / union) if union > 0 else 0.0


def compute_file_sha256(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hexadecimal digest of a file on disk.

    Args:
        path: Path to file.
        chunk_size: Read buffer size in bytes (default 64KB).

    Returns:
        64-character lowercase hex digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Target file for SHA-256 calculation does not exist: {file_path}")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_examples_cumulative_hash(examples: Iterable[TextToSQLExample]) -> str:
    """Compute a deterministic cumulative SHA-256 digest across a set of examples.

    Sorts examples deterministically by `id` to guarantee stability regardless
    of memory or partition iteration order.
    """
    sorted_examples = sorted(examples, key=lambda ex: ex.id)
    hasher = hashlib.sha256()

    for ex in sorted_examples:
        # Include canonical attributes that define identity and data integrity
        payload = {
            "id": ex.id,
            "question": ex.question,
            "db_id": ex.db_id,
            "gold_sql": ex.gold_sql,
            "dataset_name": ex.dataset_name,
            "split": str(ex.split),
            "evidence": ex.evidence,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        hasher.update(encoded)

    return hasher.hexdigest()
