"""Deterministic dataset manifest generation, JSONL serialization, and verification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlforge.data.normalization import (
    compute_examples_cumulative_hash,
    compute_file_sha256,
)
from sqlforge.schemas.dataset import (
    DatasetManifest,
    DatasetProvenance,
    DatasetSplitManifest,
)
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata


def write_processed_dataset(
    examples: list[TextToSQLExample],
    output_path: Path | str,
) -> tuple[int, str]:
    """Serialize a list of TextToSQLExample instances to JSONL deterministically.

    Examples are deterministically sorted by `id` before writing.

    Args:
        examples: List of examples to write.
        output_path: Target path for the output .jsonl file.

    Returns:
        Tuple of (example_count, sha256_hash_of_file).
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    sorted_examples = sorted(examples, key=lambda ex: ex.id)

    with open(target, "w", encoding="utf-8") as f:
        for ex in sorted_examples:
            f.write(ex.model_dump_json() + "\n")

    file_hash = compute_file_sha256(target)
    return len(sorted_examples), file_hash


def load_examples_from_file(path: Path | str) -> list[TextToSQLExample]:
    """Load TextToSQLExample instances from either a JSONL or JSON file.

    Args:
        path: Path to .jsonl or .json file.

    Returns:
        List of TextToSQLExample instances.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is malformed or records fail schema validation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    examples: list[TextToSQLExample] = []
    if file_path.suffix.lower() == ".jsonl":
        with open(file_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    examples.append(TextToSQLExample.model_validate_json(line_str))
                except Exception as exc:
                    raise ValueError(
                        f"Failed parsing TextToSQLExample at line {idx + 1} in '{file_path}': {exc}"
                    ) from exc
    else:
        with open(file_path, encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except Exception as exc:
                raise ValueError(f"Malformed JSON in '{file_path}': {exc}") from exc
        if not isinstance(raw_data, list):
            raise ValueError(f"Expected JSON list in '{file_path}', got {type(raw_data).__name__}")
        for idx, item in enumerate(raw_data):
            try:
                examples.append(TextToSQLExample(**item))
            except Exception as exc:
                raise ValueError(
                    f"Failed parsing TextToSQLExample at index {idx} in '{file_path}': {exc}"
                ) from exc

    return examples


def build_dataset_manifest(
    dataset_id: str,
    version: str,
    splits_data: dict[DatasetSplit, tuple[list[TextToSQLExample], Path | str]],
    schemas: dict[str, SchemaMetadata],
    provenance: DatasetProvenance,
) -> DatasetManifest:
    """Construct a complete DatasetManifest for all processed splits."""
    split_manifests: dict[str, DatasetSplitManifest] = {}
    total_examples = 0

    for split_enum, (ex_list, path) in splits_data.items():
        rel_path = str(Path(path).as_posix())
        file_sha256 = compute_file_sha256(Path(path))
        examples_hash = compute_examples_cumulative_hash(ex_list)
        distinct_dbs = sorted({ex.db_id for ex in ex_list})

        manifest_entry = DatasetSplitManifest(
            split=split_enum,
            count=len(ex_list),
            sha256_hash=file_sha256,
            file_path=rel_path,
            db_ids=distinct_dbs,
            examples_hash=examples_hash,
        )
        split_manifests[split_enum.value] = manifest_entry
        total_examples += len(ex_list)

    return DatasetManifest(
        dataset_id=dataset_id,
        version=version,
        splits=split_manifests,
        schema_count=len(schemas),
        total_examples=total_examples,
        manifest_created_at=datetime.now(UTC).isoformat(),
        provenance=provenance,
    )


def save_manifest(manifest: DatasetManifest, output_file: Path | str) -> Path:
    """Save a DatasetManifest to a formatted JSON file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2) + "\n")
    return path


def verify_dataset_manifest(
    manifest_file: Path | str,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Verify that all files in a dataset manifest exist and match SHA-256 hashes.

    Args:
        manifest_file: Path to dataset_manifest.json.
        project_root: Base project root to resolve relative file paths.

    Returns:
        Dict with 'verified': bool, 'missing_files': list, 'hash_mismatches': list, 'errors': list.
    """
    m_path = Path(manifest_file)
    if not m_path.is_file():
        return {
            "verified": False,
            "errors": [f"Manifest file not found: {m_path}"],
            "missing_files": [],
            "hash_mismatches": [],
        }

    try:
        with open(m_path, encoding="utf-8") as f:
            data = json.load(f)
        manifest = DatasetManifest(**data)
    except Exception as exc:
        return {
            "verified": False,
            "errors": [f"Malformed dataset manifest: {exc}"],
            "missing_files": [],
            "hash_mismatches": [],
        }

    root = Path(project_root) if project_root else m_path.parent
    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    errors: list[str] = []

    for split_name, split_info in manifest.splits.items():
        file_path = root / split_info.file_path
        if not file_path.is_file():
            missing_files.append(split_info.file_path)
            continue

        actual_hash = compute_file_sha256(file_path)
        if actual_hash.lower() != split_info.sha256_hash.lower():
            hash_mismatches.append(
                f"Split '{split_name}': expected {split_info.sha256_hash}, got {actual_hash}"
            )

    verified = not missing_files and not hash_mismatches and not errors

    return {
        "verified": verified,
        "dataset_id": manifest.dataset_id,
        "version": manifest.version,
        "total_examples": manifest.total_examples,
        "missing_files": missing_files,
        "hash_mismatches": hash_mismatches,
        "errors": errors,
    }
