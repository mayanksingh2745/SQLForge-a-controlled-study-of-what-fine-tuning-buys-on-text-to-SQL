"""Spider 1.0 benchmark dataset ingestion adapter.

Parses canonical Spider 1.0 tables.json schema definitions and train/dev JSON files,
validating foreign keys, schema integrity, duplicate instances, and required fields.
Preserves Yale LILY provenance under CC BY-SA 4.0.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from sqlforge.data.normalization import compute_file_sha256
from sqlforge.schemas.dataset import DatasetProvenance, LicenseIdentifier
from sqlforge.schemas.examples import DatasetSplit, DifficultyLevel, TextToSQLExample
from sqlforge.schemas.metadata import (
    ColumnMetadata,
    ForeignKeyMetadata,
    SchemaMetadata,
    TableMetadata,
)


class SpiderAdapterError(ValueError):
    """Raised when Spider dataset files fail validation or structural contracts."""


def parse_spider_tables(tables_file: Path | str) -> dict[str, SchemaMetadata]:
    """Parse Spider tables.json into a dictionary of SchemaMetadata keyed by db_id.

    Args:
        tables_file: Path to Spider tables.json.

    Returns:
        Mapping of db_id -> SchemaMetadata.

    Raises:
        FileNotFoundError: If tables_file does not exist.
        SpiderAdapterError: If JSON is malformed or schema structure is invalid.
    """
    path = Path(tables_file)
    if not path.is_file():
        raise FileNotFoundError(f"Spider tables file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_schemas = json.load(f)
    except json.JSONDecodeError as exc:
        raise SpiderAdapterError(f"Malformed JSON in Spider tables file '{path}': {exc}") from exc

    if not isinstance(raw_schemas, list):
        raise SpiderAdapterError(f"Spider tables file '{path}' must contain a JSON list.")

    schemas: dict[str, SchemaMetadata] = {}

    for idx, raw_db in enumerate(raw_schemas):
        if not isinstance(raw_db, dict):
            raise SpiderAdapterError(f"Database item at index {idx} must be a dictionary.")

        db_id = raw_db.get("db_id")
        if not db_id or not isinstance(db_id, str):
            raise SpiderAdapterError(f"Missing or invalid db_id at schema index {idx}.")

        table_names = raw_db.get("table_names_original") or raw_db.get("table_names")
        if not isinstance(table_names, list) or not table_names:
            raise SpiderAdapterError(f"Database '{db_id}' has no tables defined.")

        col_names = raw_db.get("column_names_original") or raw_db.get("column_names")
        col_types = raw_db.get("column_types")
        primary_keys = set(raw_db.get("primary_keys", []))
        raw_foreign_keys = raw_db.get("foreign_keys", [])

        if not isinstance(col_names, list) or not isinstance(col_types, list):
            raise SpiderAdapterError(f"Database '{db_id}' has malformed column specifications.")

        if len(col_names) != len(col_types):
            raise SpiderAdapterError(
                f"Database '{db_id}' column count mismatch: {len(col_names)} names vs {len(col_types)} types."
            )

        # Build column lookup: col_idx -> (table_idx, col_name, col_type)
        col_lookup: dict[int, tuple[int, str, str]] = {}
        for c_idx, (tbl_idx, c_name) in enumerate(col_names):
            c_type = col_types[c_idx] if c_idx < len(col_types) else "TEXT"
            col_lookup[c_idx] = (tbl_idx, c_name, str(c_type).upper())

        # Resolve foreign keys
        foreign_keys: list[ForeignKeyMetadata] = []
        # Mapping for column-level foreign key notation: (tbl_idx, col_name) -> (target_tbl, target_col)
        fk_map: dict[tuple[int, str], tuple[str, str]] = {}

        for fk_pair in raw_foreign_keys:
            if not isinstance(fk_pair, list | tuple) or len(fk_pair) != 2:
                continue
            src_cidx, tgt_cidx = fk_pair
            if src_cidx not in col_lookup or tgt_cidx not in col_lookup:
                continue

            src_tidx, src_cname, _ = col_lookup[src_cidx]
            tgt_tidx, tgt_cname, _ = col_lookup[tgt_cidx]

            if 0 <= src_tidx < len(table_names) and 0 <= tgt_tidx < len(table_names):
                src_tbl = table_names[src_tidx]
                tgt_tbl = table_names[tgt_tidx]
                foreign_keys.append(
                    ForeignKeyMetadata(
                        source_table=src_tbl,
                        source_column=src_cname,
                        target_table=tgt_tbl,
                        target_column=tgt_cname,
                    )
                )
                fk_map[(src_tidx, src_cname.lower())] = (tgt_tbl, tgt_cname)

        # Group columns into tables
        tables: list[TableMetadata] = []
        for t_idx, tbl_name in enumerate(table_names):
            tbl_columns: list[ColumnMetadata] = []
            tbl_pks: list[str] = []
            tbl_fks: list[ForeignKeyMetadata] = []

            for c_idx, (tbl_idx, c_name, c_type) in col_lookup.items():
                if tbl_idx != t_idx or c_idx == 0:  # c_idx 0 is typically '*'
                    continue

                is_pk = c_idx in primary_keys
                if is_pk:
                    tbl_pks.append(c_name)

                is_fk = (t_idx, c_name.lower()) in fk_map
                f_table, f_col = fk_map.get((t_idx, c_name.lower()), (None, None))

                tbl_columns.append(
                    ColumnMetadata(
                        name=c_name,
                        data_type=c_type,
                        is_primary_key=is_pk,
                        is_foreign_key=is_fk,
                        foreign_table=f_table,
                        foreign_column=f_col,
                    )
                )

            # Filter schema-level foreign keys that belong to this table
            for fk in foreign_keys:
                if fk.source_table == tbl_name:
                    tbl_fks.append(fk)

            tables.append(
                TableMetadata(
                    table_name=tbl_name,
                    columns=tbl_columns,
                    primary_keys=tbl_pks,
                    foreign_keys=tbl_fks,
                )
            )

        try:
            schema_meta = SchemaMetadata(
                db_id=db_id,
                dialect="sqlite",
                tables=tables,
                foreign_keys=foreign_keys,
            )
        except ValidationError as exc:
            raise SpiderAdapterError(f"Schema validation failed for '{db_id}': {exc}") from exc

        schemas[db_id] = schema_meta

    return schemas


def load_spider_split(
    examples_file: Path | str,
    schemas: dict[str, SchemaMetadata],
    split: DatasetSplit,
    id_prefix: str | None = None,
) -> list[TextToSQLExample]:
    """Ingest and validate Spider 1.0 examples from train or dev JSON.

    Args:
        examples_file: Path to train_spider.json, dev.json, etc.
        schemas: Ingested schemas dictionary to validate db_id references.
        split: Canonical split enum (DatasetSplit.TRAIN or DatasetSplit.DEV).
        id_prefix: Optional custom ID prefix. Defaults to 'spider_{split}'.

    Returns:
        List of validated TextToSQLExample instances.

    Raises:
        FileNotFoundError: If examples_file does not exist.
        SpiderAdapterError: If JSON is malformed, required fields missing, or db_id reference missing.
    """
    path = Path(examples_file)
    if not path.is_file():
        raise FileNotFoundError(f"Spider examples file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_examples = json.load(f)
    except json.JSONDecodeError as exc:
        raise SpiderAdapterError(f"Malformed JSON in Spider examples file '{path}': {exc}") from exc

    if not isinstance(raw_examples, list):
        raise SpiderAdapterError(f"Spider examples file '{path}' must contain a JSON list.")

    prefix = id_prefix or f"spider_{split.value}"
    examples: list[TextToSQLExample] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_examples):
        if not isinstance(item, dict):
            raise SpiderAdapterError(
                f"Spider example at index {idx} in '{path}' must be a dictionary."
            )

        question = item.get("question")
        gold_sql = item.get("query")
        db_id = item.get("db_id")

        if not question or not isinstance(question, str) or not question.strip():
            raise SpiderAdapterError(f"Missing or empty 'question' at index {idx} in '{path}'.")

        if not gold_sql or not isinstance(gold_sql, str) or not gold_sql.strip():
            raise SpiderAdapterError(f"Missing or empty 'query' (SQL) at index {idx} in '{path}'.")

        if not db_id or not isinstance(db_id, str) or not db_id.strip():
            raise SpiderAdapterError(f"Missing or empty 'db_id' at index {idx} in '{path}'.")

        if db_id not in schemas:
            raise SpiderAdapterError(
                f"Unknown db_id '{db_id}' referenced at index {idx} in '{path}'. Database missing from tables.json."
            )

        example_id = f"{prefix}_{idx:05d}"
        if example_id in seen_ids:
            raise SpiderAdapterError(f"Duplicate example ID generated: '{example_id}'.")
        seen_ids.add(example_id)

        try:
            example = TextToSQLExample(
                id=example_id,
                question=question.strip(),
                db_id=db_id.strip(),
                gold_sql=gold_sql.strip(),
                dataset_name="spider",
                split=split,
                difficulty=DifficultyLevel.UNCLASSIFIED,
                evidence=None,
                schema_version="spider_1.0",
            )
        except ValidationError as exc:
            raise SpiderAdapterError(
                f"Schema validation failed for example '{example_id}': {exc}"
            ) from exc

        examples.append(example)

    return examples


def create_spider_provenance(
    tables_file: Path | str,
    train_file: Path | str | None = None,
    dev_file: Path | str | None = None,
    revision: str | None = "1.0",
) -> DatasetProvenance:
    """Generate a provenance record for Spider 1.0 with raw file checksums."""
    checksums: dict[str, str] = {}

    for label, filepath in [
        ("tables.json", tables_file),
        ("train_spider.json", train_file),
        ("dev.json", dev_file),
    ]:
        if filepath is not None:
            p = Path(filepath)
            if p.is_file():
                checksums[label] = compute_file_sha256(p)

    return DatasetProvenance(
        dataset_id="spider",
        source_name="Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Text-to-SQL (Yu et al., EMNLP 2018)",
        release_version="1.0",
        revision=revision,
        license_id=LicenseIdentifier.CC_BY_SA_4_0,
        license_notes="Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). Yale LILY Lab.",
        dialect="sqlite",
        download_url="https://yale-lily.github.io/spider",
        raw_files_checksums=checksums,
        normalization_version="1.0.0",
        created_at=datetime.now(UTC).isoformat(),
        description="Canonical cross-domain text-to-SQL benchmark with cross-database partition splitting.",
    )
