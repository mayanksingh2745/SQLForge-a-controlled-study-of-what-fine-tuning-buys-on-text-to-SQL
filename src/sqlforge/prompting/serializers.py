"""Schema serializers for Text-to-SQL prompt assembly.

Provides deterministic converters for SchemaMetadata into:
1. DDL: Standard SQLite CREATE TABLE statements with column types, primary keys,
   foreign keys, and optional comments/sample rows.
2. Compact Pipe: Token-efficient single/multi-line pipe-delimited schema notation.
3. JSON Schema: Structured, machine-readable JSON schema specification.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Protocol

from sqlforge.schemas.metadata import SchemaMetadata

# Standard SQL reserved words in SQLite requiring quoting if used as identifiers
SQLITE_RESERVED = frozenset(
    {
        "abort",
        "action",
        "add",
        "after",
        "all",
        "alter",
        "always",
        "analyze",
        "and",
        "as",
        "asc",
        "attach",
        "autoincrement",
        "before",
        "begin",
        "between",
        "by",
        "cascade",
        "case",
        "cast",
        "check",
        "collate",
        "column",
        "commit",
        "conflict",
        "constraint",
        "create",
        "cross",
        "current",
        "current_date",
        "current_time",
        "current_timestamp",
        "database",
        "default",
        "deferrable",
        "deferred",
        "delete",
        "desc",
        "detach",
        "distinct",
        "do",
        "drop",
        "each",
        "else",
        "end",
        "escape",
        "except",
        "exclude",
        "exclusive",
        "exists",
        "explain",
        "fail",
        "filter",
        "first",
        "following",
        "for",
        "foreign",
        "from",
        "full",
        "generated",
        "glob",
        "group",
        "having",
        "if",
        "ignore",
        "immediate",
        "in",
        "index",
        "indexed",
        "initially",
        "inner",
        "insert",
        "instead",
        "intersect",
        "into",
        "is",
        "isnull",
        "join",
        "key",
        "last",
        "left",
        "like",
        "limit",
        "match",
        "materialized",
        "natural",
        "no",
        "not",
        "nothing",
        "notnull",
        "null",
        "nulls",
        "of",
        "offset",
        "on",
        "or",
        "order",
        "others",
        "outer",
        "over",
        "partition",
        "plan",
        "pragma",
        "preceding",
        "primary",
        "query",
        "raise",
        "range",
        "recursive",
        "references",
        "regexp",
        "reindex",
        "release",
        "rename",
        "replace",
        "restrict",
        "returning",
        "right",
        "rollback",
        "row",
        "rows",
        "savepoint",
        "select",
        "set",
        "table",
        "temp",
        "temporary",
        "then",
        "ties",
        "to",
        "transaction",
        "trigger",
        "unbounded",
        "union",
        "unique",
        "update",
        "using",
        "vacuum",
        "values",
        "view",
        "virtual",
        "when",
        "where",
        "window",
        "with",
        "without",
    }
)


def quote_identifier(name: str) -> str:
    """Escape an identifier for SQLite DDL statements.

    Uses double quotes if the identifier contains special characters, spaces,
    or matches SQLite reserved keywords.
    """
    cleaned = name.strip()
    if not cleaned:
        return '""'

    # Check if name is purely alphanumeric + underscore and not a reserved keyword
    is_simple = bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", cleaned))
    if is_simple and cleaned.lower() not in SQLITE_RESERVED:
        return cleaned

    # Escape any existing double quotes by doubling them
    escaped = cleaned.replace('"', '""')
    return f'"{escaped}"'


class SchemaSerializer(Protocol):
    """Protocol for converting SchemaMetadata into LLM prompt text."""

    def serialize(self, schema: SchemaMetadata) -> str:
        """Format schema metadata into serialized prompt representation."""
        ...


class DDLSerializer:
    """Serializes SchemaMetadata into valid SQLite DDL CREATE TABLE statements."""

    def __init__(
        self,
        include_comments: bool = True,
        include_sample_rows: int = 0,
        sort_tables: bool = True,
    ) -> None:
        """Initialize the DDL serializer.

        Args:
            include_comments: If True, renders table and column description comments.
            include_sample_rows: Max number of sample values to document per column.
            sort_tables: If True, outputs tables in deterministic alphabetical order.
        """
        self.include_comments = include_comments
        self.include_sample_rows = include_sample_rows
        self.sort_tables = sort_tables

    def serialize(self, schema: SchemaMetadata) -> str:
        """Generate valid SQLite DDL statements for the schema."""
        tables = list(schema.tables)
        if self.sort_tables:
            tables.sort(key=lambda t: t.table_name.lower())

        table_ddls: list[str] = []

        for tbl in tables:
            ddl_parts: list[str] = []

            # Optional table description comment
            if self.include_comments and tbl.description:
                ddl_parts.append(f"-- Table: {tbl.table_name} ({tbl.description.strip()})")

            quoted_tbl = quote_identifier(tbl.table_name)
            col_lines: list[str] = []

            for col in tbl.columns:
                col_name_quoted = quote_identifier(col.name)
                data_type = col.data_type.strip() if col.data_type else "TEXT"
                col_def = f"  {col_name_quoted} {data_type}"

                # Append inline sample rows if requested and available
                if self.include_sample_rows > 0 and col.sample_values:
                    samples = col.sample_values[: self.include_sample_rows]
                    sample_str = ", ".join(repr(s) for s in samples)
                    col_def += f" /* sample: [{sample_str}] */"

                # Append column description if requested
                if self.include_comments and col.description:
                    col_def += f" -- {col.description.strip()}"

                col_lines.append(col_def)

            # Table-level Primary Key
            if tbl.primary_keys:
                pk_quoted = ", ".join(quote_identifier(pk) for pk in tbl.primary_keys)
                col_lines.append(f"  PRIMARY KEY ({pk_quoted})")

            # Resolve foreign keys for this table:
            # 1. From tbl.foreign_keys
            # 2. Or from schema.foreign_keys matching this table
            fks = list(tbl.foreign_keys)
            existing_fk_sigs = {
                (fk.source_column.lower(), fk.target_table.lower(), fk.target_column.lower())
                for fk in fks
            }
            for sfk in schema.foreign_keys:
                if sfk.source_table.lower() == tbl.table_name.lower():
                    sig = (
                        sfk.source_column.lower(),
                        sfk.target_table.lower(),
                        sfk.target_column.lower(),
                    )
                    if sig not in existing_fk_sigs:
                        fks.append(sfk)
                        existing_fk_sigs.add(sig)

            # Group foreign keys pointing to the same target table to handle composite FKs cleanly
            fk_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for fk in fks:
                fk_groups[fk.target_table].append((fk.source_column, fk.target_column))

            # Emit deterministic foreign key constraints sorted by target table
            for target_table in sorted(fk_groups.keys(), key=str.lower):
                col_pairs = fk_groups[target_table]
                src_cols = ", ".join(quote_identifier(src) for src, _ in col_pairs)
                tgt_cols = ", ".join(quote_identifier(tgt) for _, tgt in col_pairs)
                quoted_target = quote_identifier(target_table)
                col_lines.append(
                    f"  FOREIGN KEY ({src_cols}) REFERENCES {quoted_target} ({tgt_cols})"
                )

            body = ",\n".join(col_lines)
            statement = f"CREATE TABLE {quoted_tbl} (\n{body}\n);"
            ddl_parts.append(statement)
            table_ddls.append("\n".join(ddl_parts))

        return "\n\n".join(table_ddls)


class CompactPipeSerializer:
    """Serializes SchemaMetadata into a token-efficient pipe-delimited schema string."""

    def __init__(
        self,
        delimiter: str = "|",
        include_types: bool = True,
        include_keys: bool = True,
        sort_tables: bool = True,
    ) -> None:
        """Initialize the Compact Pipe serializer.

        Args:
            delimiter: Token separating columns (default '|').
            include_types: Whether to display data types.
            include_keys: Whether to annotate PK and FK relationships.
            sort_tables: If True, outputs tables in deterministic alphabetical order.
        """
        self.delimiter = f" {delimiter.strip()} "
        self.include_types = include_types
        self.include_keys = include_keys
        self.sort_tables = sort_tables

    def serialize(self, schema: SchemaMetadata) -> str:
        """Generate compact pipe-delimited schema notation."""
        tables = list(schema.tables)
        if self.sort_tables:
            tables.sort(key=lambda t: t.table_name.lower())

        table_lines: list[str] = []

        # Build schema-level foreign key lookup: (table.lower(), col.lower()) -> (target_table, target_col)
        fk_lookup: dict[tuple[str, str], tuple[str, str]] = {}
        for fk in schema.foreign_keys:
            fk_lookup[(fk.source_table.lower(), fk.source_column.lower())] = (
                fk.target_table,
                fk.target_column,
            )
        for tbl in tables:
            for fk in tbl.foreign_keys:
                fk_lookup[(tbl.table_name.lower(), fk.source_column.lower())] = (
                    fk.target_table,
                    fk.target_column,
                )

        for tbl in tables:
            pk_set = {pk.lower() for pk in tbl.primary_keys}
            col_tokens: list[str] = []

            for col in tbl.columns:
                c_name = col.name
                mods: list[str] = []

                if self.include_types and col.data_type:
                    mods.append(col.data_type.upper())

                if self.include_keys:
                    is_pk = col.is_primary_key or col.name.lower() in pk_set
                    if is_pk:
                        mods.append("PK")

                    fk_target = fk_lookup.get((tbl.table_name.lower(), col.name.lower()))
                    if fk_target:
                        tgt_tbl, tgt_col = fk_target
                        mods.append(f"FK->{tgt_tbl}.{tgt_col}")
                    elif col.is_foreign_key and col.foreign_table and col.foreign_column:
                        mods.append(f"FK->{col.foreign_table}.{col.foreign_column}")

                if mods:
                    col_tokens.append(f"{c_name} ({', '.join(mods)})")
                else:
                    col_tokens.append(c_name)

            table_lines.append(f"{tbl.table_name} : {self.delimiter.join(col_tokens)}")

        return "\n".join(table_lines)


class JSONSchemaSerializer:
    """Serializes SchemaMetadata into structured, deterministic JSON."""

    def __init__(self, indent: int = 2, sort_tables: bool = True) -> None:
        """Initialize the JSON schema serializer."""
        self.indent = indent
        self.sort_tables = sort_tables

    def serialize(self, schema: SchemaMetadata) -> str:
        """Generate structured JSON representation of the schema."""
        tables = list(schema.tables)
        if self.sort_tables:
            tables.sort(key=lambda t: t.table_name.lower())

        data: dict[str, Any] = {
            "db_id": schema.db_id,
            "dialect": schema.dialect,
            "tables": [],
        }

        for tbl in tables:
            tbl_data: dict[str, Any] = {
                "table_name": tbl.table_name,
                "primary_keys": list(tbl.primary_keys),
                "columns": [
                    {
                        "name": col.name,
                        "data_type": col.data_type,
                        "is_primary_key": col.is_primary_key,
                        "is_foreign_key": col.is_foreign_key,
                        "foreign_table": col.foreign_table,
                        "foreign_column": col.foreign_column,
                    }
                    for col in tbl.columns
                ],
                "foreign_keys": [
                    {
                        "source_column": fk.source_column,
                        "target_table": fk.target_table,
                        "target_column": fk.target_column,
                    }
                    for fk in tbl.foreign_keys
                ],
            }
            data["tables"].append(tbl_data)

        return json.dumps(data, indent=self.indent)


def get_serializer(format_name: str = "ddl", **kwargs: Any) -> SchemaSerializer:
    """Factory for obtaining a configured SchemaSerializer instance.

    Args:
        format_name: Serializer format ('ddl', 'compact', or 'json').
        **kwargs: Configuration options passed to the serializer constructor.

    Returns:
        Configured SchemaSerializer instance.

    Raises:
        ValueError: If format_name is unrecognized.
    """
    fmt = format_name.strip().lower()
    if fmt == "ddl":
        return DDLSerializer(**kwargs)
    elif fmt in {"compact", "pipe"}:
        return CompactPipeSerializer(**kwargs)
    elif fmt == "json":
        return JSONSchemaSerializer(**kwargs)
    else:
        raise ValueError(
            f"Unsupported schema format: '{format_name}'. Expected 'ddl', 'compact', or 'json'."
        )
