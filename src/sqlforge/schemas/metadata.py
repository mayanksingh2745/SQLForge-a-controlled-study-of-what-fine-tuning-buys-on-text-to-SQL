"""Database and Schema representation contracts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ColumnMetadata(BaseModel):
    """Metadata describing a single database column."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Column name as defined in database")
    data_type: str = Field(..., description="Canonical data type (e.g., INTEGER, VARCHAR, TEXT)")
    is_primary_key: bool = Field(default=False, description="True if column is part of primary key")
    is_foreign_key: bool = Field(
        default=False, description="True if column references another table"
    )
    foreign_table: str | None = Field(default=None, description="Referenced table if foreign key")
    foreign_column: str | None = Field(default=None, description="Referenced column if foreign key")
    description: str | None = Field(
        default=None, description="Semantic description or column docstring"
    )
    sample_values: list[Any] = Field(
        default_factory=list, description="Small list of distinct sample values"
    )


class ForeignKeyMetadata(BaseModel):
    """Explicit relationship mapping between two table columns."""

    model_config = ConfigDict(frozen=True)

    source_table: str = Field(..., description="Child table containing foreign key")
    source_column: str = Field(..., description="Foreign key column name")
    target_table: str = Field(..., description="Parent referenced table")
    target_column: str = Field(..., description="Referenced primary key column name")


class TableMetadata(BaseModel):
    """Metadata describing a single database table."""

    model_config = ConfigDict(frozen=True)

    table_name: str = Field(..., description="Table name in database")
    columns: list[ColumnMetadata] = Field(..., description="List of columns in the table")
    primary_keys: list[str] = Field(
        default_factory=list, description="List of primary key column names"
    )
    foreign_keys: list[ForeignKeyMetadata] = Field(
        default_factory=list, description="Outgoing foreign key relationships"
    )
    row_count: int | None = Field(default=None, description="Total row count, if known")
    description: str | None = Field(default=None, description="Table purpose and context")


class SchemaMetadata(BaseModel):
    """Complete representation of a database schema for prompt serialization and validation."""

    model_config = ConfigDict(frozen=True)

    db_id: str = Field(
        ..., description="Unique database identifier (e.g. 'flight_2', 'california_schools')"
    )
    dialect: str = Field(default="sqlite", description="SQL dialect ('sqlite', 'postgresql', etc.)")
    tables: list[TableMetadata] = Field(
        default_factory=list, description="Tables belonging to this schema"
    )
    foreign_keys: list[ForeignKeyMetadata] = Field(
        default_factory=list, description="All schema-level foreign keys"
    )
    description: str | None = Field(
        default=None, description="Optional domain description of database"
    )

    def get_table(self, table_name: str) -> TableMetadata | None:
        """Retrieve table metadata by case-insensitive name."""
        name_lower = table_name.lower()
        for tbl in self.tables:
            if tbl.table_name.lower() == name_lower:
                return tbl
        return None
