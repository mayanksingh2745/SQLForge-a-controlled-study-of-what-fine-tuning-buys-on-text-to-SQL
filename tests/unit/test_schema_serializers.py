"""Unit tests for Schema serializers (DDL, Compact Pipe, JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqlforge.data.held_out import get_subscription_analytics_schema
from sqlforge.data.spider import parse_spider_tables
from sqlforge.prompting.serializers import (
    CompactPipeSerializer,
    DDLSerializer,
    JSONSchemaSerializer,
    get_serializer,
    quote_identifier,
)
from sqlforge.schemas.metadata import (
    ColumnMetadata,
    ForeignKeyMetadata,
    SchemaMetadata,
    TableMetadata,
)


@pytest.fixture
def spider_schemas() -> dict[str, SchemaMetadata]:
    """Load parsed Spider test schemas."""
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "dataset" / "spider" / "tables.json"
    )
    return parse_spider_tables(fixture_path)


@pytest.fixture
def bird_schemas() -> dict[str, SchemaMetadata]:
    """Load parsed BIRD test schemas."""
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "dataset" / "bird" / "dev_tables.json"
    )
    return parse_spider_tables(fixture_path)


def test_quote_identifier() -> None:
    """Test SQL identifier escaping for reserved words and symbols."""
    assert quote_identifier("valid_name") == "valid_name"
    assert quote_identifier("user_id") == "user_id"
    # SQLite reserved keywords
    assert quote_identifier("order") == '"order"'
    assert quote_identifier("group") == '"group"'
    assert quote_identifier("table") == '"table"'
    assert quote_identifier("select") == '"select"'
    # Spaces or special characters
    assert quote_identifier("column with space") == '"column with space"'
    assert quote_identifier('already"quoted') == '"already""quoted"'
    assert quote_identifier("") == '""'


def test_ddl_serializer_spider_stadium(spider_schemas: dict[str, SchemaMetadata]) -> None:
    """Test DDL serialization of Spider stadium schema."""
    schema = spider_schemas["stadium"]
    serializer = DDLSerializer(include_comments=True)
    ddl = serializer.serialize(schema)

    assert "CREATE TABLE" in ddl
    assert "stadium" in ddl
    assert "PRIMARY KEY" in ddl
    # Check deterministic ordering
    lines = ddl.split("\n")
    assert any("CREATE TABLE stadium" in line or 'CREATE TABLE "stadium"' in line for line in lines)


def test_ddl_serializer_foreign_keys_and_composite() -> None:
    """Test DDL serializer handles primary keys, single and composite foreign keys."""
    schema = SchemaMetadata(
        db_id="shop_db",
        dialect="sqlite",
        tables=[
            TableMetadata(
                table_name="orders",
                columns=[
                    ColumnMetadata(name="order_id", data_type="INTEGER", is_primary_key=True),
                    ColumnMetadata(name="item_id", data_type="INTEGER", is_primary_key=True),
                    ColumnMetadata(name="customer_id", data_type="INTEGER"),
                ],
                primary_keys=["order_id", "item_id"],
                foreign_keys=[
                    ForeignKeyMetadata(
                        source_table="orders",
                        source_column="customer_id",
                        target_table="customers",
                        target_column="customer_id",
                    )
                ],
            ),
            TableMetadata(
                table_name="customers",
                columns=[
                    ColumnMetadata(name="customer_id", data_type="INTEGER", is_primary_key=True),
                    ColumnMetadata(name="name", data_type="TEXT"),
                ],
                primary_keys=["customer_id"],
            ),
        ],
    )
    serializer = DDLSerializer(sort_tables=True)
    ddl = serializer.serialize(schema)

    # Customers must appear before Orders if sorted alphabetically
    cust_idx = ddl.find("customers")
    orders_idx = ddl.find("orders")
    assert cust_idx < orders_idx
    # Composite PK
    assert "PRIMARY KEY (order_id, item_id)" in ddl
    # Foreign key
    assert "FOREIGN KEY (customer_id) REFERENCES customers (customer_id)" in ddl


def test_ddl_serializer_sample_rows_and_comments() -> None:
    """Test DDL serializer appends sample rows and column comments."""
    schema = SchemaMetadata(
        db_id="sample_db",
        dialect="sqlite",
        tables=[
            TableMetadata(
                table_name="products",
                description="Catalog items",
                columns=[
                    ColumnMetadata(
                        name="product_name",
                        data_type="TEXT",
                        description="Item name in catalog",
                        sample_values=["Widget A", "Gadget B"],
                    )
                ],
            )
        ],
    )
    # With comments and samples
    s_full = DDLSerializer(include_comments=True, include_sample_rows=2)
    ddl_full = s_full.serialize(schema)
    assert "-- Table: products (Catalog items)" in ddl_full
    assert "-- Item name in catalog" in ddl_full
    assert "/* sample: ['Widget A', 'Gadget B'] */" in ddl_full

    # Stripped
    s_plain = DDLSerializer(include_comments=False, include_sample_rows=0)
    ddl_plain = s_plain.serialize(schema)
    assert "-- Table:" not in ddl_plain
    assert "/* sample:" not in ddl_plain


def test_compact_pipe_serializer(spider_schemas: dict[str, SchemaMetadata]) -> None:
    """Test Compact Pipe serializer generates expected compact representation."""
    schema = spider_schemas["stadium"]
    serializer = CompactPipeSerializer(delimiter="|", include_types=True, include_keys=True)
    compact = serializer.serialize(schema)

    assert "stadium :" in compact
    assert "|" in compact
    assert "PK" in compact


def test_json_schema_serializer() -> None:
    """Test JSON Schema serializer generates valid, parsable JSON."""
    schema = get_subscription_analytics_schema()
    serializer = JSONSchemaSerializer(indent=2)
    serialized = serializer.serialize(schema)

    data = json.loads(serialized)
    assert data["db_id"] == "subscription_analytics_db"
    assert data["dialect"] == "sqlite"
    assert len(data["tables"]) == 6
    tbl_names = {t["table_name"] for t in data["tables"]}
    assert "customers" in tbl_names
    assert "subscriptions" in tbl_names
    assert "invoices" in tbl_names


def test_get_serializer_factory() -> None:
    """Test get_serializer returns correct serializer instances and handles invalid formats."""
    assert isinstance(get_serializer("ddl"), DDLSerializer)
    assert isinstance(get_serializer("compact"), CompactPipeSerializer)
    assert isinstance(get_serializer("pipe"), CompactPipeSerializer)
    assert isinstance(get_serializer("json"), JSONSchemaSerializer)

    with pytest.raises(ValueError, match="Unsupported schema format"):
        get_serializer("invalid_format")
