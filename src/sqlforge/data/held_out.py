"""Custom Held-Out Benchmark schema foundation and evaluation contract.

Implements the subscription_analytics_db schema contract: an independently designed
6-table relational enterprise billing and support database strictly isolated
from Spider and BIRD training distributions.

Epistemic boundary note:
This schema eliminates direct benchmark overlap with Spider and BIRD. It does not
claim absolute immunization against general semantic exposure in pre-trained foundation models.
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

HELD_OUT_DB_ID = "subscription_analytics_db"


class HeldOutAdapterError(ValueError):
    """Raised when custom held-out data violates schema contracts or isolation boundaries."""


def get_subscription_analytics_schema() -> SchemaMetadata:
    """Construct the canonical SchemaMetadata for subscription_analytics_db.

    Tables:
    1. customers: customer profiles, company, country, signup dates
    2. plans: subscription tiers, monthly fee, max user quotas
    3. subscriptions: customer plan subscriptions, auto-renew, lifecycle dates
    4. invoices: billing periods, amounts, discount deductions, payment statuses
    5. transactions: gateway payments (Stripe/PayPal/Wire), success flags
    6. support_tickets: service inquiries, priority, resolution timestamps
    """
    foreign_keys = [
        ForeignKeyMetadata(
            source_table="subscriptions",
            source_column="customer_id",
            target_table="customers",
            target_column="customer_id",
        ),
        ForeignKeyMetadata(
            source_table="subscriptions",
            source_column="plan_id",
            target_table="plans",
            target_column="plan_id",
        ),
        ForeignKeyMetadata(
            source_table="invoices",
            source_column="subscription_id",
            target_table="subscriptions",
            target_column="subscription_id",
        ),
        ForeignKeyMetadata(
            source_table="transactions",
            source_column="invoice_id",
            target_table="invoices",
            target_column="invoice_id",
        ),
        ForeignKeyMetadata(
            source_table="support_tickets",
            source_column="customer_id",
            target_table="customers",
            target_column="customer_id",
        ),
    ]

    customers_table = TableMetadata(
        table_name="customers",
        columns=[
            ColumnMetadata(name="customer_id", data_type="INTEGER", is_primary_key=True),
            ColumnMetadata(name="first_name", data_type="TEXT"),
            ColumnMetadata(name="last_name", data_type="TEXT"),
            ColumnMetadata(name="email", data_type="TEXT"),
            ColumnMetadata(name="company_name", data_type="TEXT"),
            ColumnMetadata(name="country", data_type="TEXT"),
            ColumnMetadata(name="created_at", data_type="TIMESTAMP"),
        ],
        primary_keys=["customer_id"],
        foreign_keys=[],
        description="Customer profile records and company affiliations",
    )

    plans_table = TableMetadata(
        table_name="plans",
        columns=[
            ColumnMetadata(name="plan_id", data_type="INTEGER", is_primary_key=True),
            ColumnMetadata(name="plan_name", data_type="TEXT"),
            ColumnMetadata(name="tier", data_type="TEXT"),
            ColumnMetadata(name="monthly_fee_usd", data_type="REAL"),
            ColumnMetadata(name="max_users", data_type="INTEGER"),
        ],
        primary_keys=["plan_id"],
        foreign_keys=[],
        description="Available recurring subscription tiers and pricing",
    )

    subscriptions_table = TableMetadata(
        table_name="subscriptions",
        columns=[
            ColumnMetadata(name="subscription_id", data_type="INTEGER", is_primary_key=True),
            ColumnMetadata(
                name="customer_id",
                data_type="INTEGER",
                is_foreign_key=True,
                foreign_table="customers",
                foreign_column="customer_id",
            ),
            ColumnMetadata(
                name="plan_id",
                data_type="INTEGER",
                is_foreign_key=True,
                foreign_table="plans",
                foreign_column="plan_id",
            ),
            ColumnMetadata(name="status", data_type="TEXT"),
            ColumnMetadata(name="start_date", data_type="DATE"),
            ColumnMetadata(name="end_date", data_type="DATE"),
            ColumnMetadata(name="auto_renew", data_type="BOOLEAN"),
        ],
        primary_keys=["subscription_id"],
        foreign_keys=[foreign_keys[0], foreign_keys[1]],
        description="Active and canceled customer subscription agreements",
    )

    invoices_table = TableMetadata(
        table_name="invoices",
        columns=[
            ColumnMetadata(name="invoice_id", data_type="INTEGER", is_primary_key=True),
            ColumnMetadata(
                name="subscription_id",
                data_type="INTEGER",
                is_foreign_key=True,
                foreign_table="subscriptions",
                foreign_column="subscription_id",
            ),
            ColumnMetadata(name="amount_due_usd", data_type="REAL"),
            ColumnMetadata(name="discount_amount_usd", data_type="REAL"),
            ColumnMetadata(name="issue_date", data_type="DATE"),
            ColumnMetadata(name="due_date", data_type="DATE"),
            ColumnMetadata(name="paid_date", data_type="DATE"),
            ColumnMetadata(name="status", data_type="TEXT"),
        ],
        primary_keys=["invoice_id"],
        foreign_keys=[foreign_keys[2]],
        description="Billing cycle invoice statements, discounts, and payment statuses",
    )

    transactions_table = TableMetadata(
        table_name="transactions",
        columns=[
            ColumnMetadata(name="transaction_id", data_type="INTEGER", is_primary_key=True),
            ColumnMetadata(
                name="invoice_id",
                data_type="INTEGER",
                is_foreign_key=True,
                foreign_table="invoices",
                foreign_column="invoice_id",
            ),
            ColumnMetadata(name="payment_gateway", data_type="TEXT"),
            ColumnMetadata(name="amount_paid_usd", data_type="REAL"),
            ColumnMetadata(name="transaction_date", data_type="TIMESTAMP"),
            ColumnMetadata(name="success", data_type="BOOLEAN"),
        ],
        primary_keys=["transaction_id"],
        foreign_keys=[foreign_keys[3]],
        description="Financial settlement transactions and gateway authorization records",
    )

    support_tickets_table = TableMetadata(
        table_name="support_tickets",
        columns=[
            ColumnMetadata(name="ticket_id", data_type="INTEGER", is_primary_key=True),
            ColumnMetadata(
                name="customer_id",
                data_type="INTEGER",
                is_foreign_key=True,
                foreign_table="customers",
                foreign_column="customer_id",
            ),
            ColumnMetadata(name="priority", data_type="TEXT"),
            ColumnMetadata(name="category", data_type="TEXT"),
            ColumnMetadata(name="created_at", data_type="TIMESTAMP"),
            ColumnMetadata(name="resolved_at", data_type="TIMESTAMP"),
            ColumnMetadata(name="satisfaction_score", data_type="INTEGER"),
        ],
        primary_keys=["ticket_id"],
        foreign_keys=[foreign_keys[4]],
        description="Customer support cases, resolution velocity, and satisfaction metrics",
    )

    return SchemaMetadata(
        db_id=HELD_OUT_DB_ID,
        dialect="sqlite",
        tables=[
            customers_table,
            plans_table,
            subscriptions_table,
            invoices_table,
            transactions_table,
            support_tickets_table,
        ],
        foreign_keys=foreign_keys,
        description="Normalized SaaS subscription and customer analytics database schema",
    )


def load_held_out_dataset(
    examples_file: Path | str,
    expected_db_id: str = HELD_OUT_DB_ID,
) -> list[TextToSQLExample]:
    """Load and validate custom held-out evaluation examples.

    Args:
        examples_file: Path to custom held-out JSON file.
        expected_db_id: Target database ID (defaults to 'subscription_analytics_db').

    Returns:
        List of TextToSQLExample instances partitioned strictly under DatasetSplit.HELD_OUT.

    Raises:
        FileNotFoundError: If examples_file does not exist.
        HeldOutAdapterError: If JSON is malformed or records violate schema contracts.
    """
    path = Path(examples_file)
    if not path.is_file():
        raise FileNotFoundError(f"Custom held-out dataset file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_examples = json.load(f)
    except json.JSONDecodeError as exc:
        raise HeldOutAdapterError(
            f"Malformed JSON in custom held-out file '{path}': {exc}"
        ) from exc

    if not isinstance(raw_examples, list):
        raise HeldOutAdapterError(f"Custom held-out file '{path}' must contain a JSON list.")

    examples: list[TextToSQLExample] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_examples):
        if not isinstance(item, dict):
            raise HeldOutAdapterError(f"Held-out example at index {idx} must be a dictionary.")

        question = item.get("question")
        gold_sql = item.get("gold_sql") or item.get("query")
        db_id = item.get("db_id", expected_db_id)
        example_id = item.get("example_id") or item.get("id") or f"custom_held_out_{idx:04d}"
        diff_str = str(item.get("difficulty", "medium")).lower()

        if not question or not isinstance(question, str) or not question.strip():
            raise HeldOutAdapterError(f"Missing question at index {idx} in held-out dataset.")

        if not gold_sql or not isinstance(gold_sql, str) or not gold_sql.strip():
            raise HeldOutAdapterError(f"Missing gold_sql at index {idx} in held-out dataset.")

        if db_id != expected_db_id:
            raise HeldOutAdapterError(
                f"Held-out example '{example_id}' referenced unexpected db_id '{db_id}'. "
                f"Expected '{expected_db_id}'."
            )

        if example_id in seen_ids:
            raise HeldOutAdapterError(f"Duplicate example ID '{example_id}' in held-out dataset.")
        seen_ids.add(example_id)

        try:
            diff_level = DifficultyLevel(diff_str)
        except ValueError:
            diff_level = DifficultyLevel.UNCLASSIFIED

        try:
            example = TextToSQLExample(
                id=example_id,
                question=question.strip(),
                db_id=db_id.strip(),
                gold_sql=gold_sql.strip(),
                dataset_name="custom_held_out",
                split=DatasetSplit.HELD_OUT,
                difficulty=diff_level,
                evidence=item.get("evidence"),
                schema_version="custom_held_out_1.0",
            )
        except ValidationError as exc:
            raise HeldOutAdapterError(
                f"Schema validation failed for held-out example '{example_id}': {exc}"
            ) from exc

        examples.append(example)

    return examples


def create_held_out_provenance(
    schema_file: Path | str | None = None,
    examples_file: Path | str | None = None,
) -> DatasetProvenance:
    """Generate provenance record for custom held-out dataset under Apache 2.0."""
    checksums: dict[str, str] = {}
    if schema_file and Path(schema_file).is_file():
        checksums["schema.json"] = compute_file_sha256(Path(schema_file))
    if examples_file and Path(examples_file).is_file():
        checksums["examples.json"] = compute_file_sha256(Path(examples_file))

    return DatasetProvenance(
        dataset_id="custom_held_out",
        source_name="SQLForge Custom Held-Out Benchmark (subscription_analytics_db)",
        release_version="1.0.0",
        revision="git-HEAD",
        license_id=LicenseIdentifier.APACHE_2_0,
        license_notes="Authored natively by SQLForge research maintainers under Apache 2.0. Isolated OOD evaluation.",
        dialect="sqlite",
        download_url=None,
        raw_files_checksums=checksums,
        normalization_version="1.0.0",
        created_at=datetime.now(UTC).isoformat(),
        description="Enterprise subscription billing schema designed independently from academic benchmarks.",
    )
