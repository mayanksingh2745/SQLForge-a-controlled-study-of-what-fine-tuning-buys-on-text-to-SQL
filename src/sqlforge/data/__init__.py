"""Data layer: Ingestion, schema normalization, and split management."""

from sqlforge.data.audit import (
    ContaminationAuditor,
    IsolationGuard,
    LeakageContaminationError,
)
from sqlforge.data.bird import (
    BIRD_VARIANT_METADATA,
    BIRDAdapterError,
    BirdSubsetVariant,
    create_bird_provenance,
    load_bird_mini_dev,
    parse_bird_tables,
)
from sqlforge.data.fixtures import (
    FixtureExample,
    get_default_fixture_path,
    load_fixture_dataset,
)
from sqlforge.data.held_out import (
    HELD_OUT_DB_ID,
    HeldOutAdapterError,
    create_held_out_provenance,
    get_subscription_analytics_schema,
    load_held_out_dataset,
)
from sqlforge.data.manifest import (
    build_dataset_manifest,
    load_examples_from_file,
    save_manifest,
    verify_dataset_manifest,
    write_processed_dataset,
)
from sqlforge.data.normalization import (
    compute_examples_cumulative_hash,
    compute_file_sha256,
    compute_ngram_jaccard,
    normalize_question,
    normalize_sql,
)
from sqlforge.data.spider import (
    SpiderAdapterError,
    create_spider_provenance,
    load_spider_split,
    parse_spider_tables,
)

__all__ = [
    "BIRDAdapterError",
    "BIRD_VARIANT_METADATA",
    "BirdSubsetVariant",
    "ContaminationAuditor",
    "FixtureExample",
    "HELD_OUT_DB_ID",
    "HeldOutAdapterError",
    "IsolationGuard",
    "LeakageContaminationError",
    "build_dataset_manifest",
    "compute_examples_cumulative_hash",
    "compute_file_sha256",
    "compute_ngram_jaccard",
    "create_bird_provenance",
    "create_held_out_provenance",
    "create_spider_provenance",
    "get_default_fixture_path",
    "get_subscription_analytics_schema",
    "load_bird_mini_dev",
    "load_examples_from_file",
    "load_fixture_dataset",
    "load_held_out_dataset",
    "load_spider_split",
    "normalize_question",
    "normalize_sql",
    "parse_bird_tables",
    "parse_spider_tables",
    "save_manifest",
    "SpiderAdapterError",
    "verify_dataset_manifest",
    "write_processed_dataset",
]
