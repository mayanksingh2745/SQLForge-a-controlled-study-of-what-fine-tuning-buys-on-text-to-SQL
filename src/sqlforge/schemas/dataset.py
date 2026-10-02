"""Data contracts for dataset provenance, partition manifests, and contamination audit reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.schemas.examples import DatasetSplit


class LicenseIdentifier(StrEnum):
    """Standardized dataset license identifiers."""

    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC_BY_NC_SA_4_0 = "CC-BY-NC-SA-4.0"
    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    CUSTOM_RESTRICTED = "Custom-Restricted"


class DatasetProvenance(BaseModel):
    """Provenance and licensing metadata for an ingested dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(
        ...,
        min_length=1,
        description="Dataset identifier ('spider', 'bird_mini', 'custom_held_out')",
    )
    source_name: str = Field(..., min_length=1, description="Full descriptive name of data source")
    release_version: str = Field(
        ..., min_length=1, description="Official upstream release or version tag"
    )
    revision: str | None = Field(
        default=None, description="Git commit hash, tarball release tag, or archive revision"
    )
    license_id: LicenseIdentifier | str = Field(..., description="Upstream licensing identifier")
    license_notes: str | None = Field(
        default=None,
        description="Special licensing terms, citations, or non-commercial restrictions",
    )
    dialect: str = Field(
        default="sqlite", description="Database engine dialect ('sqlite', 'postgresql')"
    )
    download_url: str | None = Field(
        default=None, description="Canonical public download or homepage URL"
    )
    raw_files_checksums: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of relative raw file path to SHA-256 hash",
    )
    normalization_version: str = Field(
        default="1.0.0",
        description="Version of SQLForge ingestion and normalization logic used",
    )
    created_at: str = Field(
        ..., description="ISO 8601 UTC timestamp of ingestion or provenance generation"
    )
    description: str | None = Field(
        default=None, description="Contextual description of dataset purpose"
    )


class DatasetSplitManifest(BaseModel):
    """Integrity and distribution manifest for a single dataset partition split."""

    model_config = ConfigDict(frozen=True)

    split: DatasetSplit = Field(..., description="Partition split identifier")
    count: int = Field(..., ge=0, description="Total number of examples in this split")
    sha256_hash: str = Field(
        ..., description="Cryptographic SHA-256 hash of processed partition file"
    )
    file_path: str = Field(
        ..., min_length=1, description="Relative path to processed file from project root"
    )
    db_ids: list[str] = Field(
        default_factory=list, description="Sorted list of distinct database IDs in split"
    )
    examples_hash: str = Field(
        ...,
        description="Deterministic cumulative SHA-256 hash across sorted example IDs and contents",
    )


class DatasetManifest(BaseModel):
    """Comprehensive dataset manifest capturing all partition splits and schema integrity."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(
        ..., min_length=1, description="Dataset identifier ('spider', 'bird_mini', etc.)"
    )
    version: str = Field(..., min_length=1, description="Release or catalog version string")
    splits: dict[str, DatasetSplitManifest] = Field(
        ..., description="Mapping of split name to split manifest"
    )
    schema_count: int = Field(..., ge=0, description="Total number of database schemas ingested")
    total_examples: int = Field(..., ge=0, description="Total number of examples across all splits")
    manifest_created_at: str = Field(
        ..., description="ISO 8601 UTC timestamp when manifest was written"
    )
    provenance: DatasetProvenance = Field(..., description="Attached dataset provenance record")


class DuplicateDetail(BaseModel):
    """Record of an exact duplicate query or question identified across partitions."""

    model_config = ConfigDict(frozen=True)

    example_id_1: str = Field(..., description="First example ID")
    split_1: str = Field(..., description="Partition split of first example")
    example_id_2: str = Field(..., description="Second example ID")
    split_2: str = Field(..., description="Partition split of second example")
    matched_content: str = Field(
        ..., description="Normalized text or SQL that produced the exact collision"
    )
    db_id_1: str = Field(..., description="Database ID of first example")
    db_id_2: str = Field(..., description="Database ID of second example")


class FuzzyOverlapDetail(BaseModel):
    """Record of a fuzzy n-gram lexical similarity match exceeding the audit threshold."""

    model_config = ConfigDict(frozen=True)

    example_id_1: str = Field(..., description="First example ID")
    split_1: str = Field(..., description="Partition split of first example")
    example_id_2: str = Field(..., description="Second example ID")
    split_2: str = Field(..., description="Partition split of second example")
    question_1: str = Field(..., description="Question text of first example")
    question_2: str = Field(..., description="Question text of second example")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="N-gram Jaccard similarity coefficient"
    )
    db_id_1: str = Field(..., description="Database ID of first example")
    db_id_2: str = Field(..., description="Database ID of second example")


class SchemaDisjointnessViolation(BaseModel):
    """Violation record where evaluation databases overlap with training databases."""

    model_config = ConfigDict(frozen=True)

    db_id: str = Field(..., description="Overlapping database identifier")
    split_1: str = Field(..., description="First partition split (typically 'train')")
    split_2: str = Field(..., description="Second partition split (e.g. 'dev', 'test', 'held_out')")
    count_split_1: int = Field(
        ..., ge=1, description="Number of examples using db_id in first split"
    )
    count_split_2: int = Field(
        ..., ge=1, description="Number of examples using db_id in second split"
    )


class SplitLeakageViolation(BaseModel):
    """Violation record where an example designated for evaluation appears in training."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Violating example identifier")
    intended_split: str = Field(
        ..., description="Declared evaluation split ('dev', 'test', 'held_out')"
    )
    found_in_split: str = Field(..., description="Partition where example was incorrectly found")


class ContaminationReport(BaseModel):
    """Machine-readable report summarizing cross-partition contamination and leakage checks."""

    model_config = ConfigDict(frozen=True)

    audit_id: str = Field(..., description="Unique audit run identifier")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of audit execution")
    exact_question_duplicates: list[DuplicateDetail] = Field(
        default_factory=list,
        description="Normalized exact duplicate questions across distinct splits",
    )
    exact_sql_duplicates: list[DuplicateDetail] = Field(
        default_factory=list,
        description="Normalized exact duplicate SQL queries across distinct splits",
    )
    fuzzy_question_overlaps: list[FuzzyOverlapDetail] = Field(
        default_factory=list,
        description="Fuzzy n-gram similarity matches exceeding the audit threshold",
    )
    schema_disjointness_violations: list[SchemaDisjointnessViolation] = Field(
        default_factory=list,
        description="Databases shared between training and evaluation splits",
    )
    split_leakage_violations: list[SplitLeakageViolation] = Field(
        default_factory=list,
        description="Evaluation examples found inside training partitions",
    )
    n_gram_size: int = Field(
        default=4, ge=1, description="Word n-gram size used for fuzzy overlap calculation"
    )
    fuzzy_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Jaccard threshold above which pairs are flagged as fuzzy overlaps",
    )
    total_pairs_evaluated: int = Field(
        ..., ge=0, description="Total cross-partition question pairs evaluated"
    )
    passed: bool = Field(
        ..., description="True if no critical leakage or disjointness violations were found"
    )
    summary: str = Field(..., description="Human-readable summary of audit findings")
