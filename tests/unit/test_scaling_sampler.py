"""Unit tests for deterministic nested stratified Spider data sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.training.data_scaling import (
    DataScalingSamplingError,
)
from sqlforge.training.scaling_sampler import (
    NestedStratifiedSpiderSampler,
    SamplingManifest,
)


def _create_mock_spider_dataset(
    count: int = 100, db_count: int = 5, split: DatasetSplit = DatasetSplit.TRAIN
) -> list[TextToSQLExample]:
    """Generate deterministic mock records across multiple databases."""
    examples: list[TextToSQLExample] = []
    for i in range(count):
        db_id = f"db_{i % db_count:02d}"
        ex_id = f"spider_train_{i:05d}"
        examples.append(
            TextToSQLExample(
                id=ex_id,
                question=f"What is the total value for record {i}?",
                db_id=db_id,
                gold_sql=f"SELECT val FROM t WHERE id = {i};",
                dataset_name="spider",
                split=split,
            )
        )
    return examples


def test_sampler_exact_sample_counts() -> None:
    """Verify sampler extracts the exact requested sample counts."""
    data = _create_mock_spider_dataset(count=100, db_count=5)
    sampler = NestedStratifiedSpiderSampler(
        dataset=data,
        source_dataset_manifest_hash="test_hash_123",
    )

    manifest_20 = sampler.sample(sample_size=20, seed=42)
    assert len(manifest_20.selected_example_ids) == 20
    assert manifest_20.sample_size == 20

    manifest_50 = sampler.sample(sample_size=50, seed=42)
    assert len(manifest_50.selected_example_ids) == 50
    assert manifest_50.sample_size == 50


def test_sampler_determinism_and_stability() -> None:
    """Verify repeated sampling with identical inputs produces identical IDs and hashes."""
    data = _create_mock_spider_dataset(count=100, db_count=5)
    sampler = NestedStratifiedSpiderSampler(
        dataset=data,
        source_dataset_manifest_hash="test_hash_123",
    )

    m1 = sampler.sample(sample_size=30, seed=42)
    m2 = sampler.sample(sample_size=30, seed=42)

    assert m1.selected_example_ids == m2.selected_example_ids
    assert m1.selected_example_ids_sha256 == m2.selected_example_ids_sha256
    assert m1.database_distribution == m2.database_distribution


def test_sampler_seed_independence() -> None:
    """Verify different seeds produce distinct permutations."""
    data = _create_mock_spider_dataset(count=100, db_count=5)
    sampler = NestedStratifiedSpiderSampler(
        dataset=data,
        source_dataset_manifest_hash="test_hash_123",
    )

    m_s42 = sampler.sample(sample_size=40, seed=42)
    m_s43 = sampler.sample(sample_size=40, seed=43)
    m_s44 = sampler.sample(sample_size=40, seed=44)

    assert m_s42.selected_example_ids != m_s43.selected_example_ids
    assert m_s43.selected_example_ids != m_s44.selected_example_ids
    assert m_s42.selected_example_ids_sha256 != m_s43.selected_example_ids_sha256


def test_sampler_nested_subset_guarantee() -> None:
    """Verify smaller sample sizes are exact strictly ordered prefixes of larger sample sizes."""
    data = _create_mock_spider_dataset(count=120, db_count=6)
    sampler = NestedStratifiedSpiderSampler(
        dataset=data,
        source_dataset_manifest_hash="test_hash_123",
    )

    sizes = [10, 25, 50, 100]
    manifests: list[SamplingManifest] = []
    parent_hash: str | None = None

    for sz in sizes:
        m = sampler.sample(sample_size=sz, seed=42, parent_subset_hash=parent_hash)
        manifests.append(m)
        parent_hash = m.selected_example_ids_sha256

    # Verify nesting: IDs at size N are exact prefixes of size M (N < M)
    for i in range(len(sizes) - 1):
        smaller = manifests[i].selected_example_ids
        larger = manifests[i + 1].selected_example_ids
        assert larger[: len(smaller)] == smaller
        assert set(smaller).issubset(set(larger))
        assert manifests[i + 1].parent_subset_hash == manifests[i].selected_example_ids_sha256


def test_sampler_proportional_fair_share_stratification() -> None:
    """Verify fair-share scheduling distributes selections across database groups proportionally."""
    # Group sizes: db_00: 50, db_01: 30, db_02: 20 -> Total 100
    examples: list[TextToSQLExample] = []
    for i in range(50):
        examples.append(
            TextToSQLExample(
                id=f"db0_ex_{i}",
                question="q",
                db_id="db_00",
                gold_sql="SELECT 1;",
                dataset_name="spider",
                split=DatasetSplit.TRAIN,
            )
        )
    for i in range(30):
        examples.append(
            TextToSQLExample(
                id=f"db1_ex_{i}",
                question="q",
                db_id="db_01",
                gold_sql="SELECT 1;",
                dataset_name="spider",
                split=DatasetSplit.TRAIN,
            )
        )
    for i in range(20):
        examples.append(
            TextToSQLExample(
                id=f"db2_ex_{i}",
                question="q",
                db_id="db_02",
                gold_sql="SELECT 1;",
                dataset_name="spider",
                split=DatasetSplit.TRAIN,
            )
        )

    sampler = NestedStratifiedSpiderSampler(
        dataset=examples,
        source_dataset_manifest_hash="hash_123",
    )
    manifest = sampler.sample(sample_size=10, seed=42)
    dist = manifest.database_distribution

    # Total 10 sampled from 50/30/20 ratio (50%, 30%, 20%)
    # Fair share should yield exactly 5, 3, 2 or within 1
    assert dist["db_00"] == 5
    assert dist["db_01"] == 3
    assert dist["db_02"] == 2


def test_sampler_no_duplicate_ids() -> None:
    """Verify sampler never contains duplicate IDs."""
    data = _create_mock_spider_dataset(count=80, db_count=4)
    sampler = NestedStratifiedSpiderSampler(
        dataset=data,
        source_dataset_manifest_hash="test_hash_123",
    )
    m = sampler.sample(sample_size=80, seed=42)
    assert len(m.selected_example_ids) == len(set(m.selected_example_ids))


def test_sampler_insufficient_eligible_data() -> None:
    """Verify sampler fails explicitly when fewer records exist than requested."""
    data = _create_mock_spider_dataset(count=50, db_count=5)
    sampler = NestedStratifiedSpiderSampler(
        dataset=data,
        source_dataset_manifest_hash="test_hash_123",
    )
    with pytest.raises(DataScalingSamplingError, match="Insufficient eligible records"):
        sampler.sample(sample_size=100, seed=42)


def test_sampler_rejects_eval_splits() -> None:
    """Verify sampler rejects records belonging to DEV, TEST, or non-train splits."""
    dev_data = _create_mock_spider_dataset(count=50, db_count=5, split=DatasetSplit.DEV)
    with pytest.raises(DataScalingSamplingError, match="non-train split"):
        NestedStratifiedSpiderSampler(
            dataset=dev_data,
            source_dataset_manifest_hash="test_hash_123",
        )


def test_sampler_rejects_duplicate_input_ids() -> None:
    """Verify sampler detects duplicate IDs in input dataset."""
    data = _create_mock_spider_dataset(count=10, db_count=2)
    # Inject duplicate
    dup_example = TextToSQLExample(
        id=data[0].id,
        question="Another question",
        db_id="db_00",
        gold_sql="SELECT 2;",
        dataset_name="spider",
        split=DatasetSplit.TRAIN,
    )
    with pytest.raises(DataScalingSamplingError, match="Duplicate example IDs"):
        NestedStratifiedSpiderSampler(
            dataset=data + [dup_example],
            source_dataset_manifest_hash="test_hash_123",
        )


def test_sampler_rejects_empty_dataset() -> None:
    """Verify sampler rejects empty dataset."""
    with pytest.raises(DataScalingSamplingError, match="Empty dataset"):
        NestedStratifiedSpiderSampler(
            dataset=[],
            source_dataset_manifest_hash="test_hash_123",
        )


def test_sampler_rejects_missing_gold_sql() -> None:
    """Verify sampler rejects examples missing gold SQL."""
    data = _create_mock_spider_dataset(count=5, db_count=1)
    bad_ex = TextToSQLExample(
        id="bad_01",
        question="question without sql",
        db_id="db_00",
        gold_sql=None,  # Missing
        dataset_name="spider",
        split=DatasetSplit.TRAIN,
    )
    with pytest.raises(DataScalingSamplingError, match="Missing or empty gold_sql"):
        NestedStratifiedSpiderSampler(
            dataset=data + [bad_ex],
            source_dataset_manifest_hash="test_hash_123",
        )


def test_sampler_manifest_serialization_and_hash(tmp_path: Path) -> None:
    """Verify manifest can be saved and verified, and tampering is detected."""
    data = _create_mock_spider_dataset(count=50, db_count=5)
    sampler = NestedStratifiedSpiderSampler(
        dataset=data,
        source_dataset_manifest_hash="test_hash_123",
    )
    manifest = sampler.sample(sample_size=20, seed=42)

    manifest_file = tmp_path / "exp05_n20_s42.json"
    manifest.save_json(manifest_file)
    assert manifest_file.exists()

    loaded = SamplingManifest.load_json(manifest_file)
    assert loaded.selected_example_ids == manifest.selected_example_ids
    assert loaded.selected_example_ids_sha256 == manifest.selected_example_ids_sha256

    # Tampering test: Modify an ID
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    raw["selected_example_ids"][0] = "tampered_id"
    manifest_file.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(DataScalingSamplingError, match="Manifest checksum mismatch"):
        SamplingManifest.load_json(manifest_file)
