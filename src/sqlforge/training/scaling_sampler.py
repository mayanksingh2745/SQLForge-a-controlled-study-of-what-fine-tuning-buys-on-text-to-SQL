"""Deterministic nested stratified sampling for EXP-05 and EXP-06 data scaling.

Implements NestedStratifiedSpiderSampler with cryptographic SHA-256 ranking,
proportional fair-share database stratification, strict dataset partition isolation,
and nested-subset guarantees (N=500 subset N=1000 subset N=2000 subset N=5000 subset N=7000).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.training.data_scaling import (
    EXP05_EXPERIMENT_ID,
    EXP05_TRAINING_SIZES,
    TRAIN_SPLIT_SPIDER,
    DataScalingSamplingError,
)


class SamplingManifest(BaseModel):
    """Machine-readable manifest recording deterministic sampling provenance and subset membership."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    seed: int
    sample_size: int
    source_dataset: str
    source_dataset_manifest_hash: str
    sampling_algorithm: str = "nested_stratified_spider_sampler"
    sampling_algorithm_version: str = "1.0.0"
    selected_example_ids: list[str]
    selected_example_ids_sha256: str
    database_distribution: dict[str, int]
    created_at: str
    is_nested: bool = True
    parent_subset_hash: str | None = None

    def compute_manifest_hash(self) -> str:
        """Compute canonical SHA-256 hash of this manifest's contents."""
        payload = json.dumps(self.model_dump(), sort_keys=True, indent=2)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_canonical_json(self) -> str:
        """Serialize manifest to canonical, deterministic JSON."""
        return json.dumps(self.model_dump(), sort_keys=True, indent=2)

    def save_json(self, path: Path | str) -> None:
        """Save manifest atomically to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self.to_canonical_json())
        tmp.replace(p)

    @classmethod
    def load_json(cls, path: Path | str) -> SamplingManifest:
        """Load manifest from JSON file and verify checksum."""
        manifest = cls.load_from_file(path)
        computed_hash = hashlib.sha256(
            "\n".join(manifest.selected_example_ids).encode("utf-8")
        ).hexdigest()
        if computed_hash != manifest.selected_example_ids_sha256:
            raise DataScalingSamplingError(
                f"Manifest checksum mismatch for {path}: expected {manifest.selected_example_ids_sha256}, got {computed_hash}"
            )
        return manifest

    @classmethod
    def load_from_file(cls, path: Path | str) -> SamplingManifest:
        """Load and validate manifest from a JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Sampling manifest not found: {p}")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class NestedStratifiedSpiderSampler:
    """Deterministic, nested, database-stratified data sampler for Spider training partitions."""

    FORBIDDEN_SPLITS: tuple[DatasetSplit, ...] = (
        DatasetSplit.DEV,
        DatasetSplit.TEST,
        DatasetSplit.HELD_OUT,
    )

    def __init__(
        self,
        dataset: list[TextToSQLExample] | None = None,
        source_dataset_manifest_hash: str = "0" * 64,
    ) -> None:
        self.source_dataset_manifest_hash = source_dataset_manifest_hash
        self.dataset: list[TextToSQLExample] | None = None
        if dataset is not None:
            self.validate_eligibility(dataset, required_sample_size=0)
            self.dataset = dataset

    def validate_eligibility(
        self,
        examples: list[TextToSQLExample],
        required_sample_size: int = 0,
    ) -> list[TextToSQLExample]:
        """Validate that all candidate examples satisfy strict eligibility and isolation rules.

        Raises:
            DataScalingSamplingError: If any example violates isolation, integrity, or counts.
        """
        if not examples:
            raise DataScalingSamplingError("Empty dataset.")

        if required_sample_size > 0 and len(examples) < required_sample_size:
            raise DataScalingSamplingError(
                f"Insufficient eligible records: available {len(examples)}, required {required_sample_size}."
            )

        seen_ids: set[str] = set()
        eligible: list[TextToSQLExample] = []

        for idx, ex in enumerate(examples):
            # 1. Unique stable ID check
            if not ex.id or not isinstance(ex.id, str) or not ex.id.strip():
                raise DataScalingSamplingError(f"Example at index {idx} has missing or invalid ID.")
            if ex.id in seen_ids:
                raise DataScalingSamplingError(f"Duplicate example IDs detected: {ex.id!r}")
            seen_ids.add(ex.id)

            # 2. Strict dataset partition isolation
            if ex.split in self.FORBIDDEN_SPLITS or ex.split != DatasetSplit.TRAIN:
                raise DataScalingSamplingError(
                    f"Example {ex.id!r} has unauthorized non-train split {ex.split.value!r}. "
                    "Only spider:train is eligible for training optimization."
                )

            # 3. Origin dataset check
            if ex.dataset_name.lower() != "spider":
                raise DataScalingSamplingError(
                    f"Example {ex.id!r} originates from {ex.dataset_name!r}, not 'spider'."
                )

            # 4. Target SQL validity check
            if not ex.gold_sql or not isinstance(ex.gold_sql, str) or not ex.gold_sql.strip():
                raise DataScalingSamplingError(
                    f"Example {ex.id!r} has Missing or empty gold_sql target."
                )

            # 5. Database/schema reference check
            if not ex.db_id or not isinstance(ex.db_id, str) or not ex.db_id.strip():
                raise DataScalingSamplingError(
                    f"Example {ex.id!r} has missing or empty database reference (db_id)."
                )

            eligible.append(ex)

        return eligible

    def generate_global_order(
        self,
        eligible_records: list[TextToSQLExample],
        seed: int,
    ) -> list[TextToSQLExample]:
        """Construct one deterministic global sample order using proportional fair-share scheduling.

        Algorithm:
        1. Sort eligible records by stable example ID ascending.
        2. Partition records by database ID.
        3. Sort database IDs lexicographically.
        4. For each database group, compute a cryptographic ranking key:
           hash_key = sha256(f"{seed}:{db_id}:{ex.id}".encode()).hexdigest()
           Sort records within that database queue by (hash_key, ex.id).
        5. Interleave queues via proportional fair-share scheduling:
           At each step, select from the nonempty database group with the smallest ratio of
           already-selected examples to its original eligible group size:
           ratio = selected_count / total_eligible_in_group.
           Ties are broken lexicographically by db_id.
        """
        # 1. Sort records by example ID for stable baseline
        sorted_records = sorted(eligible_records, key=lambda x: x.id)

        # 2 & 3. Partition by database ID and sort db_ids lexicographically
        db_groups: dict[str, list[TextToSQLExample]] = {}
        for ex in sorted_records:
            db_groups.setdefault(ex.db_id, []).append(ex)

        sorted_db_ids = sorted(db_groups.keys())

        # 4. Deterministic per-database permutation via cryptographic SHA-256 ranking
        per_db_queues: dict[str, list[TextToSQLExample]] = {}
        original_sizes: dict[str, int] = {}

        for db_id in sorted_db_ids:
            records = db_groups[db_id]
            original_sizes[db_id] = len(records)

            def get_sort_key(
                record: TextToSQLExample, _seed: int = seed, _db: str = db_id
            ) -> tuple[str, str]:
                payload = f"{_seed}:{_db}:{record.id}".encode()
                h = hashlib.sha256(payload).hexdigest()
                return (h, record.id)

            per_db_queues[db_id] = sorted(records, key=get_sort_key)

        # 5. Interleave queues with proportional fair-share scheduling
        selected_counts: dict[str, int] = {db_id: 0 for db_id in sorted_db_ids}
        global_order: list[TextToSQLExample] = []
        total_records = len(sorted_records)

        while len(global_order) < total_records:
            # Candidate groups that still have unselected examples
            available_dbs = [db_id for db_id in sorted_db_ids if per_db_queues[db_id]]
            if not available_dbs:
                break

            # Find group with smallest ratio = selected_counts[db] / original_sizes[db]
            # Break ties lexicographically by db_id
            best_db = min(
                available_dbs,
                key=lambda db: (selected_counts[db] / original_sizes[db], db),
            )

            # Pop next record from chosen queue
            chosen_record = per_db_queues[best_db].pop(0)
            selected_counts[best_db] += 1
            global_order.append(chosen_record)

        return global_order

    def sample(
        self,
        sample_size: int,
        seed: int,
        examples: list[TextToSQLExample] | None = None,
        parent_subset_hash: str | None = None,
        experiment_id: str = EXP05_EXPERIMENT_ID,
    ) -> SamplingManifest:
        """Sample a single stratified subset and return its manifest."""
        target_examples = examples if examples is not None else self.dataset
        if target_examples is None:
            raise DataScalingSamplingError("No dataset provided to sample.")

        eligible = self.validate_eligibility(target_examples, required_sample_size=sample_size)
        global_order = self.generate_global_order(eligible, seed=seed)
        subset = global_order[:sample_size]
        subset_ids = [ex.id for ex in subset]

        ids_concat = "\n".join(subset_ids).encode("utf-8")
        subset_ids_sha256 = hashlib.sha256(ids_concat).hexdigest()

        db_dist: dict[str, int] = {}
        for ex in subset:
            db_dist[ex.db_id] = db_dist.get(ex.db_id, 0) + 1

        return SamplingManifest(
            experiment_id=experiment_id,
            seed=seed,
            sample_size=sample_size,
            source_dataset=TRAIN_SPLIT_SPIDER,
            source_dataset_manifest_hash=self.source_dataset_manifest_hash,
            sampling_algorithm="nested_stratified_spider_sampler",
            sampling_algorithm_version="1.0.0",
            selected_example_ids=subset_ids,
            selected_example_ids_sha256=subset_ids_sha256,
            database_distribution=db_dist,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            is_nested=True,
            parent_subset_hash=parent_subset_hash,
        )

    def sample_nested_subsets(
        self,
        examples: list[TextToSQLExample],
        seed: int,
        sizes: tuple[int, ...] = EXP05_TRAINING_SIZES,
        experiment_id: str = EXP05_EXPERIMENT_ID,
    ) -> list[tuple[SamplingManifest, list[TextToSQLExample]]]:
        """Produce strictly nested subsets and their corresponding manifests for a given seed.

        Returns list of (manifest, subset_examples) in ascending size order.
        """
        sorted_sizes = sorted(sizes)
        max_size = sorted_sizes[-1]

        # 1. Validate eligibility
        eligible = self.validate_eligibility(examples, required_sample_size=max_size)

        # 2. Generate deterministic global sample order
        global_order = self.generate_global_order(eligible, seed=seed)

        results: list[tuple[SamplingManifest, list[TextToSQLExample]]] = []
        parent_hash: str | None = None

        for size in sorted_sizes:
            subset = global_order[:size]
            subset_ids = [ex.id for ex in subset]

            # Compute subset ID hash
            ids_concat = "\n".join(subset_ids).encode("utf-8")
            subset_ids_sha256 = hashlib.sha256(ids_concat).hexdigest()

            # Compute database distribution
            db_dist: dict[str, int] = {}
            for ex in subset:
                db_dist[ex.db_id] = db_dist.get(ex.db_id, 0) + 1

            manifest = SamplingManifest(
                experiment_id=experiment_id,
                seed=seed,
                sample_size=size,
                source_dataset=TRAIN_SPLIT_SPIDER,
                source_dataset_manifest_hash=self.source_dataset_manifest_hash,
                sampling_algorithm="nested_stratified_spider_sampler",
                sampling_algorithm_version="1.0.0",
                selected_example_ids=subset_ids,
                selected_example_ids_sha256=subset_ids_sha256,
                database_distribution=db_dist,
                created_at=datetime.datetime.now(datetime.UTC).isoformat(),
                is_nested=True,
                parent_subset_hash=parent_hash,
            )

            # Next subset's parent hash is this manifest's content hash
            parent_hash = manifest.compute_manifest_hash()
            results.append((manifest, subset))

        return results

    def save_sampling_manifests(
        self,
        manifest_subsets: list[tuple[SamplingManifest, list[TextToSQLExample]]],
        output_dir: Path | str = "artifacts/data_scaling/exp05",
    ) -> list[Path]:
        """Atomically persist sampling manifests to disk with collision protection."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for manifest, _ in manifest_subsets:
            filename = f"sampling_manifest_n{manifest.sample_size}_s{manifest.seed}.json"
            target_file = out_path / filename
            temp_file = out_path / f"{filename}.tmp"

            content = manifest.to_canonical_json()
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)
            temp_file.replace(target_file)
            saved_paths.append(target_file)

        return saved_paths
