"""Unit tests for the local-first experiment tracker."""

import json
from pathlib import Path

import pytest

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.schemas.evaluation import EvaluationMetrics
from sqlforge.schemas.experiments import ExperimentConfig


def test_tracker_init_and_finish_run(temp_artifact_dir: Path) -> None:
    """Test full lifecycle of a run: init, log metrics, and finish."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)

    exp_config = ExperimentConfig(
        experiment_id="unit_test_run",
        description="Testing tracker lifecycle",
        seed=100,
        model_id="qwen25_coder_1_5b",
    )

    metadata = tracker.init_run(
        experiment_name="unit_test_run",
        config=exp_config,
        seed=100,
    )

    assert metadata.status == "running"
    assert metadata.seed == 100
    assert metadata.config_hash is not None

    run_dir = temp_artifact_dir / metadata.run_id
    assert run_dir.is_dir()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "run_metadata.json").is_file()

    # Log metrics
    metrics = EvaluationMetrics(
        total_examples=10,
        valid_sql_rate=1.0,
        exact_match_accuracy=0.8,
        execution_accuracy=0.9,
    )
    metrics_path = tracker.log_metrics(metadata.run_id, metrics)
    assert metrics_path.is_file()

    # Finish run
    finished_meta = tracker.finish_run(metadata.run_id, status="completed")
    assert finished_meta.status == "completed"

    # Read back
    retrieved = tracker.get_run(metadata.run_id)
    assert retrieved is not None
    assert retrieved.status == "completed"


def test_tracker_non_existent_run_raises(temp_artifact_dir: Path) -> None:
    """Test error raised when logging to non-existent run."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    with pytest.raises(FileNotFoundError):
        tracker.finish_run("non_existent_run_id")


def test_tracker_collision_safety(temp_artifact_dir: Path) -> None:
    """Verify that ExperimentTracker refuses to silently overwrite existing populated runs."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    fixed_id = "test_fixed_collision_run"

    # Initialize first run
    tracker.init_run(
        experiment_name="first_run",
        config={"key": "val1"},
        seed=42,
        run_id=fixed_id,
    )

    # Attempting to re-init with the same run_id must raise FileExistsError
    with pytest.raises(FileExistsError, match="Collision detected"):
        tracker.init_run(
            experiment_name="colliding_run",
            config={"key": "val2"},
            seed=99,
            run_id=fixed_id,
        )


def test_tracker_manifest_and_run_verification(temp_artifact_dir: Path) -> None:
    """Verify cryptographic manifest generation and run verification."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(
        experiment_name="manifest_test",
        config={"model": "test"},
        seed=42,
    )

    # Log metrics and finish
    tracker.log_metrics(
        meta.run_id,
        {
            "total_examples": 1,
            "valid_sql_rate": 1.0,
            "exact_match_accuracy": 1.0,
            "execution_accuracy": 1.0,
        },
    )
    tracker.finish_run(meta.run_id, status="completed")

    # Run verification must succeed
    res = tracker.verify_run(meta.run_id)
    assert res["verified"] is True
    assert res["status"] == "completed"
    assert len(res["missing_files"]) == 0
    assert len(res["hash_mismatches"]) == 0

    # Tamper with config file to verify hash mismatch detection
    run_dir = temp_artifact_dir / meta.run_id
    config_file = run_dir / "config.yaml"
    config_file.write_text("tampered: true\n", encoding="utf-8")

    tampered_res = tracker.verify_run(meta.run_id)
    assert tampered_res["verified"] is False
    assert "config.yaml" in tampered_res["hash_mismatches"]


def test_tracker_generation_and_anomaly_logging(temp_artifact_dir: Path) -> None:
    """Verify logging of per-example generations and anomalies."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(experiment_name="log_test", config={"test": True})

    # Log generation
    gen_file = tracker.log_generation(
        meta.run_id,
        {"example_id": "spider_01", "generated_sql": "SELECT 1;", "latency_ms": 12.5},
    )
    assert gen_file.is_file()
    assert "spider_01" in gen_file.read_text(encoding="utf-8")

    # Log anomaly
    anomaly_file = tracker.log_anomaly(
        meta.run_id,
        {"example_id": "spider_02", "error_code": "E09", "reason": "Query execution timed out"},
    )
    assert anomaly_file.is_file()
    assert "E09" in anomaly_file.read_text(encoding="utf-8")

    # Repeated append to verify non-destructive appending
    tracker.log_anomaly(
        meta.run_id,
        {"example_id": "spider_03", "error_code": "E08", "reason": "Syntax error"},
    )
    import json

    anomalies = json.loads(anomaly_file.read_text(encoding="utf-8"))
    assert len(anomalies) == 2
    assert anomalies[0]["example_id"] == "spider_02"
    assert anomalies[1]["example_id"] == "spider_03"


def test_tracker_invalid_run_ids_rejected(temp_artifact_dir: Path) -> None:
    """Verify that invalid run IDs are rejected before any filesystem modification."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)

    invalid_ids = [
        "../traversal",
        "..\\windows_traversal",
        "/absolute/path",
        "nested/path",
        "has spaces",
        "invalid$char",
        "",
        "a" * 129,
    ]

    for bad_id in invalid_ids:
        with pytest.raises(ValueError):
            tracker.init_run(experiment_name="test", config={}, run_id=bad_id)


def test_tracker_empty_directory_collision(temp_artifact_dir: Path) -> None:
    """Verify that ExperimentTracker rejects collisions even if the target directory is empty."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    target_id = "pre_existing_empty_dir"
    empty_dir = temp_artifact_dir / target_id
    empty_dir.mkdir(parents=True, exist_ok=False)

    with pytest.raises(FileExistsError, match="Collision detected"):
        tracker.init_run(
            experiment_name="test",
            config={},
            run_id=target_id,
        )


def test_tracker_corrupted_anomaly_log_preserves_content(temp_artifact_dir: Path) -> None:
    """Verify that corrupted anomaly logs raise ValueError and preserve data instead of overwriting."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(experiment_name="anomaly_corruption_test", config={})
    anomaly_path = temp_artifact_dir / meta.run_id / "eval_anomalies.json"

    # Write corrupted JSON
    corrupted_content = "{ malformed json: not valid ... "
    anomaly_path.write_text(corrupted_content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"Corrupted anomaly log"):
        tracker.log_anomaly(meta.run_id, {"example_id": "exp1", "error": "test"})

    # Content must NOT have been overwritten or truncated
    assert anomaly_path.read_text(encoding="utf-8") == corrupted_content

    # Write valid JSON but wrong top-level structure (dict instead of list)
    dict_content = '{"some_key": "some_value"}'
    anomaly_path.write_text(dict_content, encoding="utf-8")

    with pytest.raises(ValueError, match="expected top-level list"):
        tracker.log_anomaly(meta.run_id, {"example_id": "exp1", "error": "test"})

    assert anomaly_path.read_text(encoding="utf-8") == dict_content


def test_tracker_verify_run_strict_vs_lenient(temp_artifact_dir: Path) -> None:
    """Verify strict manifest enforcement and detection of untracked or corrupted records."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(experiment_name="strict_test", config={"seed": 42})

    # Unfinished run has no manifest.json
    # Strict verification should fail
    strict_res = tracker.verify_run(meta.run_id, strict=True)
    assert strict_res["verified"] is False
    assert strict_res["missing_manifest"] is True

    # Lenient verification allows missing manifest as long as metadata is valid
    lenient_res = tracker.verify_run(meta.run_id, strict=False)
    assert lenient_res["verified"] is True
    assert lenient_res["missing_manifest"] is True

    # Log metrics before completing, as completed runs require metrics.json
    tracker.log_metrics(
        meta.run_id,
        {"total_examples": 1, "exact_match_accuracy": 1.0, "execution_accuracy": 1.0},
    )

    # Finish run cleanly
    tracker.finish_run(meta.run_id, status="completed")

    clean_res = tracker.verify_run(meta.run_id, strict=True)
    assert clean_res["verified"] is True
    assert clean_res["missing_manifest"] is False
    assert len(clean_res["untracked_files"]) == 0

    # Add an untracked file to run dir
    run_dir = temp_artifact_dir / meta.run_id
    (run_dir / "untracked_rogue_file.txt").write_text("rogue", encoding="utf-8")

    untracked_res = tracker.verify_run(meta.run_id, strict=True)
    assert untracked_res["verified"] is False
    assert "untracked_rogue_file.txt" in untracked_res["untracked_files"]

    # Corrupt a record file (e.g. generations.jsonl)
    gen_file = run_dir / "generations.jsonl"
    gen_file.write_text("not a valid json line\n", encoding="utf-8")

    # In strict mode, corrupted records must be detected
    corrupt_res = tracker.verify_run(meta.run_id, strict=True)
    assert corrupt_res["verified"] is False
    assert any("generations.jsonl" in err for err in corrupt_res["errors"])


def test_tracker_verify_manifest_path_containment_and_traversal(
    temp_artifact_dir: Path,
) -> None:
    """Verify that manifest entries with path traversal, absolute paths, or escaping paths fail verification."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(experiment_name="adversarial_manifest_test", config={})
    tracker.log_metrics(
        meta.run_id,
        {"total_examples": 1, "exact_match_accuracy": 1.0, "execution_accuracy": 1.0},
    )
    tracker.finish_run(meta.run_id, status="completed")

    run_dir = temp_artifact_dir / meta.run_id
    manifest_path = run_dir / "manifest.json"

    # Inject adversarial path traversal and absolute path entries
    adversarial_manifest = {
        "run_metadata.json": hashlib_sha256(run_dir / "run_metadata.json"),
        "config.yaml": hashlib_sha256(run_dir / "config.yaml"),
        "metrics.json": hashlib_sha256(run_dir / "metrics.json"),
        "../outside_escape.txt": "a" * 64,
        "/absolute/root_escape.txt": "b" * 64,
        "nested/../../traversal.txt": "c" * 64,
        "\\windows_root_escape.txt": "d" * 64,
    }
    manifest_path.write_text(json.dumps(adversarial_manifest, indent=2), encoding="utf-8")

    res = tracker.verify_run(meta.run_id, strict=True)
    assert res["verified"] is False
    assert any("is an invalid or traversal path" in err for err in res["errors"])


def test_tracker_verify_manifest_malformed_hashes(temp_artifact_dir: Path) -> None:
    """Verify that malformed, non-hex, or wrong-length SHA-256 values in manifest fail verification."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(experiment_name="malformed_hash_test", config={})
    tracker.log_metrics(
        meta.run_id,
        {"total_examples": 1, "exact_match_accuracy": 1.0, "execution_accuracy": 1.0},
    )
    tracker.finish_run(meta.run_id, status="completed")

    run_dir = temp_artifact_dir / meta.run_id
    manifest_path = run_dir / "manifest.json"

    malformed_manifest = {
        "run_metadata.json": "not_64_hex_chars",
        "config.yaml": "z" * 64,  # 'z' is not valid hex
        "metrics.json": 12345678,  # not a string
    }
    manifest_path.write_text(json.dumps(malformed_manifest, indent=2), encoding="utf-8")

    res = tracker.verify_run(meta.run_id, strict=True)
    assert res["verified"] is False
    assert any("Malformed SHA-256 hash" in err for err in res["errors"])


def test_tracker_verify_nested_legitimate_artifacts(temp_artifact_dir: Path) -> None:
    """Verify that legitimate nested artifact directories are tracked and verified correctly."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(experiment_name="nested_artifact_test", config={})
    tracker.log_metrics(
        meta.run_id,
        {"total_examples": 1, "exact_match_accuracy": 1.0, "execution_accuracy": 1.0},
    )

    run_dir = temp_artifact_dir / meta.run_id
    nested_dir = run_dir / "checkpoints" / "step_100"
    nested_dir.mkdir(parents=True, exist_ok=True)
    nested_file = nested_dir / "adapter_config.json"
    nested_file.write_text('{"lora_r": 16}', encoding="utf-8")

    # Complete run, which will compute manifest across all files including nested
    tracker.finish_run(meta.run_id, status="completed")

    res = tracker.verify_run(meta.run_id, strict=True)
    assert res["verified"] is True
    assert len(res["missing_files"]) == 0
    assert len(res["hash_mismatches"]) == 0
    assert len(res["untracked_files"]) == 0


def test_tracker_verify_artifact_schemas_and_failed_runs(
    temp_artifact_dir: Path,
) -> None:
    """Verify artifact schema validation (metrics.json, eval_anomalies.json) and failed run rules."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)

    # 1. Failed run without metrics.json: valid failure state
    meta_failed = tracker.init_run(experiment_name="failed_run_test", config={})
    tracker.log_anomaly(
        meta_failed.run_id,
        {"error_code": "PIPELINE_ERROR", "reason": "Simulated component crash"},
    )
    tracker.finish_run(meta_failed.run_id, status="failed")

    failed_res = tracker.verify_run(meta_failed.run_id, strict=True)
    # A cleanly recorded failed run has manifest, valid metadata, valid config, and valid anomalies
    assert failed_res["verified"] is True
    assert failed_res["status"] == "failed"
    assert "metrics.json" not in failed_res["missing_files"]

    # 2. Corrupt metrics.json schema (e.g. invalid bounds or missing required total_examples)
    meta_bad_metrics = tracker.init_run(experiment_name="bad_metrics_test", config={})
    run_dir = temp_artifact_dir / meta_bad_metrics.run_id
    bad_metrics_file = run_dir / "metrics.json"
    bad_metrics_file.write_text(
        json.dumps({"invalid_field": "no_total_examples"}), encoding="utf-8"
    )
    tracker.finish_run(meta_bad_metrics.run_id, status="completed")

    bad_metrics_res = tracker.verify_run(meta_bad_metrics.run_id, strict=True)
    assert bad_metrics_res["verified"] is False
    assert any("Invalid metrics.json" in err for err in bad_metrics_res["errors"])

    # 3. Corrupt eval_anomalies.json structure (not a list)
    meta_bad_anomalies = tracker.init_run(experiment_name="bad_anomalies_test", config={})
    tracker.log_metrics(
        meta_bad_anomalies.run_id,
        {"total_examples": 1, "exact_match_accuracy": 1.0, "execution_accuracy": 1.0},
    )
    run_dir_anom = temp_artifact_dir / meta_bad_anomalies.run_id
    (run_dir_anom / "eval_anomalies.json").write_text(
        json.dumps({"should_be": "a_list"}), encoding="utf-8"
    )
    tracker.finish_run(meta_bad_anomalies.run_id, status="completed")

    bad_anom_res = tracker.verify_run(meta_bad_anomalies.run_id, strict=True)
    assert bad_anom_res["verified"] is False
    assert any("Invalid eval_anomalies.json" in err for err in bad_anom_res["errors"])


def hashlib_sha256(path: Path) -> str:
    """Helper to compute sha256 hex string."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
