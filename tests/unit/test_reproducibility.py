"""Unit tests for reproducibility utilities."""

import random
from pathlib import Path

import numpy as np

from sqlforge.reproducibility import (
    generate_run_id,
    get_environment_metadata,
    get_git_metadata,
    hash_dict,
    hash_file,
    set_seed,
)


def test_set_seed_reproducibility() -> None:
    """Verify that set_seed ensures reproducible random draws."""
    set_seed(12345)
    py_draw_1 = [random.random() for _ in range(5)]
    np_draw_1 = np.random.rand(5).tolist()

    set_seed(12345)
    py_draw_2 = [random.random() for _ in range(5)]
    np_draw_2 = np.random.rand(5).tolist()

    assert py_draw_1 == py_draw_2
    assert np.allclose(np_draw_1, np_draw_2)


def test_generate_run_id() -> None:
    """Verify run ID format and uniqueness."""
    id1 = generate_run_id("test_exp")
    id2 = generate_run_id("test_exp")

    assert id1.startswith("test_exp_")
    assert id2.startswith("test_exp_")
    assert id1 != id2


def test_get_git_metadata() -> None:
    """Verify Git metadata inspection."""
    meta = get_git_metadata()
    assert isinstance(meta, dict)
    assert "commit" in meta
    assert "branch" in meta
    assert "is_dirty" in meta


def test_get_environment_metadata() -> None:
    """Verify runtime environment metadata collection."""
    meta = get_environment_metadata()
    assert "python_version" in meta
    assert "platform" in meta
    assert "cpu_count" in meta
    assert "total_ram_gb" in meta
    assert "gpu" in meta
    assert meta["cpu_count"] >= 1


def test_hash_dict_deterministic() -> None:
    """Verify hash_dict is deterministic regardless of key insertion order."""
    d1 = {"b": 2, "a": 1, "c": [1, 2, 3]}
    d2 = {"a": 1, "c": [1, 2, 3], "b": 2}
    assert hash_dict(d1) == hash_dict(d2)


def test_hash_file(tmp_path: Path) -> None:
    """Verify hash_file calculates correct SHA256."""
    f = tmp_path / "sample.txt"
    f.write_text("sqlforge research", encoding="utf-8")
    h1 = hash_file(f)
    assert len(h1) == 64
