"""Utilities for experimental reproducibility, seed management, and environment auditing."""

import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (if available).

    Note:
        Setting seeds provides repeatability across runs with identical software
        and hardware environments. However, strict bitwise determinism across differing
        GPU architectures, CUDA kernels, or multi-threaded reductions is not guaranteed
        by seed setting alone.

    Args:
        seed: The integer random seed.
        deterministic: If True and PyTorch is available, enforces deterministic algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            with contextlib.suppress(AttributeError):
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def generate_run_id(prefix: str = "run") -> str:
    """Generate a readable, timestamped, unique experiment run identifier.

    Format: <prefix>_<YYYYMMDD_HHMMSS>_<short_uuid>
    Example: 'exp03_20261002_143015_a8f9b2'
    """
    now = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:6]
    clean_prefix = prefix.replace(" ", "_").replace("-", "_").lower()
    return f"{clean_prefix}_{now}_{short_hash}"


def get_git_metadata() -> dict[str, Any]:
    """Inspect Git repository status for provenance auditing."""
    metadata: dict[str, Any] = {
        "commit": None,
        "branch": None,
        "is_dirty": False,
        "error": None,
    }
    try:
        # Check commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["commit"] = commit

        # Check branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["branch"] = branch

        # Check dirty working tree
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["is_dirty"] = len(status) > 0
    except Exception as e:
        metadata["error"] = str(e)

    return metadata


def get_environment_metadata() -> dict[str, Any]:
    """Capture a snapshot of the runtime software and hardware environment."""
    git_meta = get_git_metadata()

    # Hardware info
    cpu_count = os.cpu_count() or 1
    total_ram_gb = 0.0

    try:
        # Check Windows or cross-platform memory
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatus()
            stat.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            total_ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
        elif hasattr(os, "sysconf"):
            attr_name = "sysconf"
            sysconf_fn = getattr(os, attr_name)
            total_ram_gb = round(
                sysconf_fn("SC_PAGE_SIZE") * sysconf_fn("SC_PHYS_PAGES") / (1024**3), 2
            )
    except Exception:
        total_ram_gb = 0.0

    gpu_info: dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
    }

    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            count = torch.cuda.device_count()
            gpu_info["device_count"] = count
            gpu_info["device_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
    except ImportError:
        pass

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "architecture": platform.architecture()[0],
        "cpu_count": cpu_count,
        "total_ram_gb": total_ram_gb,
        "gpu": gpu_info,
        "git": git_meta,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def hash_dict(data: dict[str, Any]) -> str:
    """Compute deterministic SHA256 checksum of a dictionary."""
    canonical_json = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def hash_file(path: Path | str) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
