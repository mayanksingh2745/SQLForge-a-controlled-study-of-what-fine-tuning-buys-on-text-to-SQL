"""Hardware-aware feasibility, pre-flight safety gates, and dependency auditing.

Checks GPU VRAM, system RAM, disk storage, CUDA availability, package versions,
dataset readiness, and configuration plausibility before model training begins.
"""

from __future__ import annotations

import logging
import platform
import shutil
import sys
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig

logger = logging.getLogger(__name__)


class PreflightStatus(StrEnum):
    """Overall feasibility evaluation status."""

    READY = "ready"
    WARNING = "warning"
    BLOCKED = "blocked"


class PreflightCheckItem(BaseModel):
    """Status record of a single feasibility verification check."""

    model_config = ConfigDict(frozen=True)

    category: str = Field(..., description="'hardware', 'dependency', 'dataset', or 'config'")
    name: str = Field(..., description="Check name")
    passed: bool = Field(..., description="True if check passed completely")
    level: str = Field(default="info", description="'info', 'warning', or 'error'")
    message: str = Field(..., description="Diagnostic description")
    recommendation: str | None = Field(default=None, description="Suggested action if failed")


class PreflightReport(BaseModel):
    """Consolidated pre-flight feasibility and environment diagnostic report."""

    model_config = ConfigDict(frozen=True)

    status: PreflightStatus = Field(..., description="Consolidated readiness verdict")
    is_runnable: bool = Field(..., description="True if training can safely proceed on this host")
    platform: str = Field(..., description="Operating system platform details")
    python_version: str = Field(..., description="Python interpreter version")
    cuda_available: bool = Field(..., description="Whether CUDA GPU acceleration is accessible")
    device_count: int = Field(default=0, description="Number of detected GPU devices")
    device_names: list[str] = Field(
        default_factory=list, description="Names of detected GPU devices"
    )
    total_vram_gb: float = Field(default=0.0, description="Total VRAM across GPUs in gigabytes")
    free_vram_gb: float = Field(default=0.0, description="Available free VRAM in gigabytes")
    total_ram_gb: float = Field(default=0.0, description="Host physical RAM in gigabytes")
    free_ram_gb: float = Field(default=0.0, description="Host free physical RAM in gigabytes")
    free_disk_gb: float = Field(
        default=0.0, description="Free disk space on target partition in GB"
    )
    checks: list[PreflightCheckItem] = Field(
        default_factory=list, description="Detailed list of individual checks"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")
    blockers: list[str] = Field(default_factory=list, description="Fatal blocking issues")
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable next steps to resolve issues"
    )

    def summary(self) -> str:
        """Formatted human-readable summary of the pre-flight report."""
        lines = [
            f"Pre-Flight Status: {self.status.upper()}",
            f"Runnable: {self.is_runnable}",
            f"Platform: {self.platform} (Python {self.python_version})",
            f"CUDA: {'Available (' + str(self.device_count) + ' devices)' if self.cuda_available else 'Not Available (CPU only)'}",
            f"Host RAM: {self.free_ram_gb:.1f} GB free / {self.total_ram_gb:.1f} GB total",
            f"Disk Space: {self.free_disk_gb:.1f} GB free",
        ]
        if self.blockers:
            lines.append("Blockers:")
            for b in self.blockers:
                lines.append(f"  - [FAIL] {b}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - [WARN] {w}")
        return "\n".join(lines)


# Minimum VRAM requirements in GB by model size and training method
MINIMUM_VRAM_REQUIREMENTS: dict[str, dict[str, float]] = {
    "1.5b": {"lora": 8.0, "qlora": 6.0},
    "7b": {"lora": 22.0, "qlora": 14.0},
    "8b": {"lora": 24.0, "qlora": 16.0},
    "default": {"lora": 16.0, "qlora": 12.0},
}


class PreflightChecker:
    """Evaluates environment readiness, dependencies, hardware feasibility, and safety gates."""

    REQUIRED_PACKAGES = ("torch", "transformers", "peft", "accelerate")
    QUANTIZATION_PACKAGES = ("bitsandbytes",)

    def __init__(self, config: SFTTrainingConfig | None = None) -> None:
        """Initialize PreflightChecker with optional training configuration."""
        self.config = config or SFTTrainingConfig()

    def run_preflight(self, check_dataset: bool = True) -> PreflightReport:
        """Execute comprehensive pre-flight verification.

        Args:
            check_dataset: Whether to verify dataset files exist on disk.

        Returns:
            PreflightReport with full diagnostic details and status.
        """
        checks: list[PreflightCheckItem] = []
        warnings: list[str] = []
        blockers: list[str] = []
        recommendations: list[str] = []

        # 1. Environment & Platform Diagnostics
        plat_str = f"{platform.system()} {platform.release()} ({platform.machine()})"
        py_ver = sys.version.split()[0]

        # 2. Host Storage
        output_path = Path(self.config.output_dir)
        try:
            target_dir = output_path if output_path.exists() else output_path.parent
            if not target_dir.exists():
                target_dir = Path(".")
            total_disk, _, free_disk = shutil.disk_usage(str(target_dir))
            free_disk_gb = free_disk / (1024**3)
        except Exception:
            free_disk_gb = 0.0

        if free_disk_gb < 3.0:
            blockers.append(
                f"Free disk space is critically low ({free_disk_gb:.2f} GB). Minimum 3.0 GB required."
            )
            checks.append(
                PreflightCheckItem(
                    category="hardware",
                    name="Disk Space",
                    passed=False,
                    level="error",
                    message=f"Free disk space is {free_disk_gb:.2f} GB (minimum 3.0 GB required).",
                    recommendation="Free disk space before starting training.",
                )
            )
        elif free_disk_gb < 10.0:
            warnings.append(
                f"Free disk space is modest ({free_disk_gb:.2f} GB). Checkpoints and cache may consume significant space."
            )
            checks.append(
                PreflightCheckItem(
                    category="hardware",
                    name="Disk Space",
                    passed=True,
                    level="warning",
                    message=f"Free disk space: {free_disk_gb:.2f} GB (recommended >= 10.0 GB).",
                    recommendation="Monitor checkpoint disk usage during training.",
                )
            )
        else:
            checks.append(
                PreflightCheckItem(
                    category="hardware",
                    name="Disk Space",
                    passed=True,
                    level="info",
                    message=f"Free disk space: {free_disk_gb:.2f} GB (sufficient).",
                )
            )

        # 3. Host RAM
        total_ram_gb, free_ram_gb = self._get_host_ram()
        if free_ram_gb < 2.0 and total_ram_gb > 0:
            warnings.append(
                f"Available system RAM is low ({free_ram_gb:.2f} GB free of {total_ram_gb:.2f} GB total)."
            )
            checks.append(
                PreflightCheckItem(
                    category="hardware",
                    name="System RAM",
                    passed=True,
                    level="warning",
                    message=f"System RAM: {free_ram_gb:.2f} GB free of {total_ram_gb:.2f} GB total.",
                    recommendation="Close background applications to free memory for data loading.",
                )
            )
        else:
            checks.append(
                PreflightCheckItem(
                    category="hardware",
                    name="System RAM",
                    passed=True,
                    level="info",
                    message=f"System RAM: {free_ram_gb:.2f} GB free / {total_ram_gb:.2f} GB total.",
                )
            )

        # 4. Dependency Checks
        installed_packages: dict[str, str] = {}
        missing_packages: list[str] = []

        for pkg in self.REQUIRED_PACKAGES:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "installed")
                installed_packages[pkg] = ver
                checks.append(
                    PreflightCheckItem(
                        category="dependency",
                        name=f"Package: {pkg}",
                        passed=True,
                        level="info",
                        message=f"Package '{pkg}' is installed ({ver}).",
                    )
                )
            except ImportError:
                missing_packages.append(pkg)
                checks.append(
                    PreflightCheckItem(
                        category="dependency",
                        name=f"Package: {pkg}",
                        passed=False,
                        level="error",
                        message=f"Required package '{pkg}' is missing.",
                        recommendation='Install training dependencies via: pip install -e ".[train]".',
                    )
                )

        if missing_packages:
            blockers.append(
                f"Missing required training packages: {', '.join(missing_packages)}. "
                'Install via: pip install -e ".[train]".'
            )

        # Check QLoRA quantization packages if QLoRA is selected
        if self.config.method == FineTuningMethod.QLORA:
            try:
                bnb = __import__("bitsandbytes")
                ver = getattr(bnb, "__version__", "installed")
                checks.append(
                    PreflightCheckItem(
                        category="dependency",
                        name="Package: bitsandbytes",
                        passed=True,
                        level="info",
                        message=f"bitsandbytes is installed ({ver}) for 4-bit QLoRA.",
                    )
                )
            except ImportError:
                blockers.append(
                    "QLoRA requires 'bitsandbytes' for 4-bit NF4 quantization. "
                    "bitsandbytes is not installed. Install via: pip install bitsandbytes."
                )
                checks.append(
                    PreflightCheckItem(
                        category="dependency",
                        name="Package: bitsandbytes",
                        passed=False,
                        level="error",
                        message="Package 'bitsandbytes' is required for QLoRA but not installed.",
                        recommendation="Install bitsandbytes or use LoRA method with full precision weights.",
                    )
                )

        # 5. CUDA & GPU Hardware Feasibility
        cuda_avail = False
        device_count = 0
        device_names: list[str] = []
        total_vram_gb = 0.0
        free_vram_gb = 0.0

        if "torch" in installed_packages:
            import torch

            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    device_names.append(props.name)
                    vram_gb = props.total_memory / (1024**3)
                    total_vram_gb += vram_gb
                    # Free memory if currently initialized
                    try:
                        free_b, _ = torch.cuda.mem_get_info(i)
                        free_vram_gb += free_b / (1024**3)
                    except Exception:
                        free_vram_gb += vram_gb

        if not cuda_avail:
            blockers.append(
                "No CUDA GPU detected. Supervised fine-tuning requires discrete NVIDIA GPU acceleration. "
                "Local environment is CPU-only."
            )
            recommendations.append(
                "Run with --dry-run to validate pipeline orchestration and schemas without a GPU."
            )
            recommendations.append(
                "For actual training, execute on a cloud GPU instance with >= 16 GB VRAM (e.g. NVIDIA A10G or L4)."
            )
            checks.append(
                PreflightCheckItem(
                    category="hardware",
                    name="CUDA Acceleration",
                    passed=False,
                    level="error",
                    message="CUDA GPU is not available on this host.",
                    recommendation="Deploy to an environment with NVIDIA GPU acceleration for real training.",
                )
            )
        else:
            checks.append(
                PreflightCheckItem(
                    category="hardware",
                    name="CUDA Acceleration",
                    passed=True,
                    level="info",
                    message=f"CUDA GPU available: {device_count} device(s) ({', '.join(device_names)}). Total VRAM: {total_vram_gb:.1f} GB.",
                )
            )

            # VRAM threshold check based on model size and method
            model_key = "default"
            for size_key in ("1.5b", "7b", "8b"):
                if size_key in self.config.base_model_id.lower():
                    model_key = size_key
                    break

            method_str = self.config.method.value
            req_vram = MINIMUM_VRAM_REQUIREMENTS[model_key][method_str]
            effective_free_vram = free_vram_gb if free_vram_gb > 0 else total_vram_gb

            if effective_free_vram < req_vram:
                warn_msg = (
                    f"Available VRAM ({effective_free_vram:.1f} GB) is below the recommended threshold "
                    f"({req_vram:.1f} GB) for {model_key.upper()} with {method_str.upper()}."
                )
                warnings.append(warn_msg)
                checks.append(
                    PreflightCheckItem(
                        category="hardware",
                        name="VRAM Feasibility",
                        passed=False,
                        level="warning",
                        message=warn_msg,
                        recommendation="Consider reducing per_device_batch_size, using gradient checkpointing, or switching to QLoRA.",
                    )
                )
            else:
                checks.append(
                    PreflightCheckItem(
                        category="hardware",
                        name="VRAM Feasibility",
                        passed=True,
                        level="info",
                        message=f"Available VRAM ({effective_free_vram:.1f} GB) satisfies threshold ({req_vram:.1f} GB).",
                    )
                )

        # 6. Windows + bitsandbytes Specific Check
        if sys.platform == "win32" and self.config.method == FineTuningMethod.QLORA:
            warnings.append(
                "Running QLoRA on native Windows may encounter bitsandbytes 4-bit CUDA kernel compatibility issues. "
                "WSL2 (Windows Subsystem for Linux) or Ubuntu 22.04+ is strongly recommended for QLoRA."
            )
            recommendations.append(
                "Use WSL2 with Ubuntu and nvidia-container-toolkit for reliable QLoRA 4-bit kernel execution."
            )

        # 7. Dataset Availability Check
        if check_dataset:
            train_ds_name = self.config.train_dataset.split(":")[0]
            # Check known paths
            possible_paths = [
                Path("data/processed") / f"{train_ds_name}_train.json",
                Path("data/raw") / train_ds_name,
                Path("data/processed") / f"{train_ds_name}_dataset.json",
            ]
            ds_found = any(p.exists() for p in possible_paths)
            if not ds_found:
                warnings.append(
                    f"Dataset '{self.config.train_dataset}' was not found in data/processed or data/raw. "
                    "Ingest dataset via 'sqlforge data ingest' before real training."
                )
                checks.append(
                    PreflightCheckItem(
                        category="dataset",
                        name="Dataset Availability",
                        passed=False,
                        level="warning",
                        message=f"Training dataset files for '{self.config.train_dataset}' not detected locally.",
                        recommendation="Run 'sqlforge data ingest --dataset spider' to prepare dataset.",
                    )
                )
            else:
                checks.append(
                    PreflightCheckItem(
                        category="dataset",
                        name="Dataset Availability",
                        passed=True,
                        level="info",
                        message=f"Training dataset files for '{self.config.train_dataset}' detected.",
                    )
                )

        # Determine consolidated status
        if blockers:
            status = PreflightStatus.BLOCKED
            is_runnable = False
        elif warnings:
            status = PreflightStatus.WARNING
            is_runnable = cuda_avail  # Warning state is runnable if CUDA is present
        else:
            status = PreflightStatus.READY
            is_runnable = True

        return PreflightReport(
            status=status,
            is_runnable=is_runnable,
            platform=plat_str,
            python_version=py_ver,
            cuda_available=cuda_avail,
            device_count=device_count,
            device_names=device_names,
            total_vram_gb=round(total_vram_gb, 2),
            free_vram_gb=round(free_vram_gb, 2),
            total_ram_gb=round(total_ram_gb, 2),
            free_ram_gb=round(free_ram_gb, 2),
            free_disk_gb=round(free_disk_gb, 2),
            checks=checks,
            warnings=warnings,
            blockers=blockers,
            recommendations=recommendations,
        )

    @staticmethod
    def _get_host_ram() -> tuple[float, float]:
        """Obtain total and free host RAM in gigabytes across OS platforms."""
        total_ram_gb = 0.0
        free_ram_gb = 0.0

        if sys.platform == "win32":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
                total_ram_gb = stat.ullTotalPhys / (1024**3)
                free_ram_gb = stat.ullAvailPhys / (1024**3)
            except Exception:
                pass
        else:
            try:
                with open("/proc/meminfo") as f:
                    meminfo = f.read()
                for line in meminfo.splitlines():
                    if line.startswith("MemTotal:"):
                        total_ram_gb = int(line.split()[1]) / (1024**2)
                    elif line.startswith("MemAvailable:"):
                        free_ram_gb = int(line.split()[1]) / (1024**2)
            except Exception:
                pass

        return total_ram_gb, free_ram_gb
