"""System environment diagnostic utilities."""

from typing import Any

from sqlforge.reproducibility import get_environment_metadata


def collect_system_diagnostics() -> dict[str, Any]:
    """Collect comprehensive hardware, OS, Python, and Git diagnostics."""
    env = get_environment_metadata()

    # Check key ML packages without crashing if absent
    packages = [
        "pydantic",
        "yaml",
        "click",
        "rich",
        "pytest",
        "ruff",
        "mypy",
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "sqlglot",
        "wandb",
    ]

    import importlib.metadata

    installed_versions: dict[str, str] = {}
    for pkg in packages:
        try:
            installed_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            try:
                mod = __import__(pkg)
                installed_versions[pkg] = getattr(mod, "__version__", "installed")
            except (ImportError, Exception):
                installed_versions[pkg] = "not installed"
        except Exception:
            installed_versions[pkg] = "not installed"

    env["packages"] = installed_versions
    return env
