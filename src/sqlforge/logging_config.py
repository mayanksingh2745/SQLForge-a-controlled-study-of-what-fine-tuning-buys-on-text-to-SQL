"""Structured logging configuration for SQLForge."""

import logging
import sys
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    format_type: str = "rich",
    log_to_file: bool = False,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Configure structured logging for SQLForge.

    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        format_type: 'rich' for colorful formatted console output, or 'text' / 'json'.
        log_to_file: Whether to attach a file handler.
        log_file: Path to destination log file.

    Returns:
        Root logger for 'sqlforge'.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("sqlforge")
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    if format_type == "rich":
        try:
            from rich.logging import RichHandler

            console_handler: logging.Handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
            )
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_to_file and log_file is not None:
        target_path = Path(log_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(target_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
