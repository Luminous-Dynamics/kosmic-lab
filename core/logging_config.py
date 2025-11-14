"""
Centralized logging configuration for Kosmic Lab.

This module provides a unified logging setup with sensible defaults,
structured logging support, and easy configuration.

Usage:
    from core.logging_config import setup_logging, get_logger

    # In main script/entry point
    setup_logging(level="INFO", log_file="logs/experiment.log")

    # In any module
    logger = get_logger(__name__)
    logger.info("Starting experiment", extra={"experiment_id": "exp_001"})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ANSI color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels in terminal output."""

    def format(self, record: logging.LogRecord) -> str:
        # Add color to levelname
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname
        return result


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path | str] = None,
    log_format: Optional[str] = None,
    colored: bool = True,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If provided, logs to both file and console
        log_format: Custom format string. If None, uses sensible default
        colored: Whether to use colored output in console (default: True)

    Example:
        >>> setup_logging(level="DEBUG", log_file="logs/experiment.log")
        >>> logger = get_logger(__name__)
        >>> logger.info("Experiment started")
    """
    # Default format includes timestamp, level, module, and message
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if colored and sys.stdout.isatty():
        console_formatter = ColoredFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        console_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)

        # File logs shouldn't have colors
        file_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Detailed diagnostic information")
        >>> logger.info("Normal operation message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
        >>> logger.critical("Critical failure message")
    """
    return logging.getLogger(name)


def log_experiment_start(
    logger: logging.Logger,
    experiment: str,
    params: dict,
    seed: int,
    run_id: str,
) -> None:
    """
    Log structured information about experiment start.

    Args:
        logger: Logger instance
        experiment: Experiment name/type
        params: Parameter dictionary
        seed: Random seed
        run_id: Unique run identifier

    Example:
        >>> logger = get_logger(__name__)
        >>> log_experiment_start(
        ...     logger,
        ...     experiment="track_b_sac",
        ...     params={"learning_rate": 0.001},
        ...     seed=42,
        ...     run_id="exp_001"
        ... )
    """
    logger.info(
        f"Starting experiment: {experiment}",
        extra={
            "experiment": experiment,
            "run_id": run_id,
            "seed": seed,
            "params": params,
        },
    )


def log_experiment_end(
    logger: logging.Logger,
    run_id: str,
    metrics: dict,
    success: bool = True,
) -> None:
    """
    Log structured information about experiment completion.

    Args:
        logger: Logger instance
        run_id: Unique run identifier
        metrics: Final metrics dictionary
        success: Whether experiment completed successfully

    Example:
        >>> logger = get_logger(__name__)
        >>> log_experiment_end(
        ...     logger,
        ...     run_id="exp_001",
        ...     metrics={"K": 1.23, "TAT": 0.45},
        ...     success=True
        ... )
    """
    level = logging.INFO if success else logging.ERROR
    status = "completed" if success else "failed"

    logger.log(
        level,
        f"Experiment {status}: {run_id}",
        extra={"run_id": run_id, "metrics": metrics, "success": success},
    )


# Initialize with basic configuration by default
# Applications can call setup_logging() again to reconfigure
setup_logging(level="INFO", colored=True)
