"""
Shared utility functions for the Kosmic Lab platform.

This module contains common functionality used across multiple modules
to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def infer_git_sha(repo_root: Optional[Path] = None) -> str:
    """
    Infer the current git commit SHA for reproducibility tracking.

    Args:
        repo_root: Optional path to repository root. If None, uses current directory.

    Returns:
        Git commit SHA string, or "unknown" if git is not available or repo not found.

    Example:
        >>> sha = infer_git_sha()
        >>> len(sha) == 40 or sha == "unknown"
        True
    """
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Input data array
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        confidence: Confidence level (default: 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> data = np.random.randn(100)
        >>> estimate, lower, upper = bootstrap_confidence_interval(data, np.mean)
        >>> lower <= estimate <= upper
        True
    """
    rng = np.random.default_rng(random_seed)
    n = len(data)

    # Compute point estimate
    point_estimate = statistic_func(data)

    # Bootstrap resampling
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_estimates[i] = statistic_func(resample)

    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    return float(point_estimate), float(lower_bound), float(upper_bound)


def validate_bounds(
    value: float, lower: float, upper: float, name: str = "value"
) -> None:
    """
    Validate that a value falls within specified bounds.

    Args:
        value: Value to validate
        lower: Lower bound (inclusive)
        upper: Upper bound (inclusive)
        name: Name of the value for error messages

    Raises:
        ValueError: If value is outside bounds

    Example:
        >>> validate_bounds(1.5, 0.0, 2.0, "K-index")  # OK
        >>> validate_bounds(3.0, 0.0, 2.0, "K-index")  # Raises ValueError
        Traceback (most recent call last):
        ...
        ValueError: K-index = 3.0 is outside valid range [0.0, 2.0]
    """
    if not (lower <= value <= upper):
        raise ValueError(
            f"{name} = {value} is outside valid range [{lower}, {upper}]"
        )


def safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero (default: 0.0)

    Returns:
        Result of division or default value

    Example:
        >>> safe_divide(10.0, 2.0)
        5.0
        >>> safe_divide(10.0, 0.0, default=np.nan)
        nan
    """
    if abs(denominator) < 1e-10:  # Avoid division by very small numbers
        return default
    return numerator / denominator


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        The same path (for chaining)

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     test_dir = Path(tmpdir) / "subdir" / "nested"
        ...     result = ensure_directory(test_dir)
        ...     result.exists() and result.is_dir()
        True
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
