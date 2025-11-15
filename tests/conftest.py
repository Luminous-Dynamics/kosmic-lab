"""
Pytest configuration and shared fixtures.

This file contains shared fixtures and configuration for all tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


@pytest.fixture
def random_seed() -> int:
    """Provide a consistent random seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """Provide a seeded numpy random generator."""
    return np.random.default_rng(random_seed)


@pytest.fixture
def small_sample_data(rng: np.random.Generator) -> np.ndarray:
    """Generate small sample data for quick tests (N=10)."""
    return np.abs(rng.standard_normal(10))


@pytest.fixture
def medium_sample_data(rng: np.random.Generator) -> np.ndarray:
    """Generate medium sample data (N=100)."""
    return np.abs(rng.standard_normal(100))


@pytest.fixture
def large_sample_data(rng: np.random.Generator) -> np.ndarray:
    """Generate large sample data for performance tests (N=1000)."""
    return np.abs(rng.standard_normal(1000))


@pytest.fixture
def correlated_data_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pair of correlated time series.

    Returns:
        Tuple of (observed, actual) arrays with high correlation
    """
    n = 100
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  # High correlation
    data = rng.multivariate_normal(mean, cov, n)
    observed = np.abs(data[:, 0])
    actual = np.abs(data[:, 1])
    return observed, actual


@pytest.fixture
def uncorrelated_data_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pair of uncorrelated time series.

    Returns:
        Tuple of (observed, actual) arrays with low correlation
    """
    observed = np.abs(rng.standard_normal(100))
    actual = np.abs(rng.standard_normal(100))
    return observed, actual


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for experiments."""
    return {
        "n_samples": 100,
        "threshold": 0.5,
        "seed": 42,
        "verbose": False,
    }


@pytest.fixture
def sample_params() -> Dict[str, float]:
    """Provide sample simulation parameters."""
    return {
        "consciousness": 0.7,
        "coherence": 0.8,
        "fep": 0.5,
    }


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """
    Provide temporary directory for test logs.

    Args:
        tmp_path: pytest built-in fixture for temporary directory

    Returns:
        Path to logs directory
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """
    Provide temporary directory for test data.

    Args:
        tmp_path: pytest built-in fixture for temporary directory

    Returns:
        Path to data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: marks tests that require GPU (deselect with '-m \"not requires_gpu\"')",
    )
    config.addinivalue_line(
        "markers",
        "network: marks tests that require network access (deselect with '-m \"not network\"')",
    )
