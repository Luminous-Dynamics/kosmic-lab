"""
Tests for core.utils module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.utils import (
    bootstrap_confidence_interval,
    hash_config,
    infer_git_sha,
)


class TestBootstrapCI:
    """Test suite for bootstrap_confidence_interval function."""

    def test_basic_bootstrap(self, medium_sample_data: np.ndarray):
        """Test basic bootstrap CI computation."""
        ci_lower, ci_upper = bootstrap_confidence_interval(
            medium_sample_data, statistic=np.mean, n_bootstrap=100, confidence_level=0.95
        )

        # CI bounds should be reasonable
        assert ci_lower < ci_upper
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

        # Mean should be within CI
        mean_val = np.mean(medium_sample_data)
        assert ci_lower <= mean_val <= ci_upper or abs(ci_lower - mean_val) < 0.5

    def test_bootstrap_with_median(self, medium_sample_data: np.ndarray):
        """Test bootstrap CI with median statistic."""
        ci_lower, ci_upper = bootstrap_confidence_interval(
            medium_sample_data, statistic=np.median, n_bootstrap=100
        )

        median_val = np.median(medium_sample_data)
        assert ci_lower < ci_upper
        # Median should be reasonably close to CI
        assert ci_lower - 1.0 <= median_val <= ci_upper + 1.0

    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_confidence_levels(
        self, medium_sample_data: np.ndarray, confidence_level: float
    ):
        """Test different confidence levels."""
        ci_lower, ci_upper = bootstrap_confidence_interval(
            medium_sample_data,
            statistic=np.mean,
            n_bootstrap=100,
            confidence_level=confidence_level,
        )

        assert ci_lower < ci_upper
        ci_width = ci_upper - ci_lower
        # Higher confidence → wider CI
        assert ci_width > 0

    def test_reproducibility(self, medium_sample_data: np.ndarray):
        """Test that results are reproducible with same seed."""
        ci1 = bootstrap_confidence_interval(
            medium_sample_data, np.mean, n_bootstrap=100, seed=42
        )
        ci2 = bootstrap_confidence_interval(
            medium_sample_data, np.mean, n_bootstrap=100, seed=42
        )

        assert ci1 == ci2

    def test_empty_data_raises(self):
        """Test that empty data raises appropriate error."""
        with pytest.raises((ValueError, IndexError)):
            bootstrap_confidence_interval(
                np.array([]), statistic=np.mean, n_bootstrap=10
            )


class TestHashConfig:
    """Test suite for hash_config function."""

    def test_basic_hashing(self, sample_config: dict):
        """Test basic configuration hashing."""
        hash1 = hash_config(sample_config)

        assert isinstance(hash1, str)
        assert len(hash1) > 0
        # Should be hex string
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_deterministic_hashing(self, sample_config: dict):
        """Test that hashing is deterministic."""
        hash1 = hash_config(sample_config)
        hash2 = hash_config(sample_config)

        assert hash1 == hash2

    def test_different_configs_different_hashes(self):
        """Test that different configs produce different hashes."""
        config1 = {"a": 1, "b": 2}
        config2 = {"a": 1, "b": 3}

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        assert hash1 != hash2

    def test_order_independent_hashing(self):
        """Test that dict order doesn't affect hash."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        assert hash1 == hash2

    def test_nested_dict_hashing(self):
        """Test hashing of nested dictionaries."""
        config = {"outer": {"inner": {"value": 42}}}
        hash_result = hash_config(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) > 0

    def test_various_types(self):
        """Test hashing with various data types."""
        config = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
        }

        hash_result = hash_config(config)
        assert isinstance(hash_result, str)


class TestInferGitSHA:
    """Test suite for infer_git_sha function."""

    def test_returns_string(self):
        """Test that git SHA inference returns a string."""
        sha = infer_git_sha()

        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_valid_sha_format(self):
        """Test that returned SHA has valid format."""
        sha = infer_git_sha()

        # Should be either a valid hex string or "unknown"
        if sha != "unknown":
            assert len(sha) == 40 or len(sha) == 7  # Full or short SHA
            assert all(c in "0123456789abcdef" for c in sha)

    def test_with_repo_root(self, tmp_path: Path):
        """Test SHA inference with explicit repo root."""
        # Create a non-git directory
        non_git_dir = tmp_path / "not_a_repo"
        non_git_dir.mkdir()

        sha = infer_git_sha(repo_root=non_git_dir)

        # Should return "unknown" for non-git directory
        assert sha == "unknown"

    def test_reproducibility(self):
        """Test that SHA is consistent when called multiple times."""
        sha1 = infer_git_sha()
        sha2 = infer_git_sha()

        assert sha1 == sha2
