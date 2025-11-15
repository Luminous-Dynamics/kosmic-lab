"""
End-to-end integration tests for Kosmic Lab.

These tests verify that the entire system works together correctly,
from data generation through analysis to K-Codex logging.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.kcodex import KCodexWriter
from core.logging_config import get_logger, setup_logging
from core.utils import bootstrap_confidence_interval, hash_config, infer_git_sha
from fre.metrics.k_index import k_index
from fre.metrics.k_lag import k_lag

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


class TestEndToEndWorkflow:
    """Test complete workflow from start to finish."""

    def test_full_experiment_workflow(self, temp_log_dir: Path, tmp_path: Path):
        """
        Test complete experiment workflow.

        This test simulates a full experiment:
        1. Setup logging
        2. Generate data
        3. Compute metrics
        4. Log to K-Codex
        5. Verify reproducibility
        """
        # Step 1: Setup logging
        log_file = temp_log_dir / "integration_test.log"
        setup_logging(level="INFO", log_file=str(log_file))
        logger = get_logger(__name__)

        logger.info("Starting integration test")

        # Step 2: Generate synthetic data
        rng = np.random.default_rng(42)
        n_samples = 100

        # Generate correlated data
        mean = [0, 0]
        cov = [[1, 0.7], [0.7, 1]]
        data = rng.multivariate_normal(mean, cov, n_samples)
        observed = np.abs(data[:, 0])
        actual = np.abs(data[:, 1])

        logger.info(f"Generated {n_samples} samples")

        # Step 3: Compute metrics
        k = k_index(observed, actual)
        logger.info(f"K-Index: {k:.4f}")

        # Compute bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(
            observed,
            lambda x: k_index(x, actual),
            n_bootstrap=100,
            confidence_level=0.95,
        )
        logger.info(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Compute K-Lag
        lag_results = k_lag(observed, actual, max_lag=10)
        logger.info(f"Best lag: {lag_results['best_lag']}")

        # Step 4: Log to K-Codex
        kcodex_path = temp_log_dir / "experiment_kcodex.json"
        kcodex = KCodexWriter(str(kcodex_path))

        params = {
            "n_samples": n_samples,
            "correlation": 0.7,
            "seed": 42,
        }

        metrics = {
            "k_index": float(k),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "best_lag": int(lag_results["best_lag"]),
            "k_at_best_lag": float(lag_results["k_at_best_lag"]),
        }

        kcodex.log_experiment(
            experiment_name="integration_test",
            params=params,
            metrics=metrics,
            seed=42,
            extra_metadata={"test": True},
        )

        logger.info("K-Codex logged successfully")

        # Step 5: Verify K-Codex was created correctly
        assert kcodex_path.exists()

        with open(kcodex_path) as f:
            kcodex_data = json.load(f)

        # Verify structure
        assert "experiment_name" in kcodex_data
        assert "timestamp" in kcodex_data
        assert "git_sha" in kcodex_data
        assert "config_hash" in kcodex_data
        assert "params" in kcodex_data
        assert "metrics" in kcodex_data
        assert "seed" in kcodex_data

        # Verify values
        assert kcodex_data["experiment_name"] == "integration_test"
        assert kcodex_data["params"] == params
        assert kcodex_data["metrics"] == metrics
        assert kcodex_data["seed"] == 42

        # Verify log file was created
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Starting integration test" in log_content
        assert "K-Index:" in log_content
        assert "K-Codex logged successfully" in log_content

        logger.info("Integration test completed successfully")

    def test_reproducibility_workflow(self, temp_log_dir: Path):
        """
        Test that experiments are reproducible.

        Runs the same experiment twice with the same seed and verifies
        identical results.
        """
        setup_logging(level="INFO")

        def run_experiment(seed: int) -> dict:
            """Run a simple experiment."""
            rng = np.random.default_rng(seed)
            data = rng.standard_normal(100)
            observed = np.abs(data)
            actual = np.abs(rng.standard_normal(100))

            k = k_index(observed, actual)

            return {"k_index": float(k), "mean": float(np.mean(observed))}

        # Run twice with same seed
        results1 = run_experiment(seed=42)
        results2 = run_experiment(seed=42)

        # Should be identical
        assert results1["k_index"] == results2["k_index"]
        assert results1["mean"] == results2["mean"]

        # Run with different seed - should differ
        results3 = run_experiment(seed=43)
        assert results3["k_index"] != results1["k_index"]

    def test_config_hashing_reproducibility(self):
        """Test that configuration hashing is reproducible."""
        config1 = {
            "model": "baseline",
            "params": {"learning_rate": 0.01, "epochs": 100},
            "seed": 42,
            "notes": "Test configuration",
        }

        # Hash multiple times
        hash1 = hash_config(config1)
        hash2 = hash_config(config1)
        hash3 = hash_config(config1)

        # All should be identical
        assert hash1 == hash2 == hash3

        # Different config should hash differently
        config2 = config1.copy()
        config2["params"]["learning_rate"] = 0.02

        hash4 = hash_config(config2)
        assert hash4 != hash1

        # Order shouldn't matter
        config3 = {
            "seed": 42,
            "notes": "Test configuration",
            "model": "baseline",
            "params": {"epochs": 100, "learning_rate": 0.01},
        }

        hash5 = hash_config(config3)
        assert hash5 == hash1  # Same content, different order

    def test_git_sha_inference(self):
        """Test git SHA inference works."""
        sha = infer_git_sha()

        # Should return a string
        assert isinstance(sha, str)

        # Should be either "unknown" or a valid git SHA
        if sha != "unknown":
            assert len(sha) >= 7  # At least short SHA
            assert all(c in "0123456789abcdef" for c in sha)

        # Should be consistent
        sha2 = infer_git_sha()
        assert sha == sha2


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_data_handling(self):
        """Test that invalid data is handled gracefully."""
        # Empty arrays
        with pytest.raises((ValueError, IndexError)):
            k_index(np.array([]), np.array([]))

        # Mismatched lengths
        with pytest.raises(ValueError):
            k_index(np.array([1, 2, 3]), np.array([1, 2]))

        # NaN values - should handle gracefully or raise clear error
        try:
            result = k_index(np.array([np.nan, 1, 2]), np.array([1, 2, 3]))
            # If it succeeds, result should be NaN or handled
            assert np.isnan(result) or isinstance(result, float)
        except ValueError as e:
            # Or it should raise a clear error
            assert "NaN" in str(e) or "invalid" in str(e).lower()

    def test_kcodex_file_errors(self, tmp_path: Path):
        """Test K-Codex handles file errors gracefully."""
        # Try to write to non-existent directory
        bad_path = tmp_path / "nonexistent" / "dir" / "kcodex.json"

        # Should either create directories or raise clear error
        try:
            kcodex = KCodexWriter(str(bad_path))
            kcodex.log_experiment(
                experiment_name="test",
                params={"a": 1},
                metrics={"b": 2},
                seed=42,
            )
            # If it succeeds, file should exist
            assert bad_path.exists()
        except (FileNotFoundError, OSError) as e:
            # Or it should raise a clear error
            assert "directory" in str(e).lower() or "path" in str(e).lower()


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance characteristics in integration scenarios."""

    def test_large_scale_workflow(self, temp_log_dir: Path):
        """Test workflow with larger datasets."""
        import time

        setup_logging(level="INFO")
        logger = get_logger(__name__)

        # Generate large dataset
        n_samples = 10_000
        rng = np.random.default_rng(42)
        observed = np.abs(rng.standard_normal(n_samples))
        actual = np.abs(rng.standard_normal(n_samples))

        # Time K-Index computation
        start = time.time()
        k = k_index(observed, actual)
        elapsed = time.time() - start

        logger.info(f"K-Index computation for N={n_samples}: {elapsed:.3f}s")

        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max

        # K-Index should be valid
        assert isinstance(k, float)
        assert not np.isnan(k)
        assert not np.isinf(k)

    def test_multiple_experiments_workflow(self, temp_log_dir: Path):
        """Test running multiple experiments in sequence."""
        setup_logging(level="INFO")

        n_experiments = 10
        results = []

        for i in range(n_experiments):
            rng = np.random.default_rng(42 + i)
            observed = np.abs(rng.standard_normal(100))
            actual = np.abs(rng.standard_normal(100))

            k = k_index(observed, actual)
            results.append(k)

            # Log to K-Codex
            kcodex = KCodexWriter(str(temp_log_dir / f"exp_{i}_kcodex.json"))
            kcodex.log_experiment(
                experiment_name=f"multi_exp_{i}",
                params={"experiment_id": i, "seed": 42 + i},
                metrics={"k_index": float(k)},
                seed=42 + i,
            )

        # Verify all K-Codex files were created
        kcodex_files = list(temp_log_dir.glob("exp_*_kcodex.json"))
        assert len(kcodex_files) == n_experiments

        # Verify results are reasonable
        assert len(results) == n_experiments
        assert all(isinstance(k, float) for k in results)
        assert all(not np.isnan(k) for k in results)
