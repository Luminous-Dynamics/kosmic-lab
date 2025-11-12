"""
Unit tests for K-Index implementation.

These tests verify:
1. K-Index is always in [0, 2]
2. Perfect correlation → K=2
3. Zero correlation → K≈0
4. Monotonicity with correlation strength
5. Robust variants produce similar results
"""
import numpy as np
import pytest
from fre.metrics.k_index import (
    k_index, k_index_robust, k_index_with_ci, verify_k_bounds
)


def test_k_bounds_perfect_correlation():
    """Test K-Index = 2 for perfect positive correlation."""
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    act = obs.copy()  # Perfect correlation

    k = k_index(obs, act)

    assert abs(k - 2.0) < 1e-6, f"Perfect correlation should give K=2, got K={k:.6f}"


def test_k_bounds_perfect_negative_correlation():
    """Test K-Index = 2 for perfect negative correlation (absolute value)."""
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    act = -obs  # Perfect negative correlation

    k = k_index(obs, act)

    assert abs(k - 2.0) < 1e-6, f"Perfect negative correlation should give K=2, got K={k:.6f}"


def test_k_bounds_zero_correlation():
    """Test K-Index ≈ 0 for zero correlation."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(1000)
    act = rng.standard_normal(1000)  # Independent

    k = k_index(obs, act)

    # With 1000 samples, should be very close to 0
    assert k < 0.2, f"Independent variables should give K≈0, got K={k:.3f}"


def test_k_monotonicity():
    """Test K-Index increases with correlation strength."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(1000)

    # Different correlation strengths
    k_values = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # act = alpha * obs + (1-alpha) * noise
        noise = rng.standard_normal(1000)
        act = alpha * obs + (1 - alpha) * noise
        k = k_index(obs, act)
        k_values.append(k)

    # Check monotonicity
    for i in range(len(k_values) - 1):
        assert k_values[i] <= k_values[i+1] + 0.1, (
            f"K-Index should increase with correlation: "
            f"K[{i}]={k_values[i]:.3f} > K[{i+1}]={k_values[i+1]:.3f}"
        )


def test_k_bounds_enforcement():
    """Test that K-Index always satisfies 0 ≤ K ≤ 2."""
    rng = np.random.default_rng(42)

    # Test 100 random configurations
    for _ in range(100):
        obs = rng.standard_normal(100)
        act = rng.standard_normal(100)
        k = k_index(obs, act)

        if not np.isnan(k):
            assert 0.0 <= k <= 2.0, f"K-Index out of bounds: K={k:.6f}"


def test_k_robust_variants_consistency():
    """Test that robust K-Index variants give similar results."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(1000)
    act = 0.7 * obs + 0.3 * rng.standard_normal(1000)

    k_variants = k_index_robust(obs, act)

    k_p = k_variants['k_pearson']
    k_pz = k_variants['k_pearson_z']
    k_s = k_variants['k_spearman']

    # All should be in bounds
    assert 0.0 <= k_p <= 2.0, f"k_pearson out of bounds: {k_p}"
    assert 0.0 <= k_pz <= 2.0, f"k_pearson_z out of bounds: {k_pz}"
    assert 0.0 <= k_s <= 2.0, f"k_spearman out of bounds: {k_s}"

    # Pearson and z-scored Pearson should be very similar
    assert abs(k_p - k_pz) < 0.1, (
        f"Pearson and z-scored Pearson should be similar: {k_p:.3f} vs {k_pz:.3f}"
    )

    # Spearman should be reasonably close (rank-based)
    assert abs(k_p - k_s) < 0.3, (
        f"Pearson and Spearman should be reasonably close: {k_p:.3f} vs {k_s:.3f}"
    )


def test_k_confidence_interval():
    """Test K-Index bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(100)
    act = 0.7 * obs + 0.3 * rng.standard_normal(100)

    k_est, k_low, k_high = k_index_with_ci(obs, act, n_bootstrap=100, rng=rng)

    # Check ordering
    assert k_low <= k_est <= k_high, (
        f"CI should contain estimate: [{k_low:.3f}, {k_high:.3f}] vs {k_est:.3f}"
    )

    # Check all in bounds
    assert 0.0 <= k_low <= 2.0, f"k_lower out of bounds: {k_low}"
    assert 0.0 <= k_est <= 2.0, f"k_estimate out of bounds: {k_est}"
    assert 0.0 <= k_high <= 2.0, f"k_upper out of bounds: {k_high}"


def test_k_insufficient_data():
    """Test K-Index returns NaN for insufficient data."""
    obs = np.array([1.0])
    act = np.array([1.0])

    k = k_index(obs, act)

    assert np.isnan(k), "K-Index should return NaN for insufficient data"


def test_k_constant_inputs():
    """Test K-Index handles constant inputs (zero variance)."""
    obs = np.ones(10)  # Constant
    act = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # This will raise a warning from pearsonr, but should not crash
    k = k_index(obs, act)

    # Should return NaN for zero variance
    assert np.isnan(k), "K-Index should return NaN for constant input"


def test_verify_k_bounds_all_valid():
    """Test verify_k_bounds with all valid K-Index values."""
    k_values = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.3])

    result = verify_k_bounds(k_values)

    assert result['all_valid'], "All K-Index values should be valid"
    assert result['total'] == 6
    assert result['valid'] == 6
    assert result['nan'] == 0
    assert len(result['violations']) == 0
    assert 0.0 <= result['min'] <= 2.0
    assert 0.0 <= result['max'] <= 2.0


def test_verify_k_bounds_with_violations():
    """Test verify_k_bounds detects out-of-bounds values."""
    # Manually create invalid values (this should not happen in practice)
    k_values = np.array([0.5, 1.0, 2.5, -0.1, 1.5])  # 2.5 and -0.1 are invalid

    result = verify_k_bounds(k_values)

    assert not result['all_valid'], "Should detect violations"
    assert len(result['violations']) == 2, "Should detect 2 violations"
    assert 2.5 in result['violations']
    assert -0.1 in result['violations']


def test_verify_k_bounds_with_nan():
    """Test verify_k_bounds handles NaN values correctly."""
    k_values = np.array([0.5, np.nan, 1.0, np.nan, 1.5])

    result = verify_k_bounds(k_values)

    assert result['total'] == 5
    assert result['valid'] == 3
    assert result['nan'] == 2
    assert result['all_valid'], "NaN values should not count as violations"


def test_k_scale_invariance():
    """Test K-Index is invariant to scale (correlation property)."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(100)
    act = 0.7 * obs + 0.3 * rng.standard_normal(100)

    k1 = k_index(obs, act)
    k2 = k_index(10 * obs, act)  # Scale obs by 10
    k3 = k_index(obs, 10 * act)  # Scale act by 10
    k4 = k_index(10 * obs, 10 * act)  # Scale both

    # All should be the same (correlation is scale-invariant)
    assert abs(k1 - k2) < 1e-6, "K-Index should be invariant to obs scaling"
    assert abs(k1 - k3) < 1e-6, "K-Index should be invariant to act scaling"
    assert abs(k1 - k4) < 1e-6, "K-Index should be invariant to both scaling"


def test_k_translation_invariance():
    """Test K-Index is invariant to translation (correlation property)."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(100)
    act = 0.7 * obs + 0.3 * rng.standard_normal(100)

    k1 = k_index(obs, act)
    k2 = k_index(obs + 100, act)  # Shift obs
    k3 = k_index(obs, act + 100)  # Shift act
    k4 = k_index(obs + 100, act + 100)  # Shift both

    # All should be the same (correlation is translation-invariant)
    assert abs(k1 - k2) < 1e-6, "K-Index should be invariant to obs translation"
    assert abs(k1 - k3) < 1e-6, "K-Index should be invariant to act translation"
    assert abs(k1 - k4) < 1e-6, "K-Index should be invariant to both translation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
