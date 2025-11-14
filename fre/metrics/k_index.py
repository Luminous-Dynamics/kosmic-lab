"""
K-Index: Observation-Action Coupling Metric

K = 2 * |ρ(||O||, ||A||)|

Where:
- ||O|| = observation norms over recent window
- ||A|| = action norms over recent window
- ρ = Pearson correlation coefficient

Bounds: K ∈ [0, 2]
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Any, Dict, Tuple


def k_index(obs_norms: np.ndarray, act_norms: np.ndarray) -> float:
    """
    Compute K-Index as twice the absolute Pearson correlation.

    Args:
        obs_norms: Observation norms [T]
        act_norms: Action norms [T]

    Returns:
        K-Index value in [0, 2]

    Raises:
        AssertionError: If K-Index is outside [0, 2] (excluding NaN)

    Example:
        >>> obs = np.array([1.0, 2.0, 3.0])
        >>> act = np.array([1.1, 2.2, 3.1])
        >>> k = k_index(obs, act)
        >>> assert 0.0 <= k <= 2.0
    """
    if len(obs_norms) < 2 or len(act_norms) < 2:
        return np.nan

    r, _ = pearsonr(obs_norms, act_norms)
    k = 2.0 * abs(r)

    # Bounds check (allow NaN for insufficient data)
    if not (0.0 <= k <= 2.0 or np.isnan(k)):
        raise AssertionError(
            f"K-Index out of bounds: {k:.4f}. "
            f"Expected K ∈ [0, 2]. "
            f"obs_norms: min={obs_norms.min():.4f}, max={obs_norms.max():.4f}, "
            f"act_norms: min={act_norms.min():.4f}, max={act_norms.max():.4f}"
        )

    return k


def k_index_robust(obs_norms: np.ndarray, act_norms: np.ndarray) -> Dict[str, float]:
    """
    Compute K-Index with robust variants.

    Variants:
    1. k_pearson: Standard K-Index (Pearson)
    2. k_pearson_z: K-Index with z-scored inputs (magnitude-normalized)
    3. k_spearman: K-Index with Spearman (rank-based, distribution-free)

    Args:
        obs_norms: Observation norms [T]
        act_norms: Action norms [T]

    Returns:
        Dictionary with all K-Index variants

    Example:
        >>> obs = np.random.randn(100)
        >>> act = obs + np.random.randn(100) * 0.1
        >>> k_variants = k_index_robust(obs, act)
        >>> assert 'k_pearson' in k_variants
        >>> assert 'k_spearman' in k_variants
    """
    if len(obs_norms) < 2 or len(act_norms) < 2:
        return {"k_pearson": np.nan, "k_pearson_z": np.nan, "k_spearman": np.nan}

    # 1. Standard Pearson K-Index
    k_p = k_index(obs_norms, act_norms)

    # 2. Z-scored K-Index (magnitude-normalized)
    obs_z = (obs_norms - obs_norms.mean()) / (obs_norms.std() + 1e-8)
    act_z = (act_norms - act_norms.mean()) / (act_norms.std() + 1e-8)
    r_z, _ = pearsonr(obs_z, act_z)
    k_pz = 2.0 * abs(r_z)

    # 3. Spearman K-Index (rank-based)
    rho, _ = spearmanr(obs_norms, act_norms)
    k_s = 2.0 * abs(rho)

    return {"k_pearson": k_p, "k_pearson_z": k_pz, "k_spearman": k_s}


def k_index_with_ci(
    obs_norms: np.ndarray,
    act_norms: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    Compute K-Index with bootstrap confidence interval.

    Args:
        obs_norms: Observation norms [T]
        act_norms: Action norms [T]
        n_bootstrap: Number of bootstrap samples
        alpha: Confidence level (default: 0.05 for 95% CI)
        rng: Random number generator

    Returns:
        (k_estimate, k_lower, k_upper) tuple

    Example:
        >>> obs = np.random.randn(100)
        >>> act = obs + np.random.randn(100) * 0.1
        >>> k, k_low, k_high = k_index_with_ci(obs, act)
        >>> assert k_low <= k <= k_high
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(obs_norms)
    k_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, T, size=T)
        obs_boot = obs_norms[idx]
        act_boot = act_norms[idx]
        k_boot = k_index(obs_boot, act_boot)
        if not np.isnan(k_boot):
            k_samples.append(k_boot)

    if len(k_samples) == 0:
        return np.nan, np.nan, np.nan

    k_samples = np.array(k_samples)
    k_estimate = k_index(obs_norms, act_norms)
    k_lower = np.percentile(k_samples, 100 * alpha / 2)
    k_upper = np.percentile(k_samples, 100 * (1 - alpha / 2))

    return k_estimate, k_lower, k_upper


def verify_k_bounds(k_values: np.ndarray) -> Dict[str, Any]:
    """
    Verify all K-Index values are within [0, 2] bounds.

    Args:
        k_values: Array of K-Index values

    Returns:
        Dictionary with verification results

    Example:
        >>> k_vals = np.array([0.5, 1.0, 1.5, 0.8])
        >>> result = verify_k_bounds(k_vals)
        >>> assert result['all_valid']
    """
    valid = k_values[~np.isnan(k_values)]

    in_bounds = np.logical_and(valid >= 0.0, valid <= 2.0)
    violations = valid[~in_bounds]

    return {
        "total": len(k_values),
        "valid": len(valid),
        "nan": np.sum(np.isnan(k_values)),
        "all_valid": len(violations) == 0,
        "violations": violations.tolist() if len(violations) > 0 else [],
        "min": valid.min() if len(valid) > 0 else np.nan,
        "max": valid.max() if len(valid) > 0 else np.nan,
        "mean": valid.mean() if len(valid) > 0 else np.nan,
    }
