"""
Partial Correlation Analysis for K-Index

Compute partial correlation controlling for reward:
ρ(||O||, ||A|| | R)

This proves K-Index measures intrinsic coherence, not just task optimization.
Expected: k_partial ≈ k_raw (reward doesn't explain the correlation)
"""
import numpy as np
from scipy.stats import pearsonr
from typing import Dict


def k_partial_reward(
    obs_norms: np.ndarray,
    act_norms: np.ndarray,
    rewards: np.ndarray
) -> Dict[str, float]:
    """
    Compute K-Index with partial correlation controlling for reward.

    Partial correlation formula:
    ρ(X, Y | Z) = (ρ(X,Y) - ρ(X,Z)ρ(Y,Z)) / √((1-ρ(X,Z)²)(1-ρ(Y,Z)²))

    Args:
        obs_norms: Observation norms [T]
        act_norms: Action norms [T]
        rewards: Reward values [T]

    Returns:
        Dictionary with:
        - k_raw: Standard K-Index (no control)
        - k_partial: K-Index controlling for reward
        - delta: k_raw - k_partial (should be small)
        - r_obs_rew: Correlation between obs_norms and rewards
        - r_act_rew: Correlation between act_norms and rewards

    Example:
        >>> obs = np.random.randn(100)
        >>> act = obs + np.random.randn(100) * 0.1
        >>> rew = np.random.randn(100)  # Random rewards
        >>> result = k_partial_reward(obs, act, rew)
        >>> assert abs(result['delta']) < 0.1, "K-Index should be reward-independent"
    """
    if len(obs_norms) < 3:
        return {
            "k_raw": np.nan,
            "k_partial": np.nan,
            "delta": np.nan,
            "r_obs_rew": np.nan,
            "r_act_rew": np.nan
        }

    # Raw correlations
    r_oa, _ = pearsonr(obs_norms, act_norms)  # ρ(O, A)
    r_or, _ = pearsonr(obs_norms, rewards)    # ρ(O, R)
    r_ar, _ = pearsonr(act_norms, rewards)    # ρ(A, R)

    # K-Index (raw, no control)
    k_raw = 2.0 * abs(r_oa)

    # Partial correlation: ρ(O, A | R)
    numerator = r_oa - r_or * r_ar
    denominator = np.sqrt((1 - r_or**2) * (1 - r_ar**2))

    if denominator < 1e-10:
        # Degenerate case: reward perfectly predicts obs or act
        r_partial = np.nan
        k_partial = np.nan
    else:
        r_partial = numerator / denominator
        k_partial = 2.0 * abs(r_partial)

    # Delta: how much does controlling for reward change K?
    delta = k_raw - k_partial if not np.isnan(k_partial) else np.nan

    return {
        "k_raw": k_raw,
        "k_partial": k_partial,
        "delta": delta,
        "r_obs_rew": r_or,
        "r_act_rew": r_ar,
        "r_obs_act": r_oa,
        "r_partial": r_partial
    }


def verify_reward_independence(
    partial_result: Dict[str, float],
    threshold: float = 0.1
) -> Dict[str, any]:
    """
    Verify K-Index is independent of reward (k_partial ≈ k_raw).

    Args:
        partial_result: Output from k_partial_reward()
        threshold: Maximum acceptable |delta| (default: 0.1)

    Returns:
        Dictionary with verification results

    Example:
        >>> result = k_partial_reward(obs, act, rew)
        >>> verify = verify_reward_independence(result)
        >>> assert verify['reward_independent']
    """
    k_raw = partial_result['k_raw']
    k_partial = partial_result['k_partial']
    delta = partial_result['delta']

    if np.isnan(delta):
        return {
            "reward_independent": None,
            "message": "Cannot compute partial correlation (insufficient data or degenerate case)"
        }

    independent = abs(delta) < threshold

    message = []
    if independent:
        message.append(f"✅ K-Index reward-independent: |Δ|={abs(delta):.3f} < {threshold}")
        message.append(f"   k_raw={k_raw:.3f}, k_partial={k_partial:.3f}")
    else:
        message.append(f"⚠️ K-Index may depend on reward: |Δ|={abs(delta):.3f} ≥ {threshold}")
        message.append(f"   k_raw={k_raw:.3f}, k_partial={k_partial:.3f}")

    # Additional context
    r_or = partial_result['r_obs_rew']
    r_ar = partial_result['r_act_rew']
    if abs(r_or) > 0.5 or abs(r_ar) > 0.5:
        message.append(f"⚠️ High reward correlation detected: "
                      f"ρ(O,R)={r_or:.3f}, ρ(A,R)={r_ar:.3f}")

    return {
        "reward_independent": independent,
        "delta": delta,
        "k_raw": k_raw,
        "k_partial": k_partial,
        "message": "\n".join(message)
    }


def k_partial_multi(
    obs_norms: np.ndarray,
    act_norms: np.ndarray,
    controls: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute K-Index controlling for multiple confounds.

    Uses semi-partial correlation approach:
    1. Regress obs_norms on controls → residuals_obs
    2. Regress act_norms on controls → residuals_act
    3. Compute K-Index on residuals

    Args:
        obs_norms: Observation norms [T]
        act_norms: Action norms [T]
        controls: Dictionary of confound variables to control for

    Returns:
        Dictionary with K-Index before and after controlling

    Example:
        >>> controls = {
        ...     'reward': rewards,
        ...     'time': np.arange(len(obs)),
        ...     'episode': episode_ids
        ... }
        >>> result = k_partial_multi(obs, act, controls)
    """
    from sklearn.linear_model import LinearRegression

    if len(obs_norms) < len(controls) + 2:
        return {
            "k_raw": np.nan,
            "k_controlled": np.nan,
            "delta": np.nan
        }

    # Stack control variables
    X = np.column_stack([v for v in controls.values()])

    # Regress obs_norms on controls
    model_obs = LinearRegression()
    model_obs.fit(X, obs_norms)
    residuals_obs = obs_norms - model_obs.predict(X)

    # Regress act_norms on controls
    model_act = LinearRegression()
    model_act.fit(X, act_norms)
    residuals_act = act_norms - model_act.predict(X)

    # K-Index on residuals
    r_residual, _ = pearsonr(residuals_obs, residuals_act)
    k_controlled = 2.0 * abs(r_residual)

    # Raw K-Index
    r_raw, _ = pearsonr(obs_norms, act_norms)
    k_raw = 2.0 * abs(r_raw)

    return {
        "k_raw": k_raw,
        "k_controlled": k_controlled,
        "delta": k_raw - k_controlled,
        "controls": list(controls.keys())
    }
