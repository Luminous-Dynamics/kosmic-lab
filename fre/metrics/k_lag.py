"""
Time-Lag Analysis for K-Index

Compute K(τ) = 2|ρ(||O_t||, ||A_{t+τ}||)| across time lags τ.

This verifies causal direction: observations → actions.
Expected: peak at τ = 0, declining for τ < 0 and τ > 0.
"""
import numpy as np
from scipy.stats import pearsonr
from typing import Any, Dict, Optional


def k_lag(
    obs_norms: np.ndarray,
    act_norms: np.ndarray,
    max_lag: int = 10
) -> Dict[str, Any]:
    """
    Compute K-Index across time lags.

    Args:
        obs_norms: Observation norms [T]
        act_norms: Action norms [T]
        max_lag: Maximum lag to test (default: 10)

    Returns:
        Dictionary with:
        - k_by_lag: Dict[int, float] mapping lag → K-Index
        - peak_lag: Lag with maximum K-Index
        - peak_k: K-Index value at peak lag
        - k_at_zero: K-Index at τ=0 (synchronous)

    Example:
        >>> T = 100
        >>> obs = np.random.randn(T)
        >>> act = np.roll(obs, 1) + np.random.randn(T) * 0.1  # act follows obs
        >>> result = k_lag(obs, act, max_lag=5)
        >>> assert result['peak_lag'] >= 0, "Actions should follow observations"
    """
    T = len(obs_norms)
    if T < max_lag + 2:
        return {
            "k_by_lag": {},
            "peak_lag": np.nan,
            "peak_k": np.nan,
            "k_at_zero": np.nan
        }

    k_by_lag = {}

    # Compute K(τ) for τ ∈ [-max_lag, +max_lag]
    for tau in range(-max_lag, max_lag + 1):
        if tau == 0:
            # Synchronous: ρ(O_t, A_t)
            r, _ = pearsonr(obs_norms, act_norms)
        elif tau > 0:
            # Positive lag: ρ(O_t, A_{t+τ})
            # Actions follow observations (expected)
            o = obs_norms[:-tau]
            a = act_norms[tau:]
            if len(o) < 2:
                k_by_lag[tau] = np.nan
                continue
            r, _ = pearsonr(o, a)
        else:  # tau < 0
            # Negative lag: ρ(O_t, A_{t+τ}) = ρ(O_{t-|τ|}, A_t)
            # Observations follow actions (unexpected)
            tau_abs = abs(tau)
            o = obs_norms[tau_abs:]
            a = act_norms[:-tau_abs]
            if len(o) < 2:
                k_by_lag[tau] = np.nan
                continue
            r, _ = pearsonr(o, a)

        k_by_lag[tau] = 2.0 * abs(r)

    # Find peak lag
    valid = {tau: k for tau, k in k_by_lag.items() if not np.isnan(k)}
    if len(valid) == 0:
        return {
            "k_by_lag": k_by_lag,
            "peak_lag": np.nan,
            "peak_k": np.nan,
            "k_at_zero": k_by_lag.get(0, np.nan)
        }

    peak_lag = max(valid, key=valid.get)
    peak_k = valid[peak_lag]

    return {
        "k_by_lag": k_by_lag,
        "peak_lag": peak_lag,
        "peak_k": peak_k,
        "k_at_zero": k_by_lag.get(0, np.nan)
    }


def verify_causal_direction(lag_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that peak K-Index occurs at τ ≥ 0.

    Expected: Actions follow observations, so peak at τ=0 or τ>0.
    If peak at τ<0, this suggests reverse causality (unlikely).

    Args:
        lag_result: Output from k_lag()

    Returns:
        Dictionary with verification results

    Example:
        >>> result = k_lag(obs, act, max_lag=10)
        >>> verify = verify_causal_direction(result)
        >>> assert verify['causal_direction_correct']
    """
    peak_lag = lag_result['peak_lag']
    k_at_zero = lag_result['k_at_zero']
    peak_k = lag_result['peak_k']

    if np.isnan(peak_lag):
        return {
            "causal_direction_correct": None,
            "message": "Insufficient data for lag analysis"
        }

    # Check if peak is at τ ≥ 0
    correct = peak_lag >= 0

    # Additional check: K(0) should be comparable to peak
    if not np.isnan(k_at_zero) and not np.isnan(peak_k):
        ratio = k_at_zero / peak_k if peak_k > 0 else np.nan
    else:
        ratio = np.nan

    message = []
    if correct:
        message.append(f"✅ Peak at τ={peak_lag} (actions follow observations)")
    else:
        message.append(f"⚠️ Peak at τ={peak_lag} (reverse causality detected)")

    if not np.isnan(ratio):
        if ratio >= 0.8:
            message.append(f"✅ K(0)={k_at_zero:.3f} ≈ peak K={peak_k:.3f}")
        else:
            message.append(f"⚠️ K(0)={k_at_zero:.3f} << peak K={peak_k:.3f}")

    return {
        "causal_direction_correct": correct,
        "peak_lag": peak_lag,
        "k_at_zero": k_at_zero,
        "peak_k": peak_k,
        "sync_to_peak_ratio": ratio,
        "message": " | ".join(message)
    }


def plot_k_lag(
    lag_result: Dict[str, Any],
    title: str = "K-Index vs Time Lag",
    save_path: Optional[str] = None
):
    """
    Plot K-Index across time lags.

    Args:
        lag_result: Output from k_lag()
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> result = k_lag(obs, act, max_lag=10)
        >>> plot_k_lag(result, save_path="k_lag_analysis.png")
    """
    import matplotlib.pyplot as plt

    k_by_lag = lag_result['k_by_lag']
    peak_lag = lag_result['peak_lag']

    lags = sorted(k_by_lag.keys())
    k_values = [k_by_lag[tau] for tau in lags]

    plt.figure(figsize=(10, 6))
    plt.plot(lags, k_values, 'o-', linewidth=2, markersize=6)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5, label='τ=0 (synchronous)')

    if not np.isnan(peak_lag):
        plt.axvline(peak_lag, color='red', linestyle='--', alpha=0.7,
                    label=f'Peak at τ={peak_lag}')

    plt.xlabel('Time Lag τ (steps)', fontsize=12)
    plt.ylabel('K-Index = 2|ρ(||O_t||, ||A_{t+τ}||)|', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved K-lag plot to {save_path}")
    else:
        plt.show()

    plt.close()
