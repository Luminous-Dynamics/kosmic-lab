"""
Null Distributions and FDR Correction

Generate null distributions for K-Index to establish statistical significance:
1. Shuffled: Random permutation of obs_norms (breaks temporal structure)
2. Random: Independent Gaussian noise (breaks all structure)
3. Magnitude-matched: Preserve marginal distributions but break correlation

Apply Benjamini-Hochberg FDR correction for multiple comparisons.
"""
import numpy as np
from scipy.stats import pearsonr
from typing import Any, Dict, List, Optional, Tuple


def null_k_distributions(
    obs_norms: np.ndarray,
    act_norms: np.ndarray,
    n: int = 1000,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, Any]:
    """
    Generate null distributions for K-Index.

    Args:
        obs_norms: Observation norms [T]
        act_norms: Action norms [T]
        n: Number of null samples (default: 1000)
        rng: Random number generator

    Returns:
        Dictionary with:
        - k_empirical: True K-Index
        - null_shuffled: K-Index distribution from shuffled obs
        - null_random: K-Index distribution from random obs
        - null_mag_matched: K-Index distribution preserving marginals
        - p_shuffled: p-value vs shuffled null
        - p_random: p-value vs random null
        - p_mag_matched: p-value vs magnitude-matched null

    Example:
        >>> obs = np.random.randn(100)
        >>> act = obs + np.random.randn(100) * 0.1
        >>> nulls = null_k_distributions(obs, act, n=1000)
        >>> assert nulls['k_empirical'] > nulls['p95_shuffled']
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(obs_norms)

    # Empirical K-Index
    r_emp, _ = pearsonr(obs_norms, act_norms)
    k_emp = 2.0 * abs(r_emp)

    # 1. Shuffled null (permutation)
    k_shuffled = []
    for _ in range(n):
        obs_shuf = rng.permutation(obs_norms)
        r, _ = pearsonr(obs_shuf, act_norms)
        k_shuffled.append(2.0 * abs(r))
    k_shuffled = np.array(k_shuffled)

    # 2. Random null (Gaussian noise)
    k_random = []
    for _ in range(n):
        obs_rand = rng.standard_normal(T)
        r, _ = pearsonr(obs_rand, act_norms)
        k_random.append(2.0 * abs(r))
    k_random = np.array(k_random)

    # 3. Magnitude-matched null (preserve marginals, break correlation)
    k_mag_matched = []
    obs_sorted = np.sort(obs_norms)
    for _ in range(n):
        # Random permutation preserves distribution
        obs_perm = rng.permutation(obs_sorted)
        r, _ = pearsonr(obs_perm, act_norms)
        k_mag_matched.append(2.0 * abs(r))
    k_mag_matched = np.array(k_mag_matched)

    # Compute p-values (one-tailed: k_emp > null)
    p_shuffled = (k_shuffled >= k_emp).mean()
    p_random = (k_random >= k_emp).mean()
    p_mag_matched = (k_mag_matched >= k_emp).mean()

    return {
        "k_empirical": k_emp,
        "null_shuffled": k_shuffled,
        "null_random": k_random,
        "null_mag_matched": k_mag_matched,
        "p_shuffled": p_shuffled,
        "p_random": p_random,
        "p_mag_matched": p_mag_matched,
        "p95_shuffled": np.percentile(k_shuffled, 95),
        "p95_random": np.percentile(k_random, 95),
        "p95_mag_matched": np.percentile(k_mag_matched, 95),
        "p99_shuffled": np.percentile(k_shuffled, 99),
        "p99_random": np.percentile(k_random, 99),
        "p99_mag_matched": np.percentile(k_mag_matched, 99)
    }


def verify_significance(
    null_result: Dict[str, any],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Verify K-Index is significantly above null distributions.

    Args:
        null_result: Output from null_k_distributions()
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with verification results

    Example:
        >>> nulls = null_k_distributions(obs, act)
        >>> verify = verify_significance(nulls)
        >>> assert verify['significant_all']
    """
    k_emp = null_result['k_empirical']
    p_shuf = null_result['p_shuffled']
    p_rand = null_result['p_random']
    p_mag = null_result['p_mag_matched']

    sig_shuf = p_shuf < alpha
    sig_rand = p_rand < alpha
    sig_mag = p_mag < alpha
    sig_all = sig_shuf and sig_rand and sig_mag

    message = [f"K-Index = {k_emp:.3f}"]
    if sig_shuf:
        message.append(f"✅ Significant vs shuffled (p={p_shuf:.4f})")
    else:
        message.append(f"❌ Not significant vs shuffled (p={p_shuf:.4f})")

    if sig_rand:
        message.append(f"✅ Significant vs random (p={p_rand:.4f})")
    else:
        message.append(f"❌ Not significant vs random (p={p_rand:.4f})")

    if sig_mag:
        message.append(f"✅ Significant vs magnitude-matched (p={p_mag:.4f})")
    else:
        message.append(f"❌ Not significant vs magnitude-matched (p={p_mag:.4f})")

    return {
        "significant_shuffled": sig_shuf,
        "significant_random": sig_rand,
        "significant_mag_matched": sig_mag,
        "significant_all": sig_all,
        "p_values": {
            "shuffled": p_shuf,
            "random": p_rand,
            "mag_matched": p_mag
        },
        "message": "\n".join(message)
    }


def pairwise_fdr(
    conditions: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform pairwise t-tests with Benjamini-Hochberg FDR correction.

    Args:
        conditions: Dictionary mapping condition names to K-Index arrays
        alpha: FDR level (default: 0.05)

    Returns:
        Dictionary with:
        - comparisons: List of (cond1, cond2, p_raw, p_fdr, significant)
        - n_comparisons: Total number of comparisons
        - n_significant: Number passing FDR threshold

    Example:
        >>> conditions = {
        ...     'baseline': np.array([0.6, 0.65, 0.7]),
        ...     'adversarial': np.array([1.1, 1.15, 1.2])
        ... }
        >>> result = pairwise_fdr(conditions)
        >>> assert result['n_significant'] > 0
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    cond_names = list(conditions.keys())
    n_conds = len(cond_names)

    # Pairwise comparisons
    comparisons = []
    p_values = []

    for i in range(n_conds):
        for j in range(i + 1, n_conds):
            name1 = cond_names[i]
            name2 = cond_names[j]
            data1 = conditions[name1]
            data2 = conditions[name2]

            # Two-sample t-test
            t_stat, p_val = ttest_ind(data1, data2)
            comparisons.append((name1, name2))
            p_values.append(p_val)

    # Benjamini-Hochberg FDR correction
    if len(p_values) > 0:
        reject, p_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    else:
        reject = []
        p_fdr = []

    # Format results
    results = []
    for (name1, name2), p_raw, p_adj, sig in zip(comparisons, p_values, p_fdr, reject):
        mean1 = conditions[name1].mean()
        mean2 = conditions[name2].mean()
        diff = mean2 - mean1

        results.append({
            "comparison": f"{name1} vs {name2}",
            "mean_1": mean1,
            "mean_2": mean2,
            "difference": diff,
            "p_raw": p_raw,
            "p_fdr": p_adj,
            "significant": bool(sig)
        })

    n_sig = sum(r['significant'] for r in results)

    return {
        "comparisons": results,
        "n_comparisons": len(comparisons),
        "n_significant": n_sig,
        "alpha": alpha,
        "method": "Benjamini-Hochberg FDR"
    }


def plot_null_distributions(
    null_result: Dict[str, any],
    title: str = "K-Index vs Null Distributions",
    save_path: Optional[str] = None
):
    """
    Plot empirical K-Index vs null distributions.

    Args:
        null_result: Output from null_k_distributions()
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> nulls = null_k_distributions(obs, act)
        >>> plot_null_distributions(nulls, save_path="null_distributions.png")
    """
    import matplotlib.pyplot as plt

    k_emp = null_result['k_empirical']
    k_shuf = null_result['null_shuffled']
    k_rand = null_result['null_random']
    k_mag = null_result['null_mag_matched']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Shuffled null
    axes[0].hist(k_shuf, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(k_emp, color='red', linestyle='--', linewidth=2, label='Empirical')
    axes[0].axvline(null_result['p95_shuffled'], color='orange', linestyle=':', label='95%')
    axes[0].set_xlabel('K-Index')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Shuffled Null')
    axes[0].legend()

    # 2. Random null
    axes[1].hist(k_rand, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(k_emp, color='red', linestyle='--', linewidth=2, label='Empirical')
    axes[1].axvline(null_result['p95_random'], color='orange', linestyle=':', label='95%')
    axes[1].set_xlabel('K-Index')
    axes[1].set_title('Random Null')
    axes[1].legend()

    # 3. Magnitude-matched null
    axes[2].hist(k_mag, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(k_emp, color='red', linestyle='--', linewidth=2, label='Empirical')
    axes[2].axvline(null_result['p95_mag_matched'], color='orange', linestyle=':', label='95%')
    axes[2].set_xlabel('K-Index')
    axes[2].set_title('Magnitude-Matched Null')
    axes[2].legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved null distributions plot to {save_path}")
    else:
        plt.show()

    plt.close()
