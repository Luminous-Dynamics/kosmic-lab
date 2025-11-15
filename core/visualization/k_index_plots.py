"""
K-Index Specific Visualization Functions

Publication-quality plots for K-Index metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Dict
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fre.metrics.k_index import k_index, bootstrap_k_ci
from fre.metrics.k_lag import k_lag


def plot_k_index_ci(
    observed: np.ndarray,
    actual: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    title: str = "K-Index with Confidence Interval",
    figsize: Tuple[int, int] = (10, 6),
    seed: Optional[int] = None
) -> Figure:
    """
    Create a comprehensive K-Index visualization with confidence intervals.

    Args:
        observed: Observed data
        actual: Actual data
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default: 0.95)
        title: Plot title
        figsize: Figure size
        seed: Random seed for reproducibility

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_k_index_ci(obs, act, title="My Experiment")
        >>> fig.savefig("k_index.png", dpi=300)
    """
    # Compute K-Index with CI
    k, ci_low, ci_high = bootstrap_k_ci(
        observed, actual,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Scatter with perfect correlation line
    ax1.scatter(actual, observed, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
             'r--', linewidth=2, label='Perfect correlation')
    ax1.set_xlabel("Actual", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Observed", fontsize=12, fontweight='bold')
    ax1.set_title("Observed vs Actual", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: K-Index bar with error bars
    ax2.bar(['K-Index'], [k], yerr=[[k - ci_low], [ci_high - k]],
            capsize=10, color='steelblue', edgecolor='black', linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel("K-Index", fontsize=12, fontweight='bold')
    ax2.set_title(f"K = {k:.4f} [{ci_low:.4f}, {ci_high:.4f}]",
                  fontsize=14, fontweight='bold')
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(axis='y', alpha=0.3)

    # Add significance annotation
    if ci_low > 0:
        ax2.text(0, k + 0.05, '*** Significant', ha='center',
                fontsize=10, color='green', fontweight='bold')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_k_index_comparison(
    data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bootstrap: int = 1000,
    figsize: Tuple[int, int] = (12, 6),
    seed: Optional[int] = None
) -> Figure:
    """
    Compare K-Index across multiple datasets/conditions.

    Args:
        data_dict: Dictionary mapping names to (observed, actual) tuples
        n_bootstrap: Number of bootstrap iterations
        figsize: Figure size
        seed: Random seed

    Returns:
        Matplotlib Figure

    Example:
        >>> data = {
        ...     "Condition A": (obs_a, act_a),
        ...     "Condition B": (obs_b, act_b),
        ...     "Condition C": (obs_c, act_c)
        ... }
        >>> fig = plot_k_index_comparison(data)
    """
    # Compute K-Index for all conditions
    names = list(data_dict.keys())
    k_values = []
    ci_lows = []
    ci_highs = []

    for name, (obs, act) in data_dict.items():
        k, ci_low, ci_high = bootstrap_k_ci(
            obs, act,
            n_bootstrap=n_bootstrap,
            seed=seed
        )
        k_values.append(k)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    # Plot bars with error bars
    bars = ax.bar(x_pos, k_values,
                   yerr=[np.array(k_values) - np.array(ci_lows),
                         np.array(ci_highs) - np.array(k_values)],
                   capsize=8, color=colors, edgecolor='black',
                   linewidth=2, alpha=0.7)

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel("K-Index", fontsize=14, fontweight='bold')
    ax.set_title("K-Index Comparison Across Conditions",
                 fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10)

    # Add value labels
    for i, (bar, k) in enumerate(zip(bars, k_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{k:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_k_lag(
    observed: np.ndarray,
    actual: np.ndarray,
    max_lag: int = 50,
    figsize: Tuple[int, int] = (14, 5),
    seed: Optional[int] = None
) -> Figure:
    """
    Visualize K-Lag temporal analysis.

    Args:
        observed: Observed time series
        actual: Actual time series
        max_lag: Maximum lag to test
        figsize: Figure size
        seed: Random seed

    Returns:
        Matplotlib Figure

    Example:
        >>> fig = plot_k_lag(obs_ts, act_ts, max_lag=50)
    """
    # Perform K-Lag analysis
    results = k_lag(observed, actual, max_lag=max_lag, seed=seed)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: K-Index vs Lag
    ax1.plot(results['lags'], results['k_values'], 'o-',
             linewidth=2, markersize=6, color='steelblue')
    ax1.axvline(results['best_lag'], color='red', linestyle='--',
                linewidth=2, label=f"Best lag: {results['best_lag']}")
    ax1.axvline(0, color='green', linestyle='--',
                linewidth=1, alpha=0.5, label="Zero lag")
    ax1.set_xlabel("Lag", fontsize=12, fontweight='bold')
    ax1.set_ylabel("K-Index", fontsize=12, fontweight='bold')
    ax1.set_title(f"K-Lag Analysis (Best: {results['best_lag']})",
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: Time series comparison
    n_display = min(200, len(observed))
    t = np.arange(n_display)
    ax2.plot(t, actual[:n_display], 'b-', linewidth=2,
             label='Actual', alpha=0.7)
    ax2.plot(t, observed[:n_display], 'r--', linewidth=2,
             label='Observed', alpha=0.7)
    ax2.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Value", fontsize=12, fontweight='bold')
    ax2.set_title("Time Series Comparison",
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_k_distribution(
    k_samples: np.ndarray,
    true_k: Optional[float] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot distribution of K-Index values (e.g., from bootstrap or simulations).

    Args:
        k_samples: Array of K-Index samples
        true_k: True K-Index value (if known)
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib Figure

    Example:
        >>> k_bootstrap = [k_index(obs, act) for _ in range(1000)]
        >>> fig = plot_k_distribution(k_bootstrap, true_k=0.75)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    counts, edges, patches = ax.hist(k_samples, bins=bins, alpha=0.7,
                                      edgecolor='black', linewidth=1.5,
                                      color='steelblue')

    # Add mean line
    mean_k = np.mean(k_samples)
    ax.axvline(mean_k, color='red', linestyle='--',
               linewidth=2, label=f'Mean: {mean_k:.4f}')

    # Add true K if provided
    if true_k is not None:
        ax.axvline(true_k, color='green', linestyle='--',
                   linewidth=2, label=f'True: {true_k:.4f}')

    # Add confidence interval
    ci_low, ci_high = np.percentile(k_samples, [2.5, 97.5])
    ax.axvspan(ci_low, ci_high, alpha=0.2, color='gray',
               label=f'95% CI: [{ci_low:.4f}, {ci_high:.4f}]')

    # Styling
    ax.set_xlabel("K-Index", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_title("K-Index Distribution", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Add statistics text
    stats_text = f"""
    n = {len(k_samples)}
    μ = {mean_k:.4f}
    σ = {np.std(k_samples):.4f}
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig
