#!/usr/bin/env python3
"""
Advanced K-Index Analysis Example
==================================

This example demonstrates advanced K-Index analysis techniques:
- Bootstrap confidence intervals
- Multiple comparison testing
- Time series analysis with K-Lag
- Visualization of results

Prerequisites:
    poetry install

Usage:
    poetry run python examples/02_advanced_k_index.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.logging_config import get_logger, setup_logging
from core.utils import bootstrap_confidence_interval
from fre.metrics.k_index import k_index
from fre.metrics.k_lag import k_lag

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def generate_correlated_data(
    n: int, correlation: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two correlated time series for testing.

    Args:
        n: Number of samples
        correlation: Target correlation coefficient (-1 to 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (observed, actual) arrays
    """
    rng = np.random.default_rng(seed)

    # Generate correlated Gaussian data
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    data = rng.multivariate_normal(mean, cov, n)

    observed = np.abs(data[:, 0])
    actual = np.abs(data[:, 1])

    return observed, actual


def compare_multiple_scenarios() -> None:
    """Compare K-Index across multiple correlation scenarios."""
    logger.info("=" * 80)
    logger.info("Comparing Multiple Correlation Scenarios")
    logger.info("=" * 80)

    correlations = [0.0, 0.3, 0.5, 0.7, 0.9]
    n_samples = 1000

    results = []

    for corr in correlations:
        obs, act = generate_correlated_data(n_samples, corr)
        k = k_index(obs, act)

        # Compute bootstrap CI
        k_estimate, ci_lower, ci_upper = bootstrap_confidence_interval(
            obs, lambda x: k_index(x, act), n_bootstrap=1000, confidence=0.95
        )

        results.append(
            {
                "correlation": corr,
                "k_index": k,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_width": ci_upper - ci_lower,
            }
        )

        logger.info(f"\nCorrelation = {corr:.1f}:")
        logger.info(f"  K-Index: {k:.4f}")
        logger.info(f"  95% CI:  [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  CI Width: {ci_upper - ci_lower:.4f}")

    return results


def analyze_time_series() -> None:
    """Demonstrate K-Lag analysis for time series data."""
    logger.info("\n" + "=" * 80)
    logger.info("Time Series K-Lag Analysis")
    logger.info("=" * 80)

    # Generate time series with autocorrelation
    n = 500
    rng = np.random.default_rng(42)

    # AR(1) process: x_t = 0.7 * x_{t-1} + noise
    noise = rng.standard_normal(n)
    observed = np.zeros(n)
    for t in range(1, n):
        observed[t] = 0.7 * observed[t - 1] + noise[t]

    # Lagged version (delayed by 5 steps)
    lag_delay = 5
    actual = np.roll(observed, lag_delay)
    actual[:lag_delay] = 0  # Handle boundary

    # Make absolute (required for K-Index)
    observed = np.abs(observed)
    actual = np.abs(actual)

    # Compute K-Lag
    max_lag = 20
    lag_results = k_lag(observed, actual, max_lag=max_lag)

    logger.info(f"\nBest lag: {lag_results['best_lag']}")
    logger.info(f"K-Index at best lag: {lag_results['k_at_best_lag']:.4f}")
    logger.info(f"All correlations (first 10 lags):")

    for i in range(min(10, len(lag_results["correlations"]))):
        logger.info(f"  Lag {i:2d}: {lag_results['correlations'][i]:.4f}")

    return lag_results


def visualize_results(
    comparison_results: list[dict], lag_results: dict
) -> None:
    """
    Create visualizations of the analysis results.

    Args:
        comparison_results: Results from compare_multiple_scenarios()
        lag_results: Results from analyze_time_series()
    """
    logger.info("\n" + "=" * 80)
    logger.info("Creating Visualizations")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path("logs/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: K-Index vs Correlation with CI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    correlations = [r["correlation"] for r in comparison_results]
    k_values = [r["k_index"] for r in comparison_results]
    ci_lower = [r["ci_lower"] for r in comparison_results]
    ci_upper = [r["ci_upper"] for r in comparison_results]

    ax1.plot(correlations, k_values, "o-", linewidth=2, markersize=8, label="K-Index")
    ax1.fill_between(correlations, ci_lower, ci_upper, alpha=0.3, label="95% CI")
    ax1.set_xlabel("True Correlation", fontsize=12)
    ax1.set_ylabel("K-Index", fontsize=12)
    ax1.set_title("K-Index vs True Correlation", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Figure 2: K-Lag Analysis
    lags = list(range(len(lag_results["correlations"])))
    ax2.plot(
        lags,
        lag_results["correlations"],
        "o-",
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    ax2.axvline(
        lag_results["best_lag"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best Lag = {lag_results['best_lag']}",
    )
    ax2.set_xlabel("Lag", fontsize=12)
    ax2.set_ylabel("Correlation", fontsize=12)
    ax2.set_title("K-Lag Analysis", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "02_advanced_k_index_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"\nVisualization saved to: {output_path}")

    plt.close()


def statistical_power_analysis() -> None:
    """
    Demonstrate statistical power analysis for K-Index.

    Shows how sample size affects the precision of K-Index estimates.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Statistical Power Analysis")
    logger.info("=" * 80)

    sample_sizes = [50, 100, 200, 500, 1000, 2000]
    true_correlation = 0.7
    n_trials = 100

    results = []

    for n in sample_sizes:
        k_values = []

        for trial in range(n_trials):
            obs, act = generate_correlated_data(n, true_correlation, seed=trial)
            k = k_index(obs, act)
            k_values.append(k)

        k_values = np.array(k_values)

        results.append(
            {
                "n": n,
                "mean_k": np.mean(k_values),
                "std_k": np.std(k_values),
                "se_k": np.std(k_values) / np.sqrt(n_trials),
            }
        )

        logger.info(f"\nN = {n}:")
        logger.info(f"  Mean K-Index: {np.mean(k_values):.4f}")
        logger.info(f"  Std Dev:      {np.std(k_values):.4f}")
        logger.info(f"  Std Error:    {np.std(k_values) / np.sqrt(n_trials):.4f}")

    logger.info("\n📊 Conclusion:")
    logger.info(
        "  Larger samples give more precise K-Index estimates (lower std deviation)"
    )
    logger.info("  For stable estimates, use N >= 200")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Advanced K-Index Analysis Example")
    logger.info("=" * 80)
    logger.info("")

    # Part 1: Compare multiple scenarios
    comparison_results = compare_multiple_scenarios()

    # Part 2: Time series analysis
    lag_results = analyze_time_series()

    # Part 3: Statistical power
    statistical_power_analysis()

    # Part 4: Visualizations
    try:
        visualize_results(comparison_results, lag_results)
    except ImportError:
        logger.warning(
            "\nMatplotlib not available - skipping visualizations. "
            "Install with: poetry add matplotlib"
        )

    logger.info("\n" + "=" * 80)
    logger.info("Analysis Complete!")
    logger.info("=" * 80)
    logger.info("\n📚 Key Takeaways:")
    logger.info("  1. K-Index increases with true correlation")
    logger.info("  2. Bootstrap CIs provide uncertainty estimates")
    logger.info("  3. K-Lag identifies temporal relationships")
    logger.info("  4. Larger samples improve estimate precision")
    logger.info("\n🚀 Next Steps:")
    logger.info("  - Try with your own data")
    logger.info("  - Experiment with different correlation patterns")
    logger.info("  - Run examples/03_multi_universe.py for multi-universe analysis")


if __name__ == "__main__":
    main()
