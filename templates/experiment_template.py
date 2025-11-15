#!/usr/bin/env python3
"""
Experiment Template for Kosmic Lab

This template provides a complete structure for running reproducible experiments
with K-Index analysis, K-Codex logging, and visualization.

Usage:
    1. Copy this template: cp templates/experiment_template.py my_experiment.py
    2. Modify the sections marked with TODO
    3. Run: python my_experiment.py

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Kosmic Lab imports
from core.utils import bootstrap_confidence_interval, infer_git_sha
from core.logging_config import setup_logging, get_logger
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index, bootstrap_k_ci
from fre.metrics.k_lag import k_lag

# Setup logging
setup_logging(level="INFO", log_file=f"logs/experiment_{datetime.now():%Y%m%d_%H%M%S}.log")
logger = get_logger(__name__)

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)


#====================================================================================
# SECTION 1: EXPERIMENT CONFIGURATION
# TODO: Modify these parameters for your experiment
#====================================================================================

EXPERIMENT_CONFIG = {
    "name": "my_experiment",  # TODO: Give your experiment a name
    "description": "Brief description of what this experiment does",  # TODO: Describe
    "n_samples": 1000,  # TODO: Set sample size
    "n_bootstrap": 1000,  # Bootstrap iterations for CI
    "confidence_level": 0.95,  # Confidence level for intervals
}


#====================================================================================
# SECTION 2: DATA GENERATION OR LOADING
# TODO: Replace with your actual data loading/generation logic
#====================================================================================

def load_or_generate_data(config: dict) -> tuple:
    """
    Load or generate experimental data.

    TODO: Replace this function with your actual data source.
    This could be:
    - Loading from files (CSV, NPY, HDF5, etc.)
    - Querying a database
    - Generating synthetic data
    - Processing real-world measurements

    Args:
        config: Experiment configuration dictionary

    Returns:
        Tuple of (observed_data, actual_data, metadata)
    """
    logger.info("Loading/generating data...")

    # TODO: Replace this with your data loading logic
    # Example: Synthetic data with known correlation
    n = config["n_samples"]

    # Generate actual data
    actual_data = rng.normal(0, 1, n)

    # Generate observed data (correlated with actual)
    noise = rng.normal(0, 0.2, n)
    observed_data = 0.8 * actual_data + 0.2 * noise

    # Metadata about the data
    metadata = {
        "data_source": "synthetic",
        "generation_method": "linear_combination",
        "noise_level": 0.2,
        "correlation_strength": 0.8,
    }

    logger.info(f"Data loaded: {len(observed_data)} samples")
    return observed_data, actual_data, metadata


#====================================================================================
# SECTION 3: ANALYSIS
# TODO: Customize analysis as needed
#====================================================================================

def analyze_data(observed: np.ndarray, actual: np.ndarray, config: dict) -> dict:
    """
    Perform K-Index analysis on the data.

    TODO: Add additional analyses as needed:
    - K-Lag analysis for temporal data
    - Statistical tests
    - Custom metrics

    Args:
        observed: Observed data
        actual: Actual/ground truth data
        config: Configuration dictionary

    Returns:
        Dictionary with analysis results
    """
    logger.info("Performing analysis...")

    # Compute K-Index with bootstrap CI
    k, ci_low, ci_high = bootstrap_k_ci(
        observed,
        actual,
        n_bootstrap=config["n_bootstrap"],
        confidence_level=config["confidence_level"],
        seed=SEED
    )

    # Additional metrics
    correlation = np.corrcoef(observed, actual)[0, 1]
    rmse = np.sqrt(np.mean((observed - actual) ** 2))

    results = {
        "k_index": k,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "ci_width": ci_high - ci_low,
        "pearson_r": correlation,
        "rmse": rmse,
        "statistically_significant": ci_low > 0,
    }

    logger.info(f"K-Index: {k:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    logger.info(f"Pearson r: {correlation:.4f}, RMSE: {rmse:.4f}")

    # TODO: Add K-Lag analysis if data is temporal
    # lag_results = k_lag(observed, actual, max_lag=50, seed=SEED)
    # results["lag_analysis"] = lag_results

    return results


#====================================================================================
# SECTION 4: VISUALIZATION
# TODO: Customize plots for your needs
#====================================================================================

def create_visualization(observed: np.ndarray, actual: np.ndarray, results: dict) -> plt.Figure:
    """
    Create visualization of analysis results.

    TODO: Customize plots for your experiment:
    - Add additional subplots
    - Change plot types
    - Add annotations

    Args:
        observed: Observed data
        actual: Actual data
        results: Analysis results

    Returns:
        Matplotlib Figure object
    """
    logger.info("Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Scatter plot
    ax = axes[0, 0]
    ax.scatter(actual, observed, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
            'r--', linewidth=2, label='Perfect correlation')
    ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
    ax.set_ylabel("Observed", fontsize=12, fontweight='bold')
    ax.set_title("Observed vs Actual", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: K-Index bar
    ax = axes[0, 1]
    ax.bar(['K-Index'], [results['k_index']],
           yerr=[[results['k_index'] - results['ci_lower']],
                 [results['ci_upper'] - results['k_index']]],
           capsize=10, color='steelblue', edgecolor='black', linewidth=2, alpha=0.7)
    ax.set_ylabel("K-Index", fontsize=12, fontweight='bold')
    ax.set_title(f"K = {results['k_index']:.4f} [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]",
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Residuals
    ax = axes[1, 0]
    residuals = observed - actual
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel("Residual (Observed - Actual)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax.set_title(f"Residual Distribution (RMSE={results['rmse']:.4f})", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    EXPERIMENT SUMMARY

    K-Index: {results['k_index']:.4f}
    95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]
    CI Width: {results['ci_width']:.4f}

    Pearson r: {results['pearson_r']:.4f}
    RMSE: {results['rmse']:.4f}

    Statistically Significant: {'✓ Yes' if results['statistically_significant'] else '✗ No'}

    Samples: {len(observed):,}
    Bootstrap Iterations: {EXPERIMENT_CONFIG['n_bootstrap']:,}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.suptitle(EXPERIMENT_CONFIG['name'].replace('_', ' ').title(),
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


#====================================================================================
# SECTION 5: MAIN EXPERIMENT
#====================================================================================

def main():
    """
    Main experiment execution.

    This function orchestrates the entire experiment:
    1. Load/generate data
    2. Perform analysis
    3. Create visualizations
    4. Log to K-Codex
    5. Save outputs
    """
    logger.info("=" * 80)
    logger.info(f"Starting Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info("=" * 80)

    # Create K-Codex writer for reproducibility
    kcodex_path = Path(f"logs/{EXPERIMENT_CONFIG['name']}_kcodex.json")
    kcodex_path.parent.mkdir(parents=True, exist_ok=True)
    kcodex = KCodexWriter(str(kcodex_path))

    # 1. Load/generate data
    observed, actual, metadata = load_or_generate_data(EXPERIMENT_CONFIG)

    # 2. Perform analysis
    results = analyze_data(observed, actual, EXPERIMENT_CONFIG)

    # 3. Create visualization
    fig = create_visualization(observed, actual, results)

    # 4. Save visualization
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{EXPERIMENT_CONFIG['name']}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Visualization saved: {output_path}")

    # 5. Log to K-Codex
    kcodex.log_experiment(
        experiment_name=EXPERIMENT_CONFIG['name'],
        params={
            **EXPERIMENT_CONFIG,
            **metadata,
        },
        metrics={
            "k_index": results["k_index"],
            "ci_lower": results["ci_lower"],
            "ci_upper": results["ci_upper"],
            "pearson_r": results["pearson_r"],
            "rmse": results["rmse"],
        },
        seed=SEED,
        extra_metadata={
            "template_version": "1.0.0",
            "author": "Your Name",  # TODO: Update with your name
        }
    )
    logger.info(f"✅ K-Codex saved: {kcodex_path}")

    # 6. Display results
    plt.show()

    # 7. Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Git SHA: {infer_git_sha()}")
    logger.info(f"Results: K-Index = {results['k_index']:.4f} [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    logger.info(f"Output: {output_path}")
    logger.info(f"K-Codex: {kcodex_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
