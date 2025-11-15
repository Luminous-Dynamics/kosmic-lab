#!/usr/bin/env python3
"""
🧠 Example 5: Neuroscience EEG Analysis with K-Index

This example demonstrates how to use Kosmic Lab to analyze EEG data
for consciousness correlation studies. We'll examine the relationship
between predicted and actual consciousness states using K-Index metrics.

**Research Application**: Anesthesia depth monitoring, sleep stage classification,
consciousness level assessment in clinical settings.

**Use Case**: Hospital ICU monitoring consciousness levels in sedated patients
by comparing EEG-based predictions with clinical observations.

Author: Kosmic Lab Contributors
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import bootstrap_confidence_interval, infer_git_sha
from core.logging_config import setup_logging, get_logger
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index, bootstrap_k_ci
from fre.metrics.k_lag import k_lag, verify_causal_direction

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)


class EEGConsciousnessAnalyzer:
    """
    Analyzes EEG signals to predict consciousness levels and validates
    predictions using K-Index metrics.
    """

    def __init__(self, seed: int = 42):
        """Initialize the analyzer."""
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        logger.info("EEG Consciousness Analyzer initialized")

    def generate_synthetic_eeg(
        self,
        n_samples: int = 1000,
        consciousness_level: float = 0.8,
        noise_level: float = 0.2
    ) -> dict:
        """
        Generate synthetic EEG-like data with consciousness correlations.

        In real applications, this would be replaced with actual EEG preprocessing:
        - Bandpass filtering (0.5-45 Hz)
        - Artifact removal (EOG, EMG)
        - Feature extraction (power spectral density, entropy, connectivity)

        Args:
            n_samples: Number of time points
            consciousness_level: True consciousness level (0=unconscious, 1=fully conscious)
            noise_level: Amount of measurement noise

        Returns:
            Dictionary containing EEG features and consciousness metrics
        """
        logger.info(f"Generating synthetic EEG: N={n_samples}, consciousness={consciousness_level:.2f}")

        # Time vector (assuming 250 Hz sampling rate)
        sampling_rate = 250  # Hz
        time = np.arange(n_samples) / sampling_rate

        # Generate EEG-like signals with consciousness-dependent features
        # Feature 1: Alpha power (8-13 Hz) - increases with consciousness
        alpha_freq = 10  # Hz
        alpha_power = consciousness_level * np.sin(2 * np.pi * alpha_freq * time)
        alpha_power += self.rng.normal(0, noise_level, n_samples)

        # Feature 2: Theta power (4-8 Hz) - decreases with consciousness
        theta_freq = 6  # Hz
        theta_power = (1 - consciousness_level) * np.sin(2 * np.pi * theta_freq * time)
        theta_power += self.rng.normal(0, noise_level, n_samples)

        # Feature 3: Gamma power (30-100 Hz) - increases with consciousness
        gamma_freq = 40  # Hz
        gamma_power = consciousness_level * 0.5 * np.sin(2 * np.pi * gamma_freq * time)
        gamma_power += self.rng.normal(0, noise_level * 0.5, n_samples)

        # Feature 4: Signal complexity (approximate entropy proxy)
        complexity = consciousness_level + self.rng.normal(0, noise_level * 0.3, n_samples)

        # Combine features into consciousness index (EEG-based prediction)
        eeg_prediction = (
            0.4 * (alpha_power + 1) / 2 +  # Normalize to [0,1]
            0.3 * (1 - (theta_power + 1) / 2) +
            0.2 * (gamma_power + 1) / 2 +
            0.1 * complexity
        )

        # Clip to valid range
        eeg_prediction = np.clip(eeg_prediction, 0, 1)

        # True consciousness (with temporal variations)
        # Simulates natural fluctuations in consciousness state
        true_consciousness = consciousness_level + self.rng.normal(0, 0.05, n_samples)
        true_consciousness = np.clip(true_consciousness, 0, 1)

        return {
            "time": time,
            "eeg_prediction": eeg_prediction,
            "true_consciousness": true_consciousness,
            "alpha_power": alpha_power,
            "theta_power": theta_power,
            "gamma_power": gamma_power,
            "complexity": complexity,
            "sampling_rate": sampling_rate,
            "consciousness_level": consciousness_level
        }

    def compute_consciousness_coherence(
        self,
        eeg_prediction: np.ndarray,
        true_consciousness: np.ndarray,
        n_bootstrap: int = 1000
    ) -> dict:
        """
        Compute K-Index between EEG predictions and true consciousness.

        Args:
            eeg_prediction: EEG-derived consciousness predictions [0,1]
            true_consciousness: Actual consciousness levels [0,1]
            n_bootstrap: Number of bootstrap iterations for CI

        Returns:
            Dictionary with K-Index results and confidence intervals
        """
        logger.info("Computing consciousness coherence with K-Index...")

        # Compute K-Index with bootstrap CI
        k, ci_lower, ci_upper = bootstrap_k_ci(
            eeg_prediction,
            true_consciousness,
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            seed=self.seed
        )

        # Compute correlation for comparison
        correlation = np.corrcoef(eeg_prediction, true_consciousness)[0, 1]

        # Compute RMSE
        rmse = np.sqrt(np.mean((eeg_prediction - true_consciousness) ** 2))

        # Compute MAE
        mae = np.mean(np.abs(eeg_prediction - true_consciousness))

        results = {
            "k_index": k,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_upper - ci_lower,
            "pearson_r": correlation,
            "rmse": rmse,
            "mae": mae,
            "statistically_significant": ci_lower > 0
        }

        logger.info(f"K-Index: {k:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"Pearson r: {correlation:.4f}, RMSE: {rmse:.4f}")

        return results

    def analyze_temporal_dynamics(
        self,
        eeg_prediction: np.ndarray,
        true_consciousness: np.ndarray,
        max_lag: int = 50
    ) -> dict:
        """
        Analyze temporal relationships using K-Lag.

        This reveals if EEG predictions lead or lag true consciousness changes,
        which is critical for real-time monitoring systems.

        Args:
            eeg_prediction: EEG-derived predictions
            true_consciousness: True consciousness levels
            max_lag: Maximum lag to test (in samples)

        Returns:
            K-Lag analysis results
        """
        logger.info("Analyzing temporal dynamics with K-Lag...")

        # Perform K-Lag analysis
        results = k_lag(
            eeg_prediction,
            true_consciousness,
            max_lag=max_lag,
            seed=self.seed
        )

        # Verify causal direction
        is_correct, forward_k, backward_k = verify_causal_direction(
            eeg_prediction,
            true_consciousness,
            max_lag=max_lag
        )

        results["causal_direction_correct"] = is_correct
        results["forward_k"] = forward_k
        results["backward_k"] = backward_k

        logger.info(f"Best lag: {results['best_lag']} samples")
        logger.info(f"K at best lag: {results['k_at_best_lag']:.4f}")
        logger.info(f"Causal direction: {'Correct' if is_correct else 'Reversed'}")

        return results


def visualize_eeg_analysis(eeg_data: dict, coherence: dict, lag_results: dict):
    """
    Create comprehensive visualization of EEG consciousness analysis.

    Args:
        eeg_data: EEG data dictionary
        coherence: K-Index coherence results
        lag_results: K-Lag temporal analysis results
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Plot 1: EEG Features Over Time
    ax1 = fig.add_subplot(gs[0, :])
    time_subset = eeg_data["time"][:500]  # First 2 seconds
    ax1.plot(time_subset, eeg_data["alpha_power"][:500], label="Alpha Power (8-13 Hz)", alpha=0.7)
    ax1.plot(time_subset, eeg_data["theta_power"][:500], label="Theta Power (4-8 Hz)", alpha=0.7)
    ax1.plot(time_subset, eeg_data["gamma_power"][:500], label="Gamma Power (30-100 Hz)", alpha=0.7)
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_ylabel("Normalized Power", fontsize=12)
    ax1.set_title("EEG Band Powers Over Time", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: Consciousness Prediction vs Truth
    ax2 = fig.add_subplot(gs[1, 0])
    subset_size = 200
    ax2.scatter(
        eeg_data["true_consciousness"][:subset_size],
        eeg_data["eeg_prediction"][:subset_size],
        alpha=0.5,
        s=20
    )
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label="Perfect prediction")
    ax2.set_xlabel("True Consciousness Level", fontsize=12)
    ax2.set_ylabel("EEG Prediction", fontsize=12)
    ax2.set_title(f"Prediction Accuracy (K={coherence['k_index']:.3f})", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Plot 3: Time Series Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    time_compare = eeg_data["time"][:500]
    ax3.plot(time_compare, eeg_data["true_consciousness"][:500], 'b-', linewidth=2, label="True", alpha=0.7)
    ax3.plot(time_compare, eeg_data["eeg_prediction"][:500], 'r--', linewidth=2, label="Predicted", alpha=0.7)
    ax3.set_xlabel("Time (seconds)", fontsize=12)
    ax3.set_ylabel("Consciousness Level", fontsize=12)
    ax3.set_title("Prediction vs Reality", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)

    # Plot 4: K-Lag Analysis
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(lag_results["lags"], lag_results["k_values"], 'o-', linewidth=2, markersize=6)
    ax4.axvline(lag_results["best_lag"], color='red', linestyle='--', linewidth=2, label=f"Best lag: {lag_results['best_lag']}")
    ax4.axvline(0, color='green', linestyle='--', linewidth=1, alpha=0.5, label="Zero lag")
    ax4.set_xlabel("Lag (samples)", fontsize=12)
    ax4.set_ylabel("K-Index", fontsize=12)
    ax4.set_title("Temporal Dynamics (K-Lag Analysis)", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # Plot 5: Residual Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    residuals = eeg_data["eeg_prediction"] - eeg_data["true_consciousness"]
    ax5.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(residuals):.4f}")
    ax5.set_xlabel("Prediction Error", fontsize=12)
    ax5.set_ylabel("Frequency", fontsize=12)
    ax5.set_title(f"Residual Distribution (RMSE={coherence['rmse']:.4f})", fontsize=14, fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Plot 6: Summary Statistics
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    summary_text = f"""
    📊 EEG CONSCIOUSNESS ANALYSIS SUMMARY

    Dataset Information:
      • Samples: {len(eeg_data['time']):,} ({len(eeg_data['time']) / eeg_data['sampling_rate']:.1f} seconds)
      • Sampling Rate: {eeg_data['sampling_rate']} Hz
      • True Consciousness Level: {eeg_data['consciousness_level']:.2f}

    K-Index Metrics:
      • K-Index: {coherence['k_index']:.4f}
      • 95% CI: [{coherence['ci_lower']:.4f}, {coherence['ci_upper']:.4f}]
      • CI Width: {coherence['ci_width']:.4f}
      • Statistically Significant: {"✓ Yes" if coherence['statistically_significant'] else "✗ No"}

    Comparison Metrics:
      • Pearson r: {coherence['pearson_r']:.4f}
      • RMSE: {coherence['rmse']:.4f}
      • MAE: {coherence['mae']:.4f}

    Temporal Dynamics:
      • Best Lag: {lag_results['best_lag']} samples ({lag_results['best_lag'] / eeg_data['sampling_rate'] * 1000:.1f} ms)
      • K at Best Lag: {lag_results['k_at_best_lag']:.4f}
      • K at Zero Lag: {lag_results['k_at_zero_lag']:.4f}
      • Improvement: {(lag_results['k_at_best_lag'] - lag_results['k_at_zero_lag']):.4f}

    Clinical Interpretation:
      • Prediction Quality: {"Excellent" if coherence['k_index'] > 0.8 else "Good" if coherence['k_index'] > 0.6 else "Moderate" if coherence['k_index'] > 0.4 else "Poor"}
      • Temporal Delay: {abs(lag_results['best_lag'] / eeg_data['sampling_rate'] * 1000):.1f} ms {"(prediction leads)" if lag_results['best_lag'] < 0 else "(prediction lags)"}
      • Suitable for Real-Time: {"✓ Yes" if abs(lag_results['best_lag']) < 50 else "✗ No (too much lag)"}
    """

    ax6.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)

    plt.suptitle("🧠 Neuroscience EEG Consciousness Analysis with K-Index",
                 fontsize=16, fontweight="bold", y=0.995)

    return fig


def main():
    """Run complete EEG consciousness analysis example."""
    logger.info("=" * 80)
    logger.info("🧠 Example 5: Neuroscience EEG Consciousness Analysis")
    logger.info("=" * 80)

    # Create K-Codex writer for reproducibility
    kcodex_path = Path("logs/example_05_eeg_kcodex.json")
    kcodex_path.parent.mkdir(parents=True, exist_ok=True)
    kcodex = KCodexWriter(str(kcodex_path))

    # Initialize analyzer
    analyzer = EEGConsciousnessAnalyzer(seed=SEED)

    # Generate synthetic EEG data for different consciousness levels
    consciousness_levels = [0.9, 0.7, 0.4, 0.2]  # Awake, drowsy, light sedation, deep sedation

    all_results = []

    for idx, c_level in enumerate(consciousness_levels, 1):
        logger.info(f"\n--- Analysis {idx}/4: Consciousness Level = {c_level:.1f} ---")

        # Generate EEG data
        eeg_data = analyzer.generate_synthetic_eeg(
            n_samples=5000,  # 20 seconds at 250 Hz
            consciousness_level=c_level,
            noise_level=0.15
        )

        # Compute coherence
        coherence = analyzer.compute_consciousness_coherence(
            eeg_data["eeg_prediction"],
            eeg_data["true_consciousness"],
            n_bootstrap=1000
        )

        # Analyze temporal dynamics
        lag_results = analyzer.analyze_temporal_dynamics(
            eeg_data["eeg_prediction"],
            eeg_data["true_consciousness"],
            max_lag=50  # 200ms at 250 Hz
        )

        # Log to K-Codex
        kcodex.log_experiment(
            experiment_name=f"eeg_consciousness_analysis_level_{c_level:.1f}",
            params={
                "n_samples": len(eeg_data["time"]),
                "sampling_rate": eeg_data["sampling_rate"],
                "consciousness_level": c_level,
                "noise_level": 0.15,
                "max_lag": 50,
                "n_bootstrap": 1000
            },
            metrics={
                "k_index": coherence["k_index"],
                "ci_lower": coherence["ci_lower"],
                "ci_upper": coherence["ci_upper"],
                "pearson_r": coherence["pearson_r"],
                "rmse": coherence["rmse"],
                "mae": coherence["mae"],
                "best_lag": int(lag_results["best_lag"]),
                "k_at_best_lag": lag_results["k_at_best_lag"]
            },
            seed=SEED,
            extra_metadata={
                "example": "05_neuroscience_eeg_analysis",
                "application": "consciousness_monitoring",
                "data_type": "synthetic_eeg"
            }
        )

        all_results.append({
            "c_level": c_level,
            "eeg_data": eeg_data,
            "coherence": coherence,
            "lag_results": lag_results
        })

    # Visualize one example (highest consciousness level)
    logger.info("\n📊 Generating visualization...")
    fig = visualize_eeg_analysis(
        all_results[0]["eeg_data"],
        all_results[0]["coherence"],
        all_results[0]["lag_results"]
    )

    # Save figure
    output_path = Path("outputs/example_05_eeg_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Visualization saved: {output_path}")

    plt.show()

    # Summary across all consciousness levels
    logger.info("\n" + "=" * 80)
    logger.info("📈 SUMMARY ACROSS CONSCIOUSNESS LEVELS")
    logger.info("=" * 80)

    print("\n| C-Level | K-Index | 95% CI | RMSE | Best Lag (ms) | Quality |")
    print("|---------|---------|--------|------|---------------|---------|")

    for result in all_results:
        c = result["coherence"]
        lag_ms = result["lag_results"]["best_lag"] / result["eeg_data"]["sampling_rate"] * 1000
        quality = "Excellent" if c["k_index"] > 0.8 else "Good" if c["k_index"] > 0.6 else "Moderate"
        print(f"| {result['c_level']:.1f}     | {c['k_index']:.4f}  | [{c['ci_lower']:.3f}, {c['ci_upper']:.3f}] | {c['rmse']:.4f} | {lag_ms:>6.1f}        | {quality:<9} |")

    logger.info(f"\n✅ K-Codex entries saved: {kcodex_path}")
    logger.info(f"✅ Git SHA: {infer_git_sha()}")
    logger.info("\n🎓 Example complete! See outputs/example_05_eeg_analysis.png")


if __name__ == "__main__":
    main()
