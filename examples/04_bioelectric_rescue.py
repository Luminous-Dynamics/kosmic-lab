#!/usr/bin/env python3
"""
Bioelectric Rescue Mechanism Example
====================================

This example demonstrates the bioelectric rescue mechanism that:
- Detects consciousness collapse via FEP errors
- Applies bioelectric interventions
- Monitors recovery trajectories
- Validates rescue effectiveness

Inspired by Michael Levin's work on bioelectric control of pattern.

Prerequisites:
    poetry install

Usage:
    poetry run python examples/04_bioelectric_rescue.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.bioelectric import (
    BioelectricCircuit,
    apply_bioelectric_rescue,
    detect_consciousness_collapse,
)
from core.kcodex import KCodexWriter
from core.logging_config import get_logger, setup_logging
from fre.metrics.k_index import k_index

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def simulate_consciousness_collapse(
    n_timesteps: int = 200, collapse_start: int = 50, collapse_duration: int = 30
) -> Dict[str, np.ndarray]:
    """
    Simulate a consciousness system that collapses and potentially recovers.

    Args:
        n_timesteps: Total simulation timesteps
        collapse_start: When collapse begins
        collapse_duration: How long collapse lasts

    Returns:
        Dictionary with time series data
    """
    logger.info("🧪 Simulating consciousness collapse scenario...")

    # Initialize healthy state
    rng = np.random.default_rng(42)

    # Bioelectric state (resting membrane potentials in mV)
    vmem = np.zeros(n_timesteps)
    vmem[0] = -70.0  # Healthy resting potential

    # Consciousness coherence measure
    coherence = np.zeros(n_timesteps)
    coherence[0] = 0.8  # High coherence

    # Free Energy Principle error
    fep_error = np.zeros(n_timesteps)
    fep_error[0] = 0.1  # Low error

    # Simulate over time
    for t in range(1, n_timesteps):
        if collapse_start <= t < collapse_start + collapse_duration:
            # Collapse phase: rapid degradation
            vmem[t] = vmem[t - 1] + rng.normal(5.0, 2.0)  # Depolarization
            coherence[t] = max(0.0, coherence[t - 1] - rng.uniform(0.05, 0.1))
            fep_error[t] = min(1.0, fep_error[t - 1] + rng.uniform(0.05, 0.15))
        else:
            # Normal or recovery phase
            vmem[t] = vmem[t - 1] + rng.normal(0.0, 1.0)  # Small noise
            coherence[t] = max(0.0, min(1.0, coherence[t - 1] + rng.normal(0.0, 0.02)))
            fep_error[t] = max(0.0, min(1.0, fep_error[t - 1] + rng.normal(0.0, 0.02)))

    data = {
        "timesteps": np.arange(n_timesteps),
        "vmem": vmem,
        "coherence": coherence,
        "fep_error": fep_error,
        "collapse_start": collapse_start,
        "collapse_end": collapse_start + collapse_duration,
    }

    logger.info(f"  Healthy phase:   t=0 to t={collapse_start}")
    logger.info(
        f"  Collapse phase:  t={collapse_start} to t={collapse_start + collapse_duration}"
    )
    logger.info(f"  Recovery phase:  t={collapse_start + collapse_duration} to t={n_timesteps}")

    return data


def apply_rescue_intervention(
    data: Dict[str, np.ndarray], intervention_start: int
) -> Dict[str, np.ndarray]:
    """
    Apply bioelectric rescue intervention.

    Args:
        data: Original simulation data
        intervention_start: When to start intervention

    Returns:
        Modified data with rescue applied
    """
    logger.info(f"\n⚡ Applying bioelectric rescue at t={intervention_start}")

    # Create bioelectric circuit
    circuit = BioelectricCircuit(n_cells=100, resting_voltage=-70.0)

    rescued_data = data.copy()
    vmem = rescued_data["vmem"].copy()
    coherence = rescued_data["coherence"].copy()
    fep_error = rescued_data["fep_error"].copy()

    intervention_log = []

    rng = np.random.default_rng(43)

    for t in range(intervention_start, len(vmem)):
        # Detect if rescue is needed
        collapse_detected = detect_consciousness_collapse(fep_error[t], coherence[t])

        if collapse_detected:
            # Apply rescue
            correction = apply_bioelectric_rescue(
                current_voltage=vmem[t],
                target_voltage=-70.0,
                fep_error=fep_error[t],
                momentum=0.9,
            )

            # Update state with correction
            vmem[t] = vmem[t] + correction * 0.3  # Partial correction
            coherence[t] = min(1.0, coherence[t] + 0.05)  # Boost coherence
            fep_error[t] = max(0.0, fep_error[t] - 0.1)  # Reduce error

            intervention_log.append(
                {
                    "timestep": t,
                    "correction": correction,
                    "vmem_after": vmem[t],
                    "coherence_after": coherence[t],
                    "fep_error_after": fep_error[t],
                }
            )

        # Add natural dynamics
        if t < len(vmem) - 1:
            vmem[t + 1] = vmem[t] + rng.normal(0.0, 0.5)
            coherence[t + 1] = max(0.0, min(1.0, coherence[t] + rng.normal(0.0, 0.01)))
            fep_error[t + 1] = max(0.0, min(1.0, fep_error[t] + rng.normal(0.0, 0.01)))

    rescued_data["vmem"] = vmem
    rescued_data["coherence"] = coherence
    rescued_data["fep_error"] = fep_error
    rescued_data["intervention_log"] = intervention_log

    logger.info(f"  Applied {len(intervention_log)} rescue interventions")

    return rescued_data


def compare_outcomes(
    baseline_data: Dict[str, np.ndarray], rescued_data: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Compare baseline vs rescued outcomes.

    Args:
        baseline_data: Data without rescue
        rescued_data: Data with rescue applied

    Returns:
        Comparison metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("Comparing Baseline vs Rescued Outcomes")
    logger.info("=" * 80)

    # Final timestep metrics
    final_idx = -1

    baseline_final = {
        "vmem": baseline_data["vmem"][final_idx],
        "coherence": baseline_data["coherence"][final_idx],
        "fep_error": baseline_data["fep_error"][final_idx],
    }

    rescued_final = {
        "vmem": rescued_data["vmem"][final_idx],
        "coherence": rescued_data["coherence"][final_idx],
        "fep_error": rescued_data["fep_error"][final_idx],
    }

    logger.info("\n📊 Final State Comparison:")
    logger.info("\nBaseline (no rescue):")
    for key, value in baseline_final.items():
        logger.info(f"  {key}: {value:.3f}")

    logger.info("\nRescued:")
    for key, value in rescued_final.items():
        logger.info(f"  {key}: {value:.3f}")

    # Compute improvement
    improvements = {
        "coherence_gain": rescued_final["coherence"] - baseline_final["coherence"],
        "fep_error_reduction": baseline_final["fep_error"]
        - rescued_final["fep_error"],
        "vmem_stabilization": abs(rescued_final["vmem"] - (-70.0))
        - abs(baseline_final["vmem"] - (-70.0)),
    }

    logger.info("\n📈 Improvements:")
    logger.info(f"  Coherence gain:     {improvements['coherence_gain']:.3f}")
    logger.info(f"  FEP error reduction: {improvements['fep_error_reduction']:.3f}")
    logger.info(
        f"  Vmem stabilization:  {improvements['vmem_stabilization']:.3f} mV closer to target"
    )

    comparison = {
        "baseline_final": baseline_final,
        "rescued_final": rescued_final,
        "improvements": improvements,
    }

    return comparison


def visualize_rescue(
    baseline_data: Dict[str, np.ndarray],
    rescued_data: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """
    Create visualization of rescue mechanism.

    Args:
        baseline_data: Original data
        rescued_data: Rescued data
        output_dir: Where to save plots
    """
    logger.info("\n📊 Creating visualizations...")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available - skipping visualizations")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    timesteps = baseline_data["timesteps"]
    collapse_start = baseline_data["collapse_start"]
    collapse_end = baseline_data["collapse_end"]

    # Plot 1: Membrane potential
    axes[0].plot(
        timesteps, baseline_data["vmem"], "b-", alpha=0.5, linewidth=2, label="Baseline"
    )
    axes[0].plot(
        timesteps, rescued_data["vmem"], "g-", linewidth=2, label="Rescued"
    )
    axes[0].axhline(-70, color="r", linestyle="--", alpha=0.5, label="Target")
    axes[0].axvspan(collapse_start, collapse_end, alpha=0.2, color="red")
    axes[0].set_ylabel("Vmem (mV)", fontsize=11)
    axes[0].set_title("Bioelectric Rescue: Membrane Potential", fontweight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Coherence
    axes[1].plot(
        timesteps,
        baseline_data["coherence"],
        "b-",
        alpha=0.5,
        linewidth=2,
        label="Baseline",
    )
    axes[1].plot(
        timesteps, rescued_data["coherence"], "g-", linewidth=2, label="Rescued"
    )
    axes[1].axvspan(collapse_start, collapse_end, alpha=0.2, color="red")
    axes[1].set_ylabel("Coherence", fontsize=11)
    axes[1].set_title("Consciousness Coherence", fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: FEP Error
    axes[2].plot(
        timesteps,
        baseline_data["fep_error"],
        "b-",
        alpha=0.5,
        linewidth=2,
        label="Baseline",
    )
    axes[2].plot(
        timesteps, rescued_data["fep_error"], "g-", linewidth=2, label="Rescued"
    )
    axes[2].axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Danger threshold")
    axes[2].axvspan(collapse_start, collapse_end, alpha=0.2, color="red", label="Collapse period")
    axes[2].set_xlabel("Timestep", fontsize=11)
    axes[2].set_ylabel("FEP Error", fontsize=11)
    axes[2].set_title("Free Energy Principle Error", fontweight="bold")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "04_bioelectric_rescue.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    plt.close()


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Bioelectric Rescue Mechanism Example")
    logger.info("=" * 80)
    logger.info("")

    # Step 1: Simulate baseline (no rescue)
    logger.info("Step 1: Simulating baseline collapse scenario")
    baseline_data = simulate_consciousness_collapse(
        n_timesteps=200, collapse_start=50, collapse_duration=30
    )

    # Step 2: Apply rescue intervention
    logger.info("\nStep 2: Applying bioelectric rescue")
    rescued_data = apply_rescue_intervention(
        baseline_data.copy(), intervention_start=60
    )

    # Step 3: Compare outcomes
    logger.info("\nStep 3: Comparing outcomes")
    comparison = compare_outcomes(baseline_data, rescued_data)

    # Step 4: Visualize
    logger.info("\nStep 4: Creating visualizations")
    try:
        visualize_rescue(baseline_data, rescued_data, Path("logs/examples"))
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # Step 5: Generate K-Codex
    logger.info("\nStep 5: Generating K-Codex for reproducibility")
    output_dir = Path("logs/bioelectric_rescue")
    output_dir.mkdir(parents=True, exist_ok=True)

    kcodex = KCodexWriter(output_dir / "rescue_experiment.json")
    kcodex.log_experiment(
        experiment_name="bioelectric_rescue_demo",
        params={"intervention_start": 60, "n_timesteps": 200},
        metrics=comparison["improvements"],
        seed=42,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Bioelectric Rescue Example Complete!")
    logger.info("=" * 80)
    logger.info("\n📚 Key Findings:")

    if comparison["improvements"]["coherence_gain"] > 0.1:
        logger.info("  ✅ Rescue significantly improved coherence")
    else:
        logger.info("  ⚠️  Rescue had limited effect on coherence")

    if comparison["improvements"]["fep_error_reduction"] > 0.1:
        logger.info("  ✅ Rescue reduced FEP error effectively")
    else:
        logger.info("  ⚠️  FEP error remained elevated")

    logger.info("\n🧬 Bioelectric Principles:")
    logger.info("  - Membrane voltage reflects system state")
    logger.info("  - Early intervention is most effective")
    logger.info("  - Multiple interventions may be needed")
    logger.info("  - Inspired by Michael Levin's morphogenetic fields")
    logger.info("\n🚀 Next Steps:")
    logger.info("  - Experiment with intervention timing")
    logger.info("  - Try different rescue strategies")
    logger.info("  - Explore scripts/kosmic_dashboard.py for real-time monitoring")


if __name__ == "__main__":
    main()
