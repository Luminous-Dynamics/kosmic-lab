#!/usr/bin/env python3
"""
Kosmic Lab Quick Start - Get started in 30 seconds!

This script demonstrates the core K-Index functionality with a minimal example.

Usage:
    python quick_start.py

    or

    poetry run python quick_start.py

Author: Kosmic Lab Team
Runtime: ~30 seconds
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Core imports
from fre.metrics.k_index import k_index, bootstrap_k_ci
from core.kcodex import KCodexWriter

# Optional: matplotlib for quick visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def main():
    """Run quick start demo."""

    print("\n" + "=" * 70)
    print("🌊 KOSMIC LAB - QUICK START")
    print("=" * 70)
    print("\nWelcome! Let's compute a K-Index in 30 seconds...\n")

    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating synthetic coherent data...")
    np.random.seed(42)
    n_samples = 1000

    # Create coherent observation-action pairs
    true_values = np.random.randn(n_samples)
    observations = true_values + np.random.randn(n_samples) * 0.3  # Add noise
    actions = true_values + np.random.randn(n_samples) * 0.3      # Correlated

    print(f"   ✓ Created {n_samples} observation-action pairs")

    # Step 2: Compute K-Index
    print("\n🔬 Step 2: Computing K-Index...")
    k = k_index(observations, actions)
    print(f"   ✓ K-Index: {k:.3f}")

    # Step 3: Compute confidence interval (quick version)
    print("\n📈 Step 3: Computing 95% confidence interval...")
    print("   (Using 100 bootstrap iterations for speed)")
    k_est, ci_low, ci_high = bootstrap_k_ci(
        observations,
        actions,
        n_bootstrap=100,  # Quick version
        confidence_level=0.95,
        seed=42
    )
    print(f"   ✓ 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # Step 4: Interpret results
    print("\n💡 Step 4: Interpreting results...")
    if k >= 1.0:
        interpretation = "HIGH COHERENCE - Strong observation-action coupling! 🎉"
        color = "green"
    elif k >= 0.5:
        interpretation = "MODERATE COHERENCE - Decent coupling"
        color = "yellow"
    else:
        interpretation = "LOW COHERENCE - Weak or no coupling"
        color = "red"

    print(f"   ✓ {interpretation}")

    # Step 5: Create K-Codex log
    print("\n📝 Step 5: Creating K-Codex experimental record...")
    kcodex = KCodexWriter("logs/quick_start_kcodex.json")
    kcodex.log_experiment(
        experiment_name="quick_start_demo",
        params={
            "n_samples": n_samples,
            "noise_level": 0.3,
            "bootstrap_iterations": 100,
        },
        metrics={
            "k_index": k,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "high_coherence": k >= 1.0,
        },
        seed=42
    )
    print(f"   ✓ K-Codex saved: logs/quick_start_kcodex.json")

    # Step 6: Quick visualization (if matplotlib available)
    if HAS_MATPLOTLIB:
        print("\n📊 Step 6: Creating visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax1.scatter(observations, actions, alpha=0.5, s=20)
        ax1.plot([observations.min(), observations.max()],
                [observations.min(), observations.max()],
                'r--', alpha=0.5, label='Perfect correlation')
        ax1.set_xlabel('Observations')
        ax1.set_ylabel('Actions')
        ax1.set_title(f'Observation-Action Coherence\n(K-Index = {k:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # K-Index with CI
        ax2.bar(['K-Index'], [k], alpha=0.7, color='blue')
        ax2.errorbar(['K-Index'], [k],
                    yerr=[[k - ci_low], [ci_high - k]],
                    fmt='none', color='black', capsize=10, linewidth=2)
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Corridor threshold')
        ax2.set_ylabel('K-Index')
        ax2.set_title('K-Index with 95% Confidence Interval')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, max(2.0, ci_high * 1.2)])

        plt.tight_layout()

        # Save visualization
        output_path = Path("outputs/quick_start_demo.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved: {output_path}")
        plt.close()
    else:
        print("\n📊 Step 6: Visualization skipped (matplotlib not installed)")
        print("   Install with: poetry add matplotlib")

    # Summary
    print("\n" + "=" * 70)
    print("✅ QUICK START COMPLETE!")
    print("=" * 70)

    print(f"\n📊 RESULTS:")
    print(f"   K-Index: {k:.3f}")
    print(f"   95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"   Status: {interpretation}")

    print(f"\n📂 OUTPUTS:")
    print(f"   • K-Codex log: logs/quick_start_kcodex.json")
    if HAS_MATPLOTLIB:
        print(f"   • Visualization: outputs/quick_start_demo.png")

    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. View K-Codex: cat logs/quick_start_kcodex.json")
    if HAS_MATPLOTLIB:
        print(f"   2. View visualization: open outputs/quick_start_demo.png")
    print(f"   3. Run examples: poetry run python examples/01_hello_kosmic.py")
    print(f"   4. Explore: make help")
    print(f"   5. Read docs: cat README.md")

    print(f"\n💡 WHAT IS K-INDEX?")
    print(f"   K-Index measures the coherence between observations and actions.")
    print(f"   Higher values indicate stronger coupling and better prediction.")
    print(f"   Values ≥ 1.0 indicate 'corridor coherence' - strong alignment!")

    print(f"\n📚 LEARN MORE:")
    print(f"   • Architecture: cat ARCHITECTURE.md")
    print(f"   • Examples: ls examples/")
    print(f"   • FAQ: cat FAQ.md")
    print(f"   • Performance: cat docs/PERFORMANCE_GUIDE.md")

    print("\n🌊 Thank you for trying Kosmic Lab!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
