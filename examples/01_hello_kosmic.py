#!/usr/bin/env python3
"""
Hello Kosmic - Your First Experiment
=====================================

This tutorial demonstrates:
1. Setting up logging
2. Running a simple simulation
3. Computing K-Index
4. Creating a K-Codex
5. Analyzing results

Expected runtime: ~30 seconds
Output: K-Codex JSON + console summary
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.kcodex import KCodexWriter
from core.logging_config import get_logger, setup_logging
from core.utils import infer_git_sha
from fre.metrics.k_index import bootstrap_k_ci, k_index

# Step 1: Setup logging
setup_logging(level="INFO", log_file="logs/hello_kosmic.log")
logger = get_logger(__name__)

logger.info("🌊 Welcome to Kosmic Lab!")

# Step 2: Create synthetic data (normally this comes from simulation)
logger.info("Generating synthetic coherence data...")

np.random.seed(42)  # For reproducibility
timesteps = 100

# Create correlated observation-action pairs (high coherence)
observations = np.random.randn(timesteps)
actions = 0.7 * observations + 0.3 * np.random.randn(timesteps)

# Compute norms
obs_norms = np.abs(observations)
act_norms = np.abs(actions)

# Step 3: Compute K-Index
logger.info("Computing K-Index...")
k_value = k_index(obs_norms, act_norms)
logger.info(f"K-Index: {k_value:.3f}")

# Compute confidence interval
k_estimate, k_lower, k_upper = bootstrap_k_ci(
    obs_norms,
    act_norms,
    n_bootstrap=1000,
    confidence=0.95,
    random_seed=42,
)

logger.info(f"95% CI: [{k_lower:.3f}, {k_upper:.3f}]")

# Step 4: Create K-Codex for perfect reproducibility
logger.info("Creating K-Codex (experimental record)...")

codex_writer = KCodexWriter(Path("schemas/k_codex.json"))

codex = codex_writer.build_record(
    experiment="hello_kosmic_tutorial",
    params={
        "correlation": 0.7,
        "noise": 0.3,
        "timesteps": timesteps,
    },
    estimators={
        "k_index_method": "pearson",
        "bootstrap_samples": 1000,
        "confidence_level": 0.95,
    },
    metrics={
        "K": k_value,
        "K_lower": k_lower,
        "K_upper": k_upper,
        "in_corridor": k_value >= 1.0,
    },
    seed=42,
    ci={"lower": k_lower, "upper": k_upper},
)

# Save K-Codex
output_path = codex_writer.write(codex, Path("logs/hello_kosmic"))
logger.info(f"✅ K-Codex saved: {output_path}")

# Step 5: Analysis & Interpretation
logger.info("\n" + "=" * 60)
logger.info("RESULTS SUMMARY")
logger.info("=" * 60)
logger.info(f"Experiment: hello_kosmic_tutorial")
logger.info(f"Git SHA: {infer_git_sha()}")
logger.info(f"Seed: 42")
logger.info("")
logger.info(f"K-Index: {k_value:.3f} (95% CI: [{k_lower:.3f}, {k_upper:.3f}])")
logger.info(f"In Corridor: {'✅ YES' if k_value >= 1.0 else '❌ NO'} (threshold: 1.0)")
logger.info("")

# Interpret results
if k_value > 1.5:
    logger.info("🎉 HIGH COHERENCE: Strong observation-action coupling!")
elif k_value > 1.0:
    logger.info("✅ MODERATE COHERENCE: System shows coupling")
else:
    logger.info("⚠️  LOW COHERENCE: Weak coupling detected")

logger.info("\n" + "=" * 60)
logger.info("NEXT STEPS")
logger.info("=" * 60)
logger.info("1. View K-Codex: cat logs/hello_kosmic/*.json | jq")
logger.info("2. Try different parameters: change correlation, noise, timesteps")
logger.info("3. Run real simulation: make fre-run")
logger.info("4. Explore dashboard: make dashboard")
logger.info("")
logger.info("📚 Learn more: cat QUICKSTART.md")
logger.info("🏗️  Architecture: cat ARCHITECTURE.md")
logger.info("")
logger.info("🌊 Thank you for using Kosmic Lab!")

if __name__ == "__main__":
    pass  # All code runs at module level for clarity
