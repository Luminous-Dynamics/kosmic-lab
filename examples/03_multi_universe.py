#!/usr/bin/env python3
"""
Multi-Universe Simulation Example
==================================

This example demonstrates running multiple universe simulations with:
- Parameter sweeps across different configurations
- Parallel execution for performance
- Comparison of outcomes across parameter space
- Visualization of results

Prerequisites:
    poetry install

Usage:
    poetry run python examples/03_multi_universe.py
    poetry run python examples/03_multi_universe.py --parallel
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.logging_config import get_logger, setup_logging
from core.utils import infer_git_sha
from fre.metrics.k_index import k_index
from fre.simulate import UniverseSimulator, compute_metrics

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def generate_parameter_sweep() -> List[Dict[str, float]]:
    """
    Generate a grid of parameters to explore.

    Returns:
        List of parameter dictionaries for simulation
    """
    # Define parameter ranges
    consciousness_values = [0.3, 0.5, 0.7, 0.9]
    coherence_values = [0.4, 0.6, 0.8]
    fep_values = [0.2, 0.5, 0.8]

    parameter_sets = []

    for consciousness in consciousness_values:
        for coherence in coherence_values:
            for fep in fep_values:
                params = {
                    "consciousness": consciousness,
                    "coherence": coherence,
                    "fep": fep,
                }
                parameter_sets.append(params)

    logger.info(f"Generated {len(parameter_sets)} parameter combinations")
    return parameter_sets


def run_single_simulation(
    params: Dict[str, float], seed: int, timesteps: int = 100
) -> Dict[str, Any]:
    """
    Run a single universe simulation.

    Args:
        params: Parameter dictionary
        seed: Random seed
        timesteps: Number of simulation timesteps

    Returns:
        Dictionary with simulation results
    """
    # Create simulator
    simulator = UniverseSimulator(n_agents=50, seed=seed)

    # Run simulation
    history = []
    for t in range(timesteps):
        # Update based on parameters
        simulator.step(params)

        # Record state
        state = {
            "timestep": t,
            "consciousness": params["consciousness"],
            "coherence": params["coherence"],
            "fep": params["fep"],
        }
        history.append(state)

    # Compute final metrics
    metrics = compute_metrics(params, seed)

    result = {
        "params": params,
        "seed": seed,
        "timesteps": timesteps,
        "final_metrics": metrics,
        "history": history,
    }

    return result


def run_parameter_sweep(
    parameter_sets: List[Dict[str, float]],
    n_seeds: int = 3,
    timesteps: int = 100,
    parallel: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run simulations across parameter sweep.

    Args:
        parameter_sets: List of parameter dictionaries
        n_seeds: Number of random seeds per parameter set
        timesteps: Number of timesteps per simulation
        parallel: Whether to use parallel execution

    Returns:
        List of simulation results
    """
    logger.info("=" * 80)
    logger.info("Running Multi-Universe Parameter Sweep")
    logger.info("=" * 80)
    logger.info(f"Parameter sets: {len(parameter_sets)}")
    logger.info(f"Seeds per set:  {n_seeds}")
    logger.info(f"Timesteps:      {timesteps}")
    logger.info(f"Total runs:     {len(parameter_sets) * n_seeds}")
    logger.info(f"Parallel:       {parallel}")
    logger.info("")

    results = []

    if parallel:
        logger.info("🚀 Running in parallel mode...")
        try:
            from multiprocessing import Pool

            # Create tasks
            tasks = []
            for params in parameter_sets:
                for seed in range(n_seeds):
                    tasks.append((params, seed, timesteps))

            # Run in parallel
            with Pool() as pool:
                results = pool.starmap(run_single_simulation, tasks)

        except ImportError:
            logger.warning("Multiprocessing not available - falling back to serial")
            parallel = False

    if not parallel:
        logger.info("🔄 Running in serial mode...")
        total = len(parameter_sets) * n_seeds
        completed = 0

        for params in parameter_sets:
            for seed in range(n_seeds):
                result = run_single_simulation(params, seed, timesteps)
                results.append(result)

                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    logger.info(f"\n✅ Completed {len(results)} simulations")
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze simulation results to find patterns.

    Args:
        results: List of simulation results

    Returns:
        Analysis summary dictionary
    """
    logger.info("\n" + "=" * 80)
    logger.info("Analyzing Results")
    logger.info("=" * 80)

    # Group by parameter combination
    param_groups = {}
    for result in results:
        param_key = tuple(sorted(result["params"].items()))
        if param_key not in param_groups:
            param_groups[param_key] = []
        param_groups[param_key].append(result)

    # Find best performing parameters
    best_params = None
    best_harmony = -np.inf

    for param_key, group in param_groups.items():
        # Average metrics across seeds
        harmonies = [r["final_metrics"].get("harmony", 0.0) for r in group]
        mean_harmony = np.mean(harmonies)

        params_dict = dict(param_key)

        logger.info(f"\nParams: {params_dict}")
        logger.info(f"  Mean Harmony: {mean_harmony:.4f} ± {np.std(harmonies):.4f}")

        if mean_harmony > best_harmony:
            best_harmony = mean_harmony
            best_params = params_dict

    logger.info("\n" + "=" * 80)
    logger.info("🏆 Best Parameters Found:")
    logger.info("=" * 80)
    for key, value in best_params.items():
        logger.info(f"  {key}: {value:.3f}")
    logger.info(f"  Mean Harmony: {best_harmony:.4f}")

    analysis = {
        "best_params": best_params,
        "best_harmony": best_harmony,
        "total_simulations": len(results),
        "param_combinations": len(param_groups),
    }

    return analysis


def visualize_parameter_space(
    results: List[Dict[str, Any]], output_dir: Path
) -> None:
    """
    Create visualizations of parameter space exploration.

    Args:
        results: Simulation results
        output_dir: Directory to save visualizations
    """
    logger.info("\n" + "=" * 80)
    logger.info("Creating Visualizations")
    logger.info("=" * 80)

    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        logger.warning("Matplotlib not available - skipping visualizations")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    consciousness_vals = []
    coherence_vals = []
    harmony_vals = []

    for result in results:
        consciousness_vals.append(result["params"]["consciousness"])
        coherence_vals.append(result["params"]["coherence"])
        harmony_vals.append(result["final_metrics"].get("harmony", 0.0))

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        consciousness_vals,
        coherence_vals,
        c=harmony_vals,
        s=100,
        alpha=0.6,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Consciousness Parameter", fontsize=12)
    ax.set_ylabel("Coherence Parameter", fontsize=12)
    ax.set_title(
        "Parameter Space Exploration: Harmony Outcomes", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Harmony", fontsize=12)

    plt.tight_layout()

    # Save
    output_path = output_dir / "03_parameter_space.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    plt.close()


def save_results(results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
    """
    Save results to JSON file.

    Args:
        results: Simulation results
        analysis: Analysis summary
    """
    import json

    output_dir = Path("logs/multi_universe")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data (remove history to reduce size)
    compact_results = []
    for r in results:
        compact = {
            "params": r["params"],
            "seed": r["seed"],
            "final_metrics": r["final_metrics"],
        }
        compact_results.append(compact)

    data = {
        "git_sha": infer_git_sha(),
        "analysis": analysis,
        "results": compact_results,
    }

    output_path = output_dir / "multi_universe_results.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_path}")


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run multi-universe parameter sweep simulation"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel execution (faster but uses more CPU)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="Timesteps per simulation"
    )
    parser.add_argument("--seeds", type=int, default=3, help="Random seeds per parameter set")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Multi-Universe Simulation Example")
    logger.info("=" * 80)
    logger.info("")

    # Step 1: Generate parameter sweep
    parameter_sets = generate_parameter_sweep()

    # Step 2: Run simulations
    results = run_parameter_sweep(
        parameter_sets,
        n_seeds=args.seeds,
        timesteps=args.timesteps,
        parallel=args.parallel,
    )

    # Step 3: Analyze results
    analysis = analyze_results(results)

    # Step 4: Visualize
    try:
        visualize_parameter_space(results, Path("logs/examples"))
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # Step 5: Save results
    save_results(results, analysis)

    logger.info("\n" + "=" * 80)
    logger.info("Multi-Universe Analysis Complete!")
    logger.info("=" * 80)
    logger.info("\n📚 Key Findings:")
    logger.info(f"  Explored {analysis['param_combinations']} parameter combinations")
    logger.info(f"  Ran {analysis['total_simulations']} total simulations")
    logger.info(f"  Best harmony: {analysis['best_harmony']:.4f}")
    logger.info("\n🚀 Next Steps:")
    logger.info("  - Refine parameter ranges around best performers")
    logger.info("  - Run longer simulations for stability analysis")
    logger.info("  - Try examples/04_bioelectric_rescue.py for rescue mechanisms")


if __name__ == "__main__":
    main()
