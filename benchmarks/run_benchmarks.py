#!/usr/bin/env python3
"""
Performance Benchmarks for Kosmic Lab

Measures performance of key operations to ensure scalability
and track regressions over time.

Usage:
    poetry run python benchmarks/run_benchmarks.py
    poetry run python benchmarks/run_benchmarks.py --save results.json

Expected runtime: ~2 minutes
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from core.logging_config import get_logger, setup_logging
from core.utils import bootstrap_confidence_interval, infer_git_sha
from fre.metrics.k_index import k_index
from fre.metrics.k_lag import k_lag

setup_logging(level="INFO")
logger = get_logger(__name__)


def benchmark(
    name: str, func: Callable, *args: Any, iterations: int = 100, **kwargs: Any
) -> Dict[str, float]:
    """
    Benchmark a function and return timing statistics.

    Args:
        name: Benchmark name
        func: Function to benchmark
        *args: Positional arguments for function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments for function

    Returns:
        Dictionary with timing statistics
    """
    logger.info(f"Running benchmark: {name} ({iterations} iterations)...")

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        timings.append(end - start)

    timings_array = np.array(timings)
    return {
        "name": name,
        "iterations": iterations,
        "mean_ms": float(np.mean(timings_array) * 1000),
        "std_ms": float(np.std(timings_array) * 1000),
        "min_ms": float(np.min(timings_array) * 1000),
        "max_ms": float(np.max(timings_array) * 1000),
        "median_ms": float(np.median(timings_array) * 1000),
        "p95_ms": float(np.percentile(timings_array, 95) * 1000),
        "p99_ms": float(np.percentile(timings_array, 99) * 1000),
    }


def run_all_benchmarks() -> list[Dict[str, Any]]:
    """Run all benchmarks and return results."""
    results = []

    # Benchmark 1: K-Index computation (small data)
    obs_small = np.random.randn(100)
    act_small = np.random.randn(100)
    results.append(
        benchmark("K-Index (N=100)", k_index, np.abs(obs_small), np.abs(act_small))
    )

    # Benchmark 2: K-Index computation (medium data)
    obs_medium = np.random.randn(1000)
    act_medium = np.random.randn(1000)
    results.append(
        benchmark(
            "K-Index (N=1000)", k_index, np.abs(obs_medium), np.abs(act_medium)
        )
    )

    # Benchmark 3: K-Index computation (large data)
    obs_large = np.random.randn(10000)
    act_large = np.random.randn(10000)
    results.append(
        benchmark("K-Index (N=10000)", k_index, np.abs(obs_large), np.abs(act_large))
    )

    # Benchmark 4: Bootstrap CI (expensive operation)
    results.append(
        benchmark(
            "Bootstrap CI (N=100, B=1000)",
            bootstrap_confidence_interval,
            np.abs(obs_small),
            np.mean,
            n_bootstrap=1000,
            iterations=10,  # Fewer iterations for expensive operation
        )
    )

    # Benchmark 5: K-Lag analysis
    results.append(
        benchmark(
            "K-Lag Analysis (N=100, lag=10)",
            k_lag,
            np.abs(obs_small),
            np.abs(act_small),
            max_lag=10,
        )
    )

    # Benchmark 6: Git SHA inference
    results.append(benchmark("Git SHA Inference", infer_git_sha, iterations=100))

    return results


def print_results(results: list[Dict[str, Any]]) -> None:
    """Print benchmark results in a nice format."""
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BENCHMARK RESULTS")
    logger.info("=" * 80)
    logger.info("")

    for result in results:
        logger.info(f"📊 {result['name']}")
        logger.info(f"   Iterations:  {result['iterations']}")
        logger.info(f"   Mean:        {result['mean_ms']:.3f} ms")
        logger.info(f"   Std Dev:     {result['std_ms']:.3f} ms")
        logger.info(f"   Median:      {result['median_ms']:.3f} ms")
        logger.info(f"   Min:         {result['min_ms']:.3f} ms")
        logger.info(f"   Max:         {result['max_ms']:.3f} ms")
        logger.info(f"   P95:         {result['p95_ms']:.3f} ms")
        logger.info(f"   P99:         {result['p99_ms']:.3f} ms")
        logger.info("")

    logger.info("=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    # Check for performance issues
    warnings = []
    for result in results:
        if result["mean_ms"] > 1000:  # > 1 second
            warnings.append(f"⚠️  {result['name']} is slow: {result['mean_ms']:.1f} ms")

    if warnings:
        logger.warning("\nPerformance Warnings:")
        for warning in warnings:
            logger.warning(warning)
    else:
        logger.info("\n✅ All operations performing well!")

    # Scalability analysis
    k_results = [r for r in results if r["name"].startswith("K-Index")]
    if len(k_results) >= 3:
        logger.info("\nScalability Analysis (K-Index):")
        logger.info(f"  N=100:     {k_results[0]['mean_ms']:.3f} ms")
        logger.info(f"  N=1000:    {k_results[1]['mean_ms']:.3f} ms (10x data)")
        logger.info(f"  N=10000:   {k_results[2]['mean_ms']:.3f} ms (100x data)")

        # Check if it's roughly O(n)
        ratio_10x = k_results[1]["mean_ms"] / k_results[0]["mean_ms"]
        ratio_100x = k_results[2]["mean_ms"] / k_results[0]["mean_ms"]
        logger.info(f"\n  Scaling factor (10x):  {ratio_10x:.2f}x")
        logger.info(f"  Scaling factor (100x): {ratio_100x:.2f}x")

        if ratio_100x < 150:  # Less than 1.5x worse than linear
            logger.info("  ✅ Excellent scaling characteristics!")
        elif ratio_100x < 200:
            logger.info("  ✅ Good scaling characteristics")
        else:
            logger.warning("  ⚠️  Scaling could be improved")


def save_results(results: list[Dict[str, Any]], output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "git_sha": infer_git_sha(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_path}")


def main() -> None:
    """Main benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Kosmic Lab performance benchmarks")
    parser.add_argument("--save", type=Path, help="Save results to JSON file")
    args = parser.parse_args()

    logger.info("🚀 Starting Kosmic Lab Performance Benchmarks")
    logger.info(f"Git SHA: {infer_git_sha()}")
    logger.info("")

    results = run_all_benchmarks()
    print_results(results)

    if args.save:
        save_results(results, args.save)

    logger.info("\n✅ Benchmarks complete!")


if __name__ == "__main__":
    main()
