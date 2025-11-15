#!/usr/bin/env python3
"""
Kosmic Lab Benchmark Suite

Comprehensive performance benchmarks for core functions.

Usage:
    # Run all benchmarks
    python benchmarks/suite.py

    # Run specific benchmark
    python benchmarks/suite.py --bench k_index

    # Save results
    python benchmarks/suite.py --output benchmarks/results.json

    # Compare serial vs parallel
    python benchmarks/suite.py --compare-parallel
"""

import numpy as np
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fre.metrics.k_index import k_index, bootstrap_k_ci
from fre.metrics.k_lag import k_lag


class BenchmarkRunner:
    """Run and track performance benchmarks."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.results = {}

    def time_function(
        self,
        func: Callable,
        *args,
        n_iterations: int = 10,
        **kwargs
    ) -> Dict:
        """
        Time a function execution.

        Returns dict with min, max, mean, std execution times.
        """
        times = []

        for _ in range(n_iterations):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            times.append(end - start)

        return {
            "min": np.min(times),
            "max": np.max(times),
            "mean": np.mean(times),
            "std": np.std(times),
            "median": np.median(times),
            "n_iterations": n_iterations
        }

    def benchmark_k_index(self, sizes: List[int] = None) -> Dict:
        """Benchmark K-Index computation across different dataset sizes."""
        if sizes is None:
            sizes = [100, 1000, 10000, 100000]

        print("\n" + "=" * 60)
        print("K-INDEX BENCHMARK")
        print("=" * 60)

        results = {}

        for n in sizes:
            print(f"\nBenchmarking N={n:,}...", end=" ")

            # Generate data
            observed = self.rng.random(n)
            actual = self.rng.random(n)

            # Time K-Index
            timing = self.time_function(k_index, observed, actual, n_iterations=10)

            results[f"n_{n}"] = {
                "n_samples": n,
                **timing
            }

            print(f"{timing['mean']*1000:.2f} ms ± {timing['std']*1000:.2f} ms")

        return results

    def benchmark_bootstrap_serial(self, sizes: List[int] = None) -> Dict:
        """Benchmark serial bootstrap CI."""
        if sizes is None:
            sizes = [100, 1000, 10000]

        print("\n" + "=" * 60)
        print("BOOTSTRAP CI BENCHMARK (Serial)")
        print("=" * 60)

        results = {}
        n_bootstrap = 1000

        for n in sizes:
            print(f"\nBenchmarking N={n:,}, {n_bootstrap} bootstrap...", end=" ")

            # Generate data
            observed = self.rng.random(n)
            actual = self.rng.random(n)

            # Time bootstrap
            timing = self.time_function(
                bootstrap_k_ci,
                observed, actual,
                n_bootstrap=n_bootstrap,
                n_jobs=1,  # Serial
                n_iterations=3  # Fewer iterations for slow operations
            )

            results[f"n_{n}"] = {
                "n_samples": n,
                "n_bootstrap": n_bootstrap,
                **timing
            }

            print(f"{timing['mean']:.2f} s ± {timing['std']:.2f} s")

        return results

    def benchmark_bootstrap_parallel(self, sizes: List[int] = None, n_jobs: int = -1) -> Dict:
        """Benchmark parallel bootstrap CI."""
        if sizes is None:
            sizes = [100, 1000, 10000]

        # Check if parallel processing is available
        try:
            from core.parallel import HAS_JOBLIB
            if not HAS_JOBLIB:
                print("\n⚠️  joblib not available, skipping parallel benchmark")
                return {}
        except ImportError:
            print("\n⚠️  parallel module not available, skipping parallel benchmark")
            return {}

        print("\n" + "=" * 60)
        print(f"BOOTSTRAP CI BENCHMARK (Parallel, n_jobs={n_jobs})")
        print("=" * 60)

        results = {}
        n_bootstrap = 1000

        for n in sizes:
            print(f"\nBenchmarking N={n:,}, {n_bootstrap} bootstrap...", end=" ")

            # Generate data
            observed = self.rng.random(n)
            actual = self.rng.random(n)

            # Time bootstrap
            timing = self.time_function(
                bootstrap_k_ci,
                observed, actual,
                n_bootstrap=n_bootstrap,
                n_jobs=n_jobs,  # Parallel
                n_iterations=3
            )

            results[f"n_{n}"] = {
                "n_samples": n,
                "n_bootstrap": n_bootstrap,
                "n_jobs": n_jobs,
                **timing
            }

            print(f"{timing['mean']:.2f} s ± {timing['std']:.2f} s")

        return results

    def benchmark_k_lag(self, sizes: List[int] = None) -> Dict:
        """Benchmark K-Lag computation."""
        if sizes is None:
            sizes = [500, 1000, 5000]

        print("\n" + "=" * 60)
        print("K-LAG BENCHMARK")
        print("=" * 60)

        results = {}
        max_lag = 50

        for n in sizes:
            print(f"\nBenchmarking N={n:,}, max_lag={max_lag}...", end=" ")

            # Generate time series data
            observed = self.rng.random(n)
            actual = self.rng.random(n)

            # Time K-Lag
            timing = self.time_function(
                k_lag,
                observed, actual,
                max_lag=max_lag,
                n_iterations=5
            )

            results[f"n_{n}"] = {
                "n_samples": n,
                "max_lag": max_lag,
                **timing
            }

            print(f"{timing['mean']*1000:.2f} ms ± {timing['std']*1000:.2f} ms")

        return results

    def run_all(self, compare_parallel: bool = False) -> Dict:
        """Run all benchmarks."""
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "benchmarks": {}
        }

        # K-Index
        all_results["benchmarks"]["k_index"] = self.benchmark_k_index()

        # Bootstrap Serial
        all_results["benchmarks"]["bootstrap_serial"] = self.benchmark_bootstrap_serial()

        # Bootstrap Parallel (if requested)
        if compare_parallel:
            all_results["benchmarks"]["bootstrap_parallel"] = self.benchmark_bootstrap_parallel()

        # K-Lag
        all_results["benchmarks"]["k_lag"] = self.benchmark_k_lag()

        return all_results

    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # K-Index summary
        if "k_index" in results["benchmarks"]:
            k_results = results["benchmarks"]["k_index"]
            print("\nK-Index Performance:")
            for key, data in k_results.items():
                n = data["n_samples"]
                mean_ms = data["mean"] * 1000
                print(f"  N={n:>7,}: {mean_ms:>8.2f} ms")

        # Bootstrap comparison
        if "bootstrap_serial" in results["benchmarks"] and "bootstrap_parallel" in results["benchmarks"]:
            serial = results["benchmarks"]["bootstrap_serial"]
            parallel = results["benchmarks"]["bootstrap_parallel"]

            print("\nBootstrap CI Speedup (Parallel vs Serial):")
            for key in serial.keys():
                if key in parallel:
                    serial_time = serial[key]["mean"]
                    parallel_time = parallel[key]["mean"]
                    speedup = serial_time / parallel_time
                    n = serial[key]["n_samples"]
                    print(f"  N={n:>7,}: {speedup:.1f}x faster ({serial_time:.2f}s → {parallel_time:.2f}s)")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Kosmic Lab benchmarks")
    parser.add_argument("--bench", type=str, help="Specific benchmark to run (k_index, bootstrap, k_lag)")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--compare-parallel", action="store_true", help="Compare serial vs parallel")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create benchmark runner
    runner = BenchmarkRunner(seed=args.seed)

    print("🚀 Kosmic Lab Benchmark Suite")
    print(f"Seed: {args.seed}")

    # Run benchmarks
    if args.bench:
        # Run specific benchmark
        if args.bench == "k_index":
            results = {"benchmarks": {"k_index": runner.benchmark_k_index()}}
        elif args.bench == "bootstrap":
            results = {"benchmarks": {"bootstrap_serial": runner.benchmark_bootstrap_serial()}}
        elif args.bench == "k_lag":
            results = {"benchmarks": {"k_lag": runner.benchmark_k_lag()}}
        else:
            print(f"Unknown benchmark: {args.bench}")
            return
    else:
        # Run all benchmarks
        results = runner.run_all(compare_parallel=args.compare_parallel)

    # Print summary
    runner.print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
