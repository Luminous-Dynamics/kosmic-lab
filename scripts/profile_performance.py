#!/usr/bin/env python3
"""
Performance Profiling Utility for Kosmic Lab

This script profiles key functions to identify performance bottlenecks.

Usage:
    poetry run python scripts/profile_performance.py [--output DIR] [--format FORMAT]

Options:
    --output DIR        Output directory for profile results (default: profiling/)
    --format FORMAT     Output format: text, json, html (default: text)
    --function FUNC     Profile specific function (k_index, bootstrap, k_lag, all)
    --samples N         Number of samples to use (default: 1000)
    --help              Show this help message
"""

import argparse
import cProfile
import json
import pstats
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np


def profile_function(
    func: Callable,
    *args: Any,
    iterations: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Profile a function and return timing and statistics.

    Args:
        func: Function to profile
        *args: Positional arguments for function
        iterations: Number of times to run function
        **kwargs: Keyword arguments for function

    Returns:
        Dictionary with profiling results
    """
    # Warmup
    func(*args, **kwargs)

    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        timings.append(end - start)

    profiler.disable()

    # Get statistics
    stats_stream = StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # Top 20 functions

    return {
        "function": func.__name__,
        "iterations": iterations,
        "mean_time": float(np.mean(timings)),
        "std_time": float(np.std(timings)),
        "min_time": float(np.min(timings)),
        "max_time": float(np.max(timings)),
        "median_time": float(np.median(timings)),
        "p95_time": float(np.percentile(timings, 95)),
        "p99_time": float(np.percentile(timings, 99)),
        "total_time": float(np.sum(timings)),
        "profile_stats": stats_stream.getvalue(),
    }


def profile_k_index(n_samples: int = 1000) -> Dict[str, Any]:
    """Profile K-Index computation."""
    from fre.metrics.k_index import k_index

    print(f"Profiling K-Index (N={n_samples})...")

    rng = np.random.default_rng(42)
    observed = np.abs(rng.normal(0, 1, n_samples))
    actual = np.abs(rng.normal(0, 1, n_samples))

    result = profile_function(k_index, observed, actual, iterations=100)
    result["sample_size"] = n_samples
    return result


def profile_bootstrap_ci(n_samples: int = 1000) -> Dict[str, Any]:
    """Profile Bootstrap CI computation."""
    from core.utils import bootstrap_confidence_interval

    print(f"Profiling Bootstrap CI (N={n_samples})...")

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, n_samples)

    result = profile_function(
        bootstrap_confidence_interval,
        data,
        statistic=np.mean,
        n_bootstrap=100,  # Reduced for faster profiling
        confidence_level=0.95,
        iterations=10,
    )
    result["sample_size"] = n_samples
    result["bootstrap_iterations"] = 100
    return result


def profile_k_lag(n_samples: int = 1000) -> Dict[str, Any]:
    """Profile K-Lag analysis."""
    from fre.metrics.k_lag import k_lag

    print(f"Profiling K-Lag (N={n_samples})...")

    rng = np.random.default_rng(42)
    observed = np.abs(rng.normal(0, 1, n_samples))
    actual = np.abs(rng.normal(0, 1, n_samples))

    result = profile_function(
        k_lag, observed, actual, max_lag=10, iterations=10
    )
    result["sample_size"] = n_samples
    result["max_lag"] = 10
    return result


def profile_git_sha_inference() -> Dict[str, Any]:
    """Profile Git SHA inference."""
    from core.utils import infer_git_sha

    print("Profiling Git SHA inference...")

    result = profile_function(infer_git_sha, iterations=100)
    return result


def save_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
    format: str = "text",
) -> None:
    """Save profiling results to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if format == "json":
        # Save as JSON (without profile stats)
        output_file = output_dir / f"profile_{timestamp}.json"
        json_results = []
        for result in results:
            json_result = result.copy()
            json_result.pop("profile_stats", None)  # Remove verbose stats
            json_results.append(json_result)

        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ JSON results saved to: {output_file}")

    elif format == "html":
        # Save as HTML
        output_file = output_dir / f"profile_{timestamp}.html"
        with open(output_file, "w") as f:
            f.write("<html><head><title>Kosmic Lab Profiling Results</title>")
            f.write("<style>")
            f.write("body { font-family: monospace; margin: 20px; }")
            f.write("h1 { color: #333; }")
            f.write("h2 { color: #666; margin-top: 30px; }")
            f.write("table { border-collapse: collapse; margin: 20px 0; }")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            f.write("th { background-color: #f2f2f2; }")
            f.write("pre { background: #f5f5f5; padding: 10px; overflow: auto; }")
            f.write("</style></head><body>")
            f.write("<h1>Kosmic Lab Profiling Results</h1>")
            f.write(f"<p>Generated: {timestamp}</p>")

            for result in results:
                f.write(f"<h2>{result['function']}</h2>")
                f.write("<table>")
                f.write("<tr><th>Metric</th><th>Value</th></tr>")
                for key, value in result.items():
                    if key not in ["function", "profile_stats"]:
                        if isinstance(value, float):
                            f.write(f"<tr><td>{key}</td><td>{value:.6f} s</td></tr>")
                        else:
                            f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
                f.write("</table>")

                if "profile_stats" in result:
                    f.write("<h3>Detailed Profile</h3>")
                    f.write(f"<pre>{result['profile_stats']}</pre>")

            f.write("</body></html>")

        print(f"\n✓ HTML results saved to: {output_file}")

    else:  # text format
        output_file = output_dir / f"profile_{timestamp}.txt"
        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Kosmic Lab Performance Profiling Results\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                f.write(f"\n{result['function']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Iterations:    {result['iterations']}\n")
                f.write(f"Mean time:     {result['mean_time']:.6f} s\n")
                f.write(f"Std dev:       {result['std_time']:.6f} s\n")
                f.write(f"Min time:      {result['min_time']:.6f} s\n")
                f.write(f"Max time:      {result['max_time']:.6f} s\n")
                f.write(f"Median time:   {result['median_time']:.6f} s\n")
                f.write(f"P95 time:      {result['p95_time']:.6f} s\n")
                f.write(f"P99 time:      {result['p99_time']:.6f} s\n")

                if "sample_size" in result:
                    f.write(f"Sample size:   {result['sample_size']}\n")

                f.write(f"\nDetailed Profile:\n")
                f.write(result["profile_stats"])
                f.write("\n")

        print(f"\n✓ Text results saved to: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile Kosmic Lab performance"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiling",
        help="Output directory (default: profiling/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--function",
        type=str,
        choices=["k_index", "bootstrap", "k_lag", "git_sha", "all"],
        default="all",
        help="Function to profile (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    results = []

    print("=" * 80)
    print("Kosmic Lab Performance Profiler v1.1.0")
    print("=" * 80)
    print()

    try:
        if args.function in ["k_index", "all"]:
            results.append(profile_k_index(args.samples))

        if args.function in ["bootstrap", "all"]:
            results.append(profile_bootstrap_ci(args.samples))

        if args.function in ["k_lag", "all"]:
            results.append(profile_k_lag(args.samples))

        if args.function in ["git_sha", "all"]:
            results.append(profile_git_sha_inference())

        # Print summary to console
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        print(f"{'Function':<25} {'Mean Time':<15} {'P95 Time':<15} {'Samples':<10}")
        print("-" * 80)

        for result in results:
            samples = result.get("sample_size", "N/A")
            print(
                f"{result['function']:<25} "
                f"{result['mean_time']:.6f} s     "
                f"{result['p95_time']:.6f} s     "
                f"{samples}"
            )

        # Save results
        save_results(results, output_dir, args.format)

        print("\n✓ Profiling complete!")
        return 0

    except Exception as e:
        print(f"\n✗ Error during profiling: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
