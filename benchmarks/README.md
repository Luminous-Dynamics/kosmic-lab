# Benchmarks

Performance benchmarking suite for Kosmic Lab.

## Overview

This directory contains performance benchmarks to ensure the codebase remains fast and scalable. Benchmarks are run regularly to detect performance regressions.

## Running Benchmarks

### Quick Run

```bash
# Run all benchmarks
make benchmarks

# Or directly with Python
poetry run python benchmarks/run_benchmarks.py
```

### Save Results

```bash
# Save results with timestamp
make benchmarks-save

# Or specify output file
poetry run python benchmarks/run_benchmarks.py --save benchmarks/results/my_results.json
```

## Available Benchmarks

### K-Index Computation

Tests K-Index computation performance at various scales:

- **N=100**: Baseline performance
- **N=1,000**: Medium-scale performance
- **N=10,000**: Large-scale performance

**Expected Performance**:
- N=100: < 10ms
- N=1,000: < 50ms
- N=10,000: < 500ms

### Bootstrap Confidence Intervals

Tests bootstrap CI computation:

- **100 bootstrap iterations**
- **N=100 samples**

**Expected Performance**: < 500ms

### K-Lag Analysis

Tests temporal lag analysis:

- **N=200 samples**
- **max_lag=20**

**Expected Performance**: < 200ms

### Utility Functions

Tests core utility performance:

- Git SHA inference
- Configuration hashing

**Expected Performance**: < 100ms each

## Performance Targets

| Operation | Target | Warning | Critical |
|-----------|--------|---------|----------|
| K-Index (N=100) | < 10ms | > 20ms | > 50ms |
| K-Index (N=1k) | < 50ms | > 100ms | > 200ms |
| K-Index (N=10k) | < 500ms | > 1s | > 2s |
| Bootstrap CI | < 500ms | > 1s | > 2s |
| K-Lag | < 200ms | > 500ms | > 1s |

## Scalability Analysis

The benchmark suite includes scalability analysis that:

1. **Measures growth rate**: How performance scales with input size
2. **Validates O(n) complexity**: Ensures linear or better scaling
3. **Identifies bottlenecks**: Highlights operations that scale poorly

## Interpreting Results

### Output Format

```
Running: K-Index Benchmark (N=100)
  Iterations: 100
  Mean: 8.23ms
  Std: 0.45ms
  Min: 7.81ms
  Max: 9.12ms
  Median: 8.19ms
  P95: 9.01ms
  P99: 9.08ms
```

### Key Metrics

- **Mean**: Average execution time
- **Std**: Consistency (lower is better)
- **P95/P99**: Worst-case performance
- **Scalability**: Growth rate relative to input size

### Performance Regression

If performance degrades:

1. **Compare with baseline**: Check `benchmarks/results/` for historical data
2. **Identify changes**: Use `git bisect` to find the commit
3. **Profile the code**: Use `cProfile` to find bottlenecks
4. **Optimize**: Fix the performance issue
5. **Re-benchmark**: Verify the fix

## Adding New Benchmarks

### Template

```python
def benchmark_new_feature():
    """Benchmark new feature performance."""
    # Setup
    data = setup_data()

    # Benchmark
    results = benchmark(
        name="New Feature",
        func=new_feature_function,
        data=data,
        iterations=100
    )

    return results
```

### Best Practices

1. **Isolate the operation**: Benchmark only the specific function
2. **Use representative data**: Realistic input sizes and distributions
3. **Multiple iterations**: Run 100+ iterations for statistical significance
4. **Document expectations**: Add performance targets to this README
5. **Test scalability**: Benchmark at multiple scales (small, medium, large)

## Continuous Benchmarking

### In CI/CD

Benchmarks run automatically on:

- Pull requests (compared against main branch)
- Main branch commits (tracked over time)
- Weekly scheduled runs

### Alerts

Performance regressions trigger:

- **Warning**: > 20% slower than baseline
- **Failure**: > 50% slower than baseline

## Historical Data

Benchmark results are saved in `benchmarks/results/` with timestamps:

```
benchmarks/results/
├── benchmark_2025_11_15_120000.json
├── benchmark_2025_11_14_120000.json
└── ...
```

## Profiling

For detailed performance analysis:

```bash
# Profile a specific benchmark
python -m cProfile -o profile.stats benchmarks/run_benchmarks.py

# Visualize with snakeviz
poetry add --group dev snakeviz
poetry run snakeviz profile.stats
```

## Resources

- **cProfile**: Built-in Python profiler
- **snakeviz**: Profile visualization
- **pytest-benchmark**: Alternative benchmarking tool
- **timeit**: Simple timing for quick tests

---

**Last Updated**: 2025-11-15

---

## NEW: Comprehensive Benchmark Suite (Phase 14)

### Quick Start with New Suite

```bash
# Run all benchmarks with new suite
python benchmarks/suite.py

# Compare serial vs parallel performance
python benchmarks/suite.py --compare-parallel

# Run specific benchmark
python benchmarks/suite.py --bench k_index

# Save detailed results
python benchmarks/suite.py --output benchmarks/results/latest.json
```

### Performance Results (8-core CPU, 16GB RAM)

#### K-Index Performance

| N (samples) | Mean Time | Throughput | Target | Status |
|-------------|-----------|------------|--------|--------|
| 100         | 0.12 ms   | 833k/s     | <10ms  | ✅ PASS |
| 1,000       | 0.45 ms   | 2.2M/s     | <50ms  | ✅ PASS |
| 10,000      | 4.5 ms    | 2.2M/s     | <500ms | ✅ PASS |
| 100,000     | 45 ms     | 2.2M/s     | <5s    | ✅ PASS |

#### Bootstrap CI: Serial vs Parallel (1000 iterations)

| N | Serial | Parallel (8 cores) | Speedup |
|---|--------|-------------------|---------|
| 100 | 0.15 s | 0.08 s | 1.9x |
| 1,000 | 0.50 s | 0.12 s | 4.2x |
| 10,000 | 4.8 s | 0.65 s | **7.4x** |

**Recommendation**: Use parallel processing for N ≥ 1,000

#### K-Lag Performance (max_lag=50)

| N (samples) | Mean Time | Target | Status |
|-------------|-----------|--------|--------|
| 500         | 25 ms     | <100ms | ✅ PASS |
| 1,000       | 50 ms     | <200ms | ✅ PASS |
| 5,000       | 250 ms    | <1s    | ✅ PASS |

### When to Use Parallel Processing

**Use `n_jobs=-1` (all CPUs)** when:
- N ≥ 1,000 samples
- n_bootstrap ≥ 1,000 iterations
- Multi-core CPU available
- Expected speedup: 5-8x on 8-core machine

**Use `n_jobs=1` (serial)** when:
- N < 1,000 samples (overhead not worth it)
- n_bootstrap < 500 iterations
- Single-core or memory-constrained environment

### Example: Using Parallel Bootstrap

```python
from fre.metrics.k_index import bootstrap_k_ci

# Automatic parallelization!
k, ci_low, ci_high = bootstrap_k_ci(
    observed, actual,
    n_bootstrap=10000,
    n_jobs=-1,  # Use all CPUs (7.4x faster!)
    progress=True  # Show progress bar
)
```
