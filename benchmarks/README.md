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
