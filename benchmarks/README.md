# Performance Benchmarks

This directory contains performance benchmarks for Kosmic Lab to ensure scalability and track performance over time.

## Quick Start

```bash
# Run all benchmarks
poetry run python benchmarks/run_benchmarks.py

# Save results to file
poetry run python benchmarks/run_benchmarks.py --save results/benchmark_$(date +%Y%m%d).json
```

## Benchmarks Included

1. **K-Index Computation**
   - Small data (N=100)
   - Medium data (N=1000)
   - Large data (N=10,000)
   - Tests scalability

2. **Bootstrap Confidence Intervals**
   - N=100, B=1000 bootstrap samples
   - Tests statistical operations

3. **K-Lag Analysis**
   - Time series lag analysis
   - Tests correlation computations

4. **Git SHA Inference**
   - Reproducibility infrastructure
   - Tests subprocess operations

## Expected Performance

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| K-Index (N=100) | < 1 ms | Should be very fast |
| K-Index (N=1000) | < 10 ms | Linear scaling |
| K-Index (N=10000) | < 100 ms | Should scale well |
| Bootstrap CI | 100-500 ms | Computationally intensive |
| K-Lag Analysis | < 5 ms | Multiple correlations |
| Git SHA | < 10 ms | Subprocess call |

## Tracking Performance

Save benchmarks regularly and compare:

```bash
# Save baseline
poetry run python benchmarks/run_benchmarks.py --save results/baseline.json

# After changes
poetry run python benchmarks/run_benchmarks.py --save results/after_change.json

# Compare (manual for now)
diff results/baseline.json results/after_change.json
```

## Performance Goals

- **Scalability**: O(n) or better for core operations
- **Responsiveness**: Sub-second for interactive operations
- **Predictability**: Low variance in timings
- **No Regressions**: Track over time

## Adding New Benchmarks

```python
def benchmark_new_operation():
    """Add to run_all_benchmarks()"""
    results.append(
        benchmark(
            "New Operation Description",
            your_function,
            arg1,
            arg2,
            iterations=100
        )
    )
```

---

**Last Updated**: 2025-11-14
**Maintained By**: Kosmic Lab Team
