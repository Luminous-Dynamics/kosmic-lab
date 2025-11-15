# Kosmic Lab API Quick Reference

**Version**: 1.1.0
**Last Updated**: 2025-11-15

Quick reference for the most commonly used functions and classes in Kosmic Lab.

---

## Table of Contents

1. [Core Utilities](#core-utilities)
2. [K-Index Metrics](#k-index-metrics)
3. [K-Codex System](#k-codex-system)
4. [Logging](#logging)
5. [Bioelectric Simulation](#bioelectric-simulation)
6. [FRE Simulation](#fre-simulation)
7. [Common Patterns](#common-patterns)

---

## Core Utilities

### `core.utils`

#### Bootstrap Confidence Interval

```python
from core.utils import bootstrap_confidence_interval
import numpy as np

data = np.array([1, 2, 3, 4, 5])

# Basic usage (mean)
ci_lower, ci_upper = bootstrap_confidence_interval(
    data,
    statistic=np.mean,
    n_bootstrap=1000,
    confidence_level=0.95
)

# Custom statistic
ci_lower, ci_upper = bootstrap_confidence_interval(
    data,
    statistic=lambda x: np.percentile(x, 75),
    n_bootstrap=1000
)
```

**Parameters**:
- `data`: Input array
- `statistic`: Function to compute (default: `np.mean`)
- `n_bootstrap`: Bootstrap iterations (default: 1000)
- `confidence_level`: CI level (default: 0.95)
- `seed`: Random seed for reproducibility

**Returns**: `(lower_bound, upper_bound)`

#### Git SHA Inference

```python
from core.utils import infer_git_sha

sha = infer_git_sha()
print(f"Current commit: {sha}")  # 40-char SHA or "unknown"
```

#### Safe Divide

```python
from core.utils import safe_divide

result = safe_divide(10, 0, default=0.0)  # Returns 0.0 instead of error
```

---

## K-Index Metrics

### `fre.metrics.k_index`

#### Basic K-Index

```python
from fre.metrics.k_index import k_index
import numpy as np

# Generate data
rng = np.random.default_rng(42)
observed = rng.random(100)
actual = rng.random(100)

# Compute K-Index
k = k_index(observed, actual)
print(f"K-Index: {k:.4f}")  # Range: [0, 1]
```

**Parameters**:
- `observed`: Observed/predicted values (1D array)
- `actual`: Actual/true values (1D array)
- `correlation_method`: Pearson correlation method (default: scipy.stats.pearsonr)

**Returns**: `float` (K-Index value, 0-1 range)

#### Bootstrap K-Index CI

```python
from fre.metrics.k_index import bootstrap_k_ci

k, ci_lower, ci_upper = bootstrap_k_ci(
    observed,
    actual,
    n_bootstrap=1000,
    confidence_level=0.95,
    seed=42
)

print(f"K-Index: {k:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**Returns**: `(k_index, ci_lower, ci_upper)`

### `fre.metrics.k_lag`

#### K-Lag Analysis

```python
from fre.metrics.k_lag import k_lag

results = k_lag(
    observed,
    actual,
    max_lag=20,
    seed=42
)

print(f"Best lag: {results['best_lag']} timesteps")
print(f"K at best lag: {results['k_at_best_lag']:.4f}")
print(f"K at zero lag: {results['k_at_zero_lag']:.4f}")
```

**Parameters**:
- `observed`: Observed time series
- `actual`: Actual time series
- `max_lag`: Maximum lag to test (default: 10)
- `seed`: Random seed

**Returns**: Dictionary with:
- `best_lag`: Optimal lag value
- `k_at_best_lag`: K-Index at best lag
- `k_at_zero_lag`: K-Index at zero lag
- `lags`: Array of all tested lags
- `k_values`: K-Index values for each lag

#### Verify Causal Direction

```python
from fre.metrics.k_lag import verify_causal_direction

is_correct, forward_k, backward_k = verify_causal_direction(
    observed,
    actual,
    max_lag=20
)

if is_correct:
    print("✓ Causal direction confirmed")
else:
    print("⚠ Possible reversed causality")
```

---

## K-Codex System

### `core.kcodex.KCodexWriter`

#### Basic Usage

```python
from core.kcodex import KCodexWriter

# Create writer
kcodex = KCodexWriter("logs/experiment_kcodex.json")

# Log experiment
kcodex.log_experiment(
    experiment_name="my_experiment",
    params={"n_samples": 100, "threshold": 0.5},
    metrics={"k_index": 0.875, "accuracy": 0.92},
    seed=42,
    extra_metadata={"notes": "Baseline run"}
)
```

#### Complete Example

```python
import numpy as np
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index

# Initialize
kcodex = KCodexWriter("logs/my_experiment.json")
rng = np.random.default_rng(42)

# Run experiment
observed = rng.random(1000)
actual = rng.random(1000)
k = k_index(observed, actual)

# Log everything
kcodex.log_experiment(
    experiment_name="k_index_validation",
    params={
        "n_samples": 1000,
        "correlation_method": "pearsonr"
    },
    metrics={
        "k_index": k,
        "mean_observed": float(np.mean(observed)),
        "mean_actual": float(np.mean(actual))
    },
    seed=42,
    estimators={
        "correlation": "scipy.stats.pearsonr"
    },
    extra_metadata={
        "experiment_type": "validation",
        "notes": "Baseline performance test"
    }
)
```

#### K-Codex Fields

Automatically captured:
- `run_id`: UUID4 unique identifier
- `commit`: Git SHA (exact code version)
- `config_hash`: SHA256 of parameters
- `timestamp`: ISO 8601 timestamp
- `environment`: Python version, platform, hostname

User provided:
- `experiment_name`: Descriptive name
- `params`: Configuration parameters (dict)
- `metrics`: Results (dict)
- `seed`: Random seed
- `estimators`: Algorithm details (optional)
- `extra_metadata`: Additional info (optional)

---

## Logging

### `core.logging_config`

#### Setup Logging

```python
from core.logging_config import setup_logging

# Basic setup
setup_logging(level="INFO")

# With file output
setup_logging(
    level="DEBUG",
    log_file="logs/experiment.log",
    colored=True  # Colored console output
)
```

**Parameters**:
- `level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_file`: Optional file path
- `colored`: Use colored output (default: True)

#### Get Logger

```python
from core.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error")
```

#### Log Experiment Progress

```python
import logging
from core.logging_config import setup_logging, get_logger

setup_logging(level="INFO", log_file="logs/experiment.log")
logger = get_logger(__name__)

logger.info("Starting experiment...")
logger.info(f"Parameters: n_samples={n_samples}, seed={seed}")

# Your experiment code
result = run_experiment()

logger.info(f"Results: K-Index={result:.4f}")
logger.info("Experiment complete!")
```

---

## Bioelectric Simulation

### `core.bioelectric.BioelectricGrid`

```python
from core.bioelectric import BioelectricGrid
import numpy as np

# Create grid
grid = BioelectricGrid(
    size=(10, 10),
    diffusion_rate=0.1,
    resting_potential=-70.0,
    seed=42
)

# Set initial conditions
grid.voltage_grid[5, 5] = 30.0  # Spike at center

# Simulate
for timestep in range(100):
    grid.step()

# Get results
final_voltage = grid.voltage_grid
avg_voltage = grid.get_average_voltage()

print(f"Average voltage: {avg_voltage:.2f} mV")
```

**Methods**:
- `step()`: Advance one timestep
- `get_average_voltage()`: Mean voltage across grid
- `apply_stimulation(x, y, intensity)`: Apply stimulus
- `reset()`: Reset to initial state

---

## FRE Simulation

### `fre.universe.UniverseSimulator`

#### Basic Simulation

```python
from fre.universe import UniverseSimulator

# Create simulator
simulator = UniverseSimulator(
    seed=42,
    n_timesteps=1000
)

# Run simulation
results = simulator.run(
    params={
        "consciousness": 0.5,
        "coherence": 0.7,
        "fep_learning_rate": 0.01
    }
)

# Access results
trajectory = results["trajectory"]
final_state = results["final_state"]
metrics = results["metrics"]

print(f"K-Index: {metrics['k_index']:.4f}")
```

#### Multi-Universe Sweep

```python
from fre.universe import UniverseSimulator
import numpy as np

param_sweep = {
    "consciousness": np.linspace(0.1, 0.9, 5),
    "coherence": np.linspace(0.1, 0.9, 5)
}

results = []
for c_val in param_sweep["consciousness"]:
    for coh_val in param_sweep["coherence"]:
        sim = UniverseSimulator(seed=42)
        result = sim.run({
            "consciousness": c_val,
            "coherence": coh_val
        })
        results.append({
            "params": {"consciousness": c_val, "coherence": coh_val},
            "k_index": result["metrics"]["k_index"]
        })

# Find best params
best = max(results, key=lambda x: x["k_index"])
print(f"Best params: {best['params']}")
print(f"Best K-Index: {best['k_index']:.4f}")
```

---

## Common Patterns

### Pattern 1: Complete Experiment with K-Codex

```python
import numpy as np
from core.logging_config import setup_logging, get_logger
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index, bootstrap_k_ci

# Setup
setup_logging(level="INFO", log_file="logs/experiment.log")
logger = get_logger(__name__)
kcodex = KCodexWriter("logs/experiment_kcodex.json")

# Parameters
params = {
    "n_samples": 1000,
    "seed": 42,
    "method": "pearsonr"
}

# Run experiment
logger.info("Starting experiment...")
rng = np.random.default_rng(params["seed"])
observed = rng.random(params["n_samples"])
actual = rng.random(params["n_samples"])

# Compute metrics
k, ci_lower, ci_upper = bootstrap_k_ci(observed, actual, seed=params["seed"])

logger.info(f"K-Index: {k:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

# Log to K-Codex
kcodex.log_experiment(
    experiment_name="complete_example",
    params=params,
    metrics={
        "k_index": k,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    },
    seed=params["seed"]
)

logger.info("Experiment complete! Results logged to K-Codex.")
```

### Pattern 2: Parameter Sweep with Progress

```python
import numpy as np
from tqdm import tqdm
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index

# Setup
kcodex = KCodexWriter("logs/param_sweep.json")

# Parameter grid
n_samples_range = [100, 500, 1000, 5000]
seeds = range(5)  # 5 replicates per config

# Sweep
results = []
total = len(n_samples_range) * len(seeds)

with tqdm(total=total, desc="Parameter sweep") as pbar:
    for n_samples in n_samples_range:
        for seed in seeds:
            # Run experiment
            rng = np.random.default_rng(seed)
            observed = rng.random(n_samples)
            actual = rng.random(n_samples)
            k = k_index(observed, actual)

            # Store result
            results.append({
                "n_samples": n_samples,
                "seed": seed,
                "k_index": k
            })

            # Log to K-Codex
            kcodex.log_experiment(
                experiment_name="param_sweep",
                params={"n_samples": n_samples},
                metrics={"k_index": k},
                seed=seed
            )

            pbar.update(1)

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
summary = df.groupby("n_samples")["k_index"].agg(["mean", "std"])
print(summary)
```

### Pattern 3: Reproducible Analysis

```python
import json
import numpy as np
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index

# Load existing K-Codex
with open("logs/experiment_kcodex.json") as f:
    kcodex_data = json.load(f)

# Extract parameters
params = kcodex_data["params"]
seed = kcodex_data["seed"]

# Reproduce experiment
rng = np.random.default_rng(seed)
observed = rng.random(params["n_samples"])
actual = rng.random(params["n_samples"])
k_reproduced = k_index(observed, actual)

# Compare with original
k_original = kcodex_data["metrics"]["k_index"]

print(f"Original K-Index:    {k_original:.10f}")
print(f"Reproduced K-Index:  {k_reproduced:.10f}")
print(f"Difference:          {abs(k_original - k_reproduced):.2e}")

# Should be bit-for-bit identical!
assert abs(k_original - k_reproduced) < 1e-10, "Not reproducible!"
print("✓ Experiment successfully reproduced!")
```

---

## Type Hints

All functions are fully typed. Example:

```python
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

def bootstrap_confidence_interval(
    data: NDArray[np.float64],
    statistic: Callable[[NDArray], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    ...
```

Use mypy for type checking:
```bash
poetry run mypy your_script.py
```

---

## Error Handling

```python
from fre.metrics.k_index import k_index
import numpy as np

try:
    observed = np.array([1, 2, 3])
    actual = np.array([4, 5])  # Mismatched length!
    k = k_index(observed, actual)
except ValueError as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

Common errors:
- `ValueError`: Mismatched array lengths, invalid parameters
- `TypeError`: Wrong input types
- `AssertionError`: Validation failures

---

## Performance Tips

1. **Use appropriate sample sizes**:
   - Small exploratory: N=100-1000
   - Standard analysis: N=1000-10000
   - High precision: N=10000+

2. **Bootstrap iterations**:
   - Quick tests: n_bootstrap=100
   - Standard: n_bootstrap=1000
   - Publication: n_bootstrap=10000

3. **Vectorize operations**:
   ```python
   # Slow (Python loop)
   result = [k_index(obs[i], act[i]) for i in range(len(obs))]

   # Fast (NumPy vectorization)
   result = np.array([k_index(obs[i], act[i]) for i in range(len(obs))])
   ```

4. **Use seeds for reproducibility**:
   ```python
   # Always specify seed for reproducible results
   rng = np.random.default_rng(42)  # Good!
   rng = np.random.default_rng()    # Not reproducible!
   ```

---

## Resources

- **Full Documentation**: [docs/](../docs/)
- **Examples**: [examples/](../examples/)
- **FAQ**: [FAQ.md](../FAQ.md)
- **Quick Reference**: [QUICK_REFERENCE.md](../QUICK_REFERENCE.md)
- **API Docs** (Sphinx): `make docs-serve`

---

*This API reference covers the most commonly used functions. For complete documentation, see the full API docs or docstrings.*

**Last Updated**: 2025-11-15 | **Version**: 1.1.0
