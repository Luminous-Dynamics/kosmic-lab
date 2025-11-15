# Kosmic Lab Templates

This directory contains templates to help you get started quickly with common workflows.

## Available Templates

### 1. `experiment_template.py`
**Purpose**: Complete experiment template with K-Index analysis, K-Codex logging, and visualization

**Usage**:
```bash
# Copy the template
cp templates/experiment_template.py my_experiment.py

# Customize the TODOs
# - Update EXPERIMENT_CONFIG
# - Modify data loading in load_or_generate_data()
# - Add custom analyses in analyze_data()
# - Customize visualization in create_visualization()

# Run your experiment
python my_experiment.py
```

**Features**:
- ✅ K-Index computation with bootstrap CI
- ✅ Automatic K-Codex logging for perfect reproducibility
- ✅ Publication-quality visualization
- ✅ Comprehensive logging
- ✅ Modular structure (easy to extend)

**Sections**:
1. Configuration - Set experiment parameters
2. Data Loading - Load or generate data
3. Analysis - K-Index and additional metrics
4. Visualization - Create plots
5. Main - Orchestrate the experiment

---

## Quick Start Guide

### For a Basic K-Index Experiment

```python
# 1. Copy template
cp templates/experiment_template.py my_analysis.py

# 2. Minimal changes needed:
#    - Update EXPERIMENT_CONFIG name and description
#    - Modify load_or_generate_data() to load your data
#    - Run!

python my_analysis.py
```

### For Advanced Experiments

Add K-Lag analysis (for temporal data):
```python
# In analyze_data() function:
from fre.metrics.k_lag import k_lag

lag_results = k_lag(observed, actual, max_lag=50, seed=SEED)
results["best_lag"] = lag_results["best_lag"]
results["k_at_best_lag"] = lag_results["k_at_best_lag"]
```

Add custom metrics:
```python
# In analyze_data() function:
from sklearn.metrics import mean_absolute_error

results["mae"] = mean_absolute_error(actual, observed)
results["max_error"] = np.max(np.abs(observed - actual))
```

Add more visualizations:
```python
# In create_visualization(), add subplot:
ax = axes[2, 0]  # Add a third row
ax.plot(time, observed, label='Observed')
ax.plot(time, actual, label='Actual')
ax.legend()
```

---

## Template Best Practices

### 1. Always Use K-Codex
```python
kcodex = KCodexWriter("logs/my_experiment_kcodex.json")
kcodex.log_experiment(
    experiment_name="my_experiment",
    params={...},  # All parameters
    metrics={...},  # All results
    seed=SEED  # For reproducibility
)
```

### 2. Set a Fixed Seed
```python
SEED = 42
rng = np.random.default_rng(SEED)
# Use rng.random(), rng.normal(), etc. instead of np.random
```

### 3. Log Everything
```python
logger.info("Starting analysis...")
logger.info(f"K-Index: {k:.4f}")
logger.warning("Unusual pattern detected")
```

### 4. Save Visualizations
```python
fig.savefig("outputs/my_figure.png", dpi=300, bbox_inches='tight')
```

---

## Common Modifications

### Loading Real Data

Replace `load_or_generate_data()`:
```python
def load_or_generate_data(config: dict) -> tuple:
    # Load from CSV
    import pandas as pd
    df = pd.read_csv("data/my_data.csv")
    observed = df['observed'].values
    actual = df['actual'].values

    metadata = {
        "data_source": "my_data.csv",
        "n_rows": len(df),
    }

    return observed, actual, metadata
```

### Multiple Conditions

```python
conditions = ["A", "B", "C"]
all_results = {}

for condition in conditions:
    observed, actual, _ = load_data(condition)
    results = analyze_data(observed, actual, config)
    all_results[condition] = results

    # Log each condition separately
    kcodex.log_experiment(
        experiment_name=f"my_experiment_{condition}",
        params={"condition": condition, ...},
        metrics=results,
        seed=SEED
    )
```

### Parameter Sweep

```python
param_values = [0.1, 0.5, 1.0, 2.0]
sweep_results = []

for param in param_values:
    # Generate data with this parameter
    observed, actual, _ = generate_data(param)

    # Analyze
    results = analyze_data(observed, actual, config)
    results["param_value"] = param
    sweep_results.append(results)

    # Log
    kcodex.log_experiment(
        experiment_name=f"sweep_param_{param}",
        params={"param": param, ...},
        metrics=results,
        seed=SEED
    )

# Visualize sweep results
plt.plot(param_values, [r["k_index"] for r in sweep_results])
plt.xlabel("Parameter")
plt.ylabel("K-Index")
```

---

## Getting Help

- **Documentation**: See [API_REFERENCE.md](../API_REFERENCE.md) for function details
- **Examples**: Check [examples/](../examples/) for complete working examples
- **Tutorial**: Run [examples/interactive_tutorial.ipynb](../examples/interactive_tutorial.ipynb)
- **FAQ**: See [FAQ.md](../FAQ.md) for common questions

---

## Contributing Templates

If you create a useful template, consider contributing it back to Kosmic Lab!

1. Ensure it follows the structure of existing templates
2. Add clear TODO comments for customization points
3. Include comprehensive docstrings
4. Test it with real data
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
