# Full K-Index Migration Guide

**Date**: November 19, 2025
**Purpose**: Guide for migrating from Simple K to Full 7-Harmony K-Index

---

## Why Migrate?

| Metric | Correlation with Performance | Status |
|--------|------------------------------|--------|
| Simple K | r = **-0.41** (anti-correlates!) | ❌ DEPRECATED |
| Full K | r = **+0.26** (positively correlates) | ✅ RECOMMENDED |
| H2 (Diversity) | r = **+0.46** (best single predictor) | ✅ RECOMMENDED |

**Simple K measures rigidity, not coherence.** All experiments should migrate to Full K.

---

## Installation

### Step 1: Copy the Implementation

```bash
# From kosmic-lab directory
sudo cp docs/session-notes/2025-11-18/full_k_index_production.py fre/metrics/full_k_index.py
sudo chown $USER:$USER fre/metrics/full_k_index.py
```

### Step 2: Verify Installation

```python
from fre.metrics.full_k_index import compute_full_k_index, compute_h2_diversity
print("✅ Full K-Index installed successfully!")
```

---

## Usage Examples

### Basic Usage

```python
from fre.metrics.full_k_index import compute_full_k_index

# Compute Full K from episode data
k_value, harmonies = compute_full_k_index(
    observations=obs_array,      # np.ndarray shape (n_steps, obs_dim)
    actions=action_list,         # List[int] of discrete actions
    q_values=q_values_history    # List[np.ndarray] of Q-values
)

print(f"Full K-Index: {k_value:.4f}")
print(f"H2 (Diversity): {harmonies['h2_diversity']:.4f}")
```

### Streaming Calculator (During Episodes)

```python
from fre.metrics.full_k_index import FullKIndexCalculator

calculator = FullKIndexCalculator()

for step in range(episode_length):
    obs = env.get_observation()
    q_values = agent.get_q_values(obs)
    action = agent.act(obs)

    calculator.add_step(obs, action, q_values)

    next_obs, reward, done = env.step(action)

# Get final K-Index
k_value, harmonies = calculator.compute()
```

### Quick H2 (Best Single Metric)

```python
from fre.metrics.full_k_index import compute_h2_diversity

# Just compute H2 - the best predictor
h2 = compute_h2_diversity(action_list)
print(f"Action Diversity: {h2:.4f}")
```

---

## Migration from Simple K

### Before (Deprecated)

```python
# OLD CODE - DO NOT USE
from fre.metrics.k_index import k_index

obs_norms = [np.linalg.norm(o) for o in observations]
act_norms = [np.linalg.norm(a) for a in actions]
k = k_index(obs_norms, act_norms)  # ❌ ANTI-CORRELATES WITH PERFORMANCE
```

### After (Recommended)

```python
# NEW CODE - USE THIS
from fre.metrics.full_k_index import compute_full_k_index

k, harmonies = compute_full_k_index(
    observations=observations,
    actions=action_list,
    q_values=q_values_history
)
# ✅ POSITIVELY CORRELATES WITH PERFORMANCE
```

---

## Track Runner Migration

### Update track_g_runner.py

```python
# Replace this:
def compute_k_index(self, observations, actions):
    obs_flat = observations.flatten()
    act_flat = actions.flatten()
    correlation = np.corrcoef(obs_flat, act_flat)[0, 1]
    k_index = 2.0 * np.abs(correlation)
    return float(np.clip(k_index, 0, 2))

# With this:
from fre.metrics.full_k_index import compute_full_k_index

def compute_k_index(self, observations, actions, q_values=None):
    k, harmonies = compute_full_k_index(
        observations=observations,
        actions=list(actions.flatten()),
        q_values=q_values
    )
    return k, harmonies
```

---

## API Reference

### Main Functions

#### `compute_full_k_index(...)`

Main function for computing Full K-Index.

**Parameters:**
- `observations`: np.ndarray - Observation array
- `actions`: List[int] - Discrete actions
- `q_values`: List[np.ndarray] - Q-value vectors
- `hidden_states`: List[np.ndarray] - Hidden states (optional)
- `predictions`: List[float] - Predicted values (optional)
- `actuals`: List[float] - Actual values (optional)
- `k_history`: List[float] - Historical K values (optional)
- `weights`: np.ndarray - Custom harmony weights (optional)

**Returns:** Tuple[float, Dict[str, float]]

#### `compute_h2_diversity(actions)`

Compute H2 (Diversity) - the best single predictor.

**Parameters:**
- `actions`: List[int] - Discrete actions

**Returns:** float in [0, 1]

### Individual Harmonies

- `compute_h1_resonant_coherence(hidden_states)` - Integration measure
- `compute_h2_diversity(actions)` - Action entropy
- `compute_h3_prediction_accuracy(predictions, actuals)` - Prediction quality
- `compute_h4_behavioral_entropy(q_values)` - Q-value entropy
- `compute_h5_mutual_transfer_entropy(q_values)` - Cross-dimension flow
- `compute_h6_flow_symmetry(obs_norms)` - Temporal symmetry
- `compute_h7_growth_rate(k_history)` - Improvement rate

---

## The 7 Harmonies Explained

| Harmony | Name | What It Measures | Range |
|---------|------|------------------|-------|
| H1 | Resonant Coherence | Integration across dimensions | [0, 2] |
| H2 | Diversity | Action distribution entropy | [0, 1] |
| H3 | Prediction Accuracy | Model quality | [0, 1] |
| H4 | Behavioral Entropy | Q-value richness | [0, 1] |
| H5 | Mutual Transfer Entropy | Information flow | [0, 1] |
| H6 | Flow Symmetry | Temporal balance | [0, 1] |
| H7 | Growth Rate | Improvement trend | [-1, 1] |

**H2 (Diversity) is the best single predictor** with r = +0.46 correlation to performance.

---

## Deprecation Warnings

The old Simple K functions now emit deprecation warnings:

```python
from fre.metrics.full_k_index import compute_simple_k_index

# This will show:
# DeprecationWarning: Simple K-Index is DEPRECATED.
# It anti-correlates with performance (r ≈ -0.41).
# Use compute_full_k_index() instead.
```

---

## Best Practices

### 1. Always Use Full K for Evaluation

```python
# Good
k, harmonies = compute_full_k_index(...)

# Avoid
k = 2 * abs(correlation)  # This is Simple K!
```

### 2. Report H2 Separately

H2 is the best predictor - always report it:

```python
k, harmonies = compute_full_k_index(...)
print(f"Full K: {k:.4f}")
print(f"H2 (Diversity): {harmonies['h2_diversity']:.4f}")
```

### 3. Use Streaming Calculator for Long Episodes

```python
calculator = FullKIndexCalculator()
# ... add steps incrementally
k, harmonies = calculator.compute()
```

### 4. Custom Weights for Specific Domains

```python
# Weight H2 more heavily for tasks where diversity matters
weights = np.array([0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1])
k, harmonies = compute_full_k_index(..., weights=weights)
```

---

## Troubleshooting

### ImportError

```python
# If you get: ModuleNotFoundError: No module named 'fre.metrics.full_k_index'
# Make sure you installed the file:
sudo cp docs/session-notes/2025-11-18/full_k_index_production.py fre/metrics/full_k_index.py
```

### Low Full K Values

Full K values typically range 0.3-0.8. If you're getting very low values:
- Check that you're passing enough data (>10 steps)
- Ensure actions are discrete integers
- Verify q_values are numpy arrays

### Comparing Old and New Results

If comparing with old Simple K results, note that **the metrics have opposite meanings**:
- Simple K ↑ = worse performance
- Full K ↑ = better performance

---

## Summary

| Use Case | Function | Why |
|----------|----------|-----|
| General evaluation | `compute_full_k_index()` | Best overall metric |
| Quick single metric | `compute_h2_diversity()` | Best predictor |
| Streaming data | `FullKIndexCalculator` | Memory efficient |
| Legacy code | Migrate to above | Simple K is wrong |

**Remember: Simple K anti-correlates with performance. Always use Full K.**

---

*"Measure correctly, and the truth emerges."*

