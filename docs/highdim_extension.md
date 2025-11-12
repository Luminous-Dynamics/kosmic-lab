# High-Dimensional Corridor Analysis (5D/6D)
## Design Document v1.0

**Project:** Beyond 3D - Robustness & Generalization Testing  
**Prerequisites:** Successful v0.5 reproduction  
**Timeline:** 1-2 weeks  
**Compute:** ~50-60 GPU-hours

---

## Executive Summary

This document specifies two critical experiments that test whether the 3D Kosmic corridor is:
1. **Robust** to additional "nuisance" parameters (5D robustness sweep)
2. **General** across richer action spaces (6D policy generalization)

These experiments directly address the question: *"Is the corridor a 3-knob artifact, or a low-dimensional manifold embedded in higher-dimensional reality?"*

**Success Outcome:** Corridor persists as a **low-dimensional attractor** (effective dimension ≤ 3) even when embedded in 5-6D space, and SAC policies generalize without retraining.

---

## 1. Motivation: Why >3D?

### 1.1 The Dimensionality Question

Your v0.5 results show a clear corridor in 3D parameter space:
- (energy_gradient, communication_cost, plasticity_rate)

**Skeptical Interpretation:**  
> "This could be a lucky choice of 3 parameters. Add different knobs and the corridor might vanish."

**Strong Claim We Want to Make:**  
> "The corridor is a fundamental attractor that persists across parameter spaces. We've simply identified the most influential dimensions, but the structure is robust to embedding in higher dimensions."

### 1.2 What >3D Tests

| Experiment | Tests | Falsifiable Claim |
|------------|-------|-------------------|
| **5D Robustness** | Corridor persistence | Corridor survives marginalization over nuisance dims |
| **6D Policy Gen** | Controller generalization | SAC learned principle, not 3D memorization |

**If both pass:** Corridor is a **real attractor**, not artifact  
**If either fails:** Need to revise K-index or parameter selection

---

## 2. Experiment A: 5D Robustness Sweep

### 2.1 Design Overview

**Core 3D (from v0.5):**
- energy_gradient: [0.2, 0.8]
- communication_cost: [0.1, 0.5]
- plasticity_rate: [0.05, 0.25]

**Added Nuisance Dimensions:**
- **topology_rewire_prob**: [0.0, 0.3] — Controls small-world network formation
- **stim_dose_jitter**: [0.0, 0.2] — Bioelectric stimulus variability

**Why These?**
- Both held fixed in v0.5 (rewire=0.15, jitter=0.05)
- Theoretically could disrupt corridor (rewiring affects Φ, jitter affects regeneration)
- But not expected to be primary drivers

### 2.2 Sampling Strategy

**Method:** Latin Hypercube + Sobol sequence (quasi-random)

**Budget:**
```
N = 15 × D = 15 × 5 = 75 unique parameter combinations
Seeds per point: 10 (balancing coverage vs. depth)
Total runs: 75 × 10 = 750
```

**Why not more?**
- 5D exhaustive grid at 10 points/dim = 10^5 = 100,000 runs (infeasible)
- LHS gives good coverage with ~10-15× dimensions
- Sobol ensures low-discrepancy (avoids clustering)

### 2.3 Configuration File: `configs/experiment_5d_robustness.yaml`

```yaml
experiment:
  name: "5D_robustness_sweep"
  parent_experiment: "v05_track_a"
  seed_base: 44
  
parameters:
  # Core 3D
  energy_gradient: [0.2, 0.8]
  communication_cost: [0.1, 0.5]
  plasticity_rate: [0.05, 0.25]
  
  # Nuisance dimensions
  topology_rewire_prob: [0.0, 0.3]
  stim_dose_jitter: [0.0, 0.2]
  
sampling:
  method: "sobol"  # Low-discrepancy
  n_points: 75
  n_seeds_per_point: 10
  seed_base: 44  # reuse canonical seed base so reruns align exactly
  
analysis:
  # Compute 3D marginal corridor (averaging over 2 nuisance dims)
  marginalization:
    method: "average"
    nuisance_dims: ["topology_rewire_prob", "stim_dose_jitter"]
  
  # Active subspace analysis
  active_subspace:
    method: "gradient_covariance"
    n_samples_gradient: 100  # Finite-difference gradients
    expected_k_dims: 3
  
  # Sensitivity analysis
  sensitivity:
    methods: ["sobol", "morris"]
    n_bootstrap: 1000
    
targets:
  core_overlap:
    jaccard_min: 0.70  # vs. 3D corridor from v0.5
    description: "Marginalized 5D corridor overlaps with 3D"
  
  core_shift:
    centroid_shift_max: 0.12
    description: "Centroid stable when adding dimensions"
  
  effective_dimension:
    k_max: 3
    variance_explained_min: 0.80
    description: "Top 3 PCs explain ≥80% of K variance"
  
  sobol_indices:
    first_order_sum_min: 0.80
    description: "Main effects dominate (low interactions)"
    
outputs:
  - active_subspace_plot     # 2D projection of 5D space
  - sobol_indices_chart      # Bar chart of parameter importance
  - corridor_overlay_3d      # Original vs. marginalized
  - effective_dimension_table
  - gradient_streamlines     # ∇K in active subspace coords

> **Storage note:** store all raw run artefacts under `outputs/highdim/5d_robustness/`
> (e.g., `metrics.parquet`, `active_subspace.npy`) so subsequent analyses can be
> rehydrated without recomputation.
```

### 2.4 Analysis Pipeline

#### Step 1: Compute K for all 750 runs

```python
import numpy as np
from scipy.stats import qmc
from core.harmonics import HarmonyCalculator

# Generate Sobol samples
sampler = qmc.Sobol(d=5, seed=44)
samples_unit = sampler.random(n=75)

# Scale to parameter bounds
bounds = np.array([
    [0.2, 0.8],   # energy
    [0.1, 0.5],   # comm_cost
    [0.05, 0.25], # plasticity
    [0.0, 0.3],   # rewire
    [0.0, 0.2]    # jitter
])

samples = qmc.scale(samples_unit, bounds[:, 0], bounds[:, 1])

# Run simulations
results = []
for params in samples:
    for seed in range(10):
        k = run_simulation(params, seed)  # Your simulation function
        results.append({
            'energy': params[0],
            'comm_cost': params[1],
            'plasticity': params[2],
            'rewire': params[3],
            'jitter': params[4],
            'seed': seed,
            'k_index': k
        })

df = pd.DataFrame(results)
```

#### Step 2: Marginalize to 3D Corridor

```python
# Average K over nuisance dimensions
df_3d = df.groupby(['energy', 'comm_cost', 'plasticity'])['k_index'].mean().reset_index()

# Identify marginalized corridor
corridor_5d_marginalized = df_3d[df_3d['k_index'] > 1.0]

# Load v0.5 reference corridor
corridor_v05 = pd.read_csv('reference_v05/corridor_params.csv')

# Compute Jaccard overlap
from sklearn.metrics import jaccard_score

# Discretize to grid for comparison
grid_v05 = discretize_to_grid(corridor_v05, n_bins=20)
grid_5d = discretize_to_grid(corridor_5d_marginalized, n_bins=20)

jaccard = jaccard_score(grid_v05.flatten(), grid_5d.flatten())
print(f"Jaccard overlap: {jaccard:.3f}")
```

**Expected Outcome:** Jaccard ≥ 0.70 (corridor core persists)

#### Step 3: Active Subspace Analysis

```python
from sklearn.decomposition import PCA

# Compute gradients ∇K numerically
def compute_gradient(params, epsilon=0.01):
    """Finite-difference gradient of K w.r.t. parameters."""
    grad = np.zeros(5)
    k0 = run_simulation(params, seed=44)
    
    for i in range(5):
        params_plus = params.copy()
        params_plus[i] += epsilon
        k_plus = run_simulation(params_plus, seed=44)
        grad[i] = (k_plus - k0) / epsilon
    
    return grad

# Compute gradient at 100 random points
gradients = []
for _ in range(100):
    params = np.random.uniform(bounds[:, 0], bounds[:, 1])
    grad = compute_gradient(params)
    gradients.append(grad)

gradients = np.array(gradients)

# Covariance of gradients
C = np.cov(gradients.T)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(C)
eigenvalues = eigenvalues[::-1]  # Descending order
eigenvectors = eigenvectors[:, ::-1]

# Active subspace: top k eigenvectors
k_active = 3
W = eigenvectors[:, :k_active]

print(f"Variance explained by top {k_active} dims: {eigenvalues[:k_active].sum() / eigenvalues.sum():.3f}")
```

**Expected Outcome:** Top 3 dimensions explain ≥80% variance

#### Step 4: Sobol Sensitivity Indices

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define problem
problem = {
    'num_vars': 5,
    'names': ['energy', 'comm_cost', 'plasticity', 'rewire', 'jitter'],
    'bounds': bounds.tolist()
}

# Generate Saltelli samples (for Sobol analysis)
n_samples = 1024
param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

# Evaluate K for all samples
Y = np.array([run_simulation(p, seed=44) for p in param_values])

# Compute Sobol indices
Si = sobol.analyze(problem, Y, calc_second_order=False)

print("First-order Sobol indices (main effects):")
for name, s1 in zip(problem['names'], Si['S1']):
    print(f"  {name}: {s1:.3f}")

print(f"\nTotal first-order: {Si['S1'].sum():.3f}")
```

**Expected Outcome:** S1_sum ≥ 0.80 (main effects dominate, low interactions)

### 2.5 Decision Criteria

**GO if ALL of:**
1. Jaccard(5D_marginalized, 3D_v05) ≥ 0.70
2. Centroid shift ≤ 0.12 (Euclidean distance in 3D subspace)
3. Top-3 PCA eigenvalues explain ≥ 80% variance
4. Sobol S1_sum ≥ 0.80

**REVISE if 2-3 pass:**
- Tune K-index weights via PCA on 5D data
- Increase sampling density (N=150)

**NO-GO if ≤1 passes:**
- Corridor is a 3D artifact
- Need to reconsider parameter selection

### 2.6 Expected Runtime

- 75 points × 10 seeds × 30 min/run = **375 hours CPU**
- Parallelized on 8 cores: **~47 hours**
- With GPU acceleration: **~24 hours**

---

## 3. Experiment B: 6D Policy Generalization

### 3.1 Design Overview

**Objective:** Test if SAC policy trained in 3D generalizes to 6D action space

**Training (already done in v0.5 Track B):**
- 3D action space: (energy_gradient, communication_cost, plasticity_rate)
- 100 episodes, converged at episode ~40

**Evaluation (new):**
- 6D action space: add (topology_rewire, stim_intensity, anneal_rate)
- Use **frozen trained policy** (no retraining)
- Test on 120 Sobol-sampled 6D configurations

**Rationale:** If policy generalizes, it learned **principles** (e.g., "increase energy when K drops"), not just memorized 3D lookup table.

### 3.2 Configuration File: `configs/experiment_6d_policy_gen.yaml`

```yaml
experiment:
  name: "6D_policy_generalization"
  parent_experiment: "v05_track_b"
  seed_base: 44
  
training:
  load_checkpoint: "data/results/track_b/trained_sac_policy.pt"
  freeze_policy: true  # No further training
  
action_space:
  # Original 3D
  energy_gradient: [-0.1, 0.1]
  communication_cost: [-0.05, 0.05]
  plasticity_rate: [-0.02, 0.02]
  
  # New 3D (previously fixed)
  topology_rewire: [0.0, 0.3]
  stim_intensity: [0.0, 1.0]
  anneal_rate: [0.0, 0.1]
  
evaluation:
  sampling:
    method: "sobol"
    n_points: 120  # 20 × 6 dimensions
    n_seeds_per_point: 15
  
  shocks: # Same as v0.5 Track B
    - {t: 30, type: "noise", magnitude: 0.3}
    - {t: 60, type: "lesion", pct: 0.2}
    - {t: 90, type: "topology_flip"}
    - {every: 5, type: "sensor_dropout", pct: 0.1}
  
  horizon: 180
  
baselines:
  - "open_loop_6d"  # Random actions in 6D
  - "3d_policy_padded"  # SAC only controls 3D, others zero
  
targets:
  # Median performance
  tat_median_min: 0.75
  recovery_median_max: 0.5  # × open-loop
  auc_median_gain_min: 0.25
  iou_median_min: 0.85
  te_drift_median_max: 0.05
  
  # Robustness (75th percentile)
  pass_rate_75th_min: 0.70  # ≥70% of points meet all criteria
  
outputs:
  - performance_heatmap_6d  # Projected to 2D active subspace
  - pass_fail_map
  - action_utilization      # Which of 6 actions used most?
  - comparison_3d_vs_6d     # Policy effectiveness drop?
```

### 3.3 Evaluation Pipeline

```python
import torch
from controllers.sac import SACAgent

# Load trained 3D policy
agent_3d = SACAgent(state_dim=15, action_dim=3)
agent_3d.load('data/results/track_b/trained_sac_policy.pt')

# Extend to 6D action space
class ExtendedSAC:
    def __init__(self, trained_3d_agent):
        self.agent_3d = trained_3d_agent
        self.extended_dims = 3  # topology, stim, anneal
    
    def select_action(self, state):
        # Get 3D action from trained policy
        action_3d = self.agent_3d.select_action(state)
        
        # Heuristic for new dims (or learn mapping)
        # For now, use simple rules:
        k_current = state[0]  # K is first state element
        
        if k_current < 0.9:  # Low coherence
            topology_rewire = 0.2  # Increase rewiring
            stim_intensity = 0.7   # Boost stimulation
            anneal_rate = 0.05     # Moderate annealing
        else:
            topology_rewire = 0.05
            stim_intensity = 0.3
            anneal_rate = 0.02
        
        action_6d = np.concatenate([
            action_3d,
            [topology_rewire, stim_intensity, anneal_rate]
        ])
        
        return action_6d

agent_6d = ExtendedSAC(agent_3d)

# Generate 6D test configurations
sampler = qmc.Sobol(d=6, seed=44)
test_configs = sampler.random(n=120)
# Scale to bounds...

# Evaluate on each config
results = []
for config in test_configs:
    for seed in range(15):
        metrics = evaluate_policy(
            agent=agent_6d,
            initial_params=config,
            shocks=SHOCKS,
            horizon=180,
            seed=seed
        )
        results.append(metrics)

df_eval = pd.DataFrame(results)
```

### 3.4 Analysis: Generalization Success

```python
# Compute median metrics
medians = df_eval.groupby('config_id').median()

# Check targets
pass_tat = (medians['tat'] >= 0.75).sum() / len(medians)
pass_recovery = (medians['recovery_time'] <= medians['recovery_open_loop'] * 0.5).sum() / len(medians)
pass_auc = (medians['auc'] >= medians['auc_open_loop'] * 1.25).sum() / len(medians)
pass_iou = (medians['iou'] >= 0.85).sum() / len(medians)
pass_te = (medians['te_drift'] <= 0.05).sum() / len(medians)

# Overall pass rate
pass_all = (medians[['tat', 'recovery', 'auc', 'iou', 'te_drift']] >= THRESHOLDS).all(axis=1)
pass_rate = pass_all.mean()

print(f"Pass rate (all criteria): {pass_rate:.2%}")
print(f"Target: ≥70%")

if pass_rate >= 0.70:
    print("✓ GO: Policy generalizes to 6D")
else:
    print("✗ REVISE: Need domain randomization or 6D retraining")
```

### 3.5 Decision Criteria

**GO if:**
- Median TAT ≥ 0.75
- Median recovery ≤ 0.5× open-loop
- Median AUC gain ≥ 25%
- Median IoU ≥ 0.85
- Median TE drift ≤ 5%
- **≥70% of 120 configs pass ALL criteria**

**PARTIAL (Domain Randomization Needed):**
- 50-70% pass rate → Train with 6D action space from scratch

**NO-GO:**
- <50% pass → Policy is 3D-specific, doesn't capture generalizable principles

### 3.6 Expected Runtime

- 120 configs × 15 seeds × 3 min/eval = **90 hours CPU**
- Parallelized: **~12 hours**

---

## 4. Visualization Strategy

### 4.1 Active Subspace Projection (5D → 2D)

```python
import matplotlib.pyplot as plt

# Project 5D data to 2D active subspace
params_5d = df[['energy', 'comm_cost', 'plasticity', 'rewire', 'jitter']].values
k_values = df['k_index'].values

# Active subspace coordinates
W_2d = W[:, :2]  # Top 2 eigenvectors
coords_2d = params_5d @ W_2d

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    coords_2d[:, 0], 
    coords_2d[:, 1],
    c=k_values,
    cmap='RdYlGn',
    s=50,
    alpha=0.6
)
plt.colorbar(scatter, label='K-index')
plt.contour(
    coords_2d[:, 0],
    coords_2d[:, 1],
    k_values,
    levels=[1.0],
    colors='black',
    linewidths=2
)
plt.xlabel('Active Direction 1')
plt.ylabel('Active Direction 2')
plt.title('5D Parameter Space Projected to Active Subspace')
plt.tight_layout()
plt.savefig('active_subspace_5d.png', dpi=300)
```

### 4.2 Sobol Indices Bar Chart

```python
fig, ax = plt.subplots(figsize=(10, 6))

params = ['Energy', 'Comm Cost', 'Plasticity', 'Rewire', 'Jitter']
s1_values = Si['S1']
st_values = Si['ST']  # Total-order indices

x = np.arange(len(params))
width = 0.35

ax.bar(x - width/2, s1_values, width, label='First-order (S1)', color='steelblue')
ax.bar(x + width/2, st_values, width, label='Total-order (ST)', color='coral')

ax.set_ylabel('Sensitivity Index')
ax.set_title('Sobol Sensitivity Analysis (5D Parameter Space)')
ax.set_xticks(x)
ax.set_xticklabels(params)
ax.legend()
ax.axhline(y=0.1, color='gray', linestyle='--', label='Threshold')

plt.tight_layout()
plt.savefig('sobol_indices_5d.png', dpi=300)
```

### 4.3 Policy Generalization Heatmap

```python
# Project 6D results to 2D (via PCA or selected dims)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords_6d = pca.fit_transform(test_configs)

# Create grid for heatmap
grid_resolution = 30
xi = np.linspace(coords_6d[:, 0].min(), coords_6d[:, 0].max(), grid_resolution)
yi = np.linspace(coords_6d[:, 1].min(), coords_6d[:, 1].max(), grid_resolution)
xi, yi = np.meshgrid(xi, yi)

# Interpolate pass/fail
from scipy.interpolate import griddata
zi = griddata(coords_6d, pass_all.astype(int), (xi, yi), method='linear')

plt.figure(figsize=(10, 8))
plt.contourf(xi, yi, zi, levels=1, cmap='RdYlGn', alpha=0.6)
plt.colorbar(label='Pass All Criteria')
plt.scatter(coords_6d[:, 0], coords_6d[:, 1], c=pass_all, cmap='RdYlGn', edgecolors='black', s=30)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('6D Policy Generalization: Pass/Fail Map')
plt.tight_layout()
plt.savefig('policy_gen_6d_heatmap.png', dpi=300)
```

---

## 5. Integration with v0.5 Results

### 5.1 Unified Interpretation

| Finding | Interpretation |
|---------|----------------|
| **5D corridor persists** | Core attractor is robust to nuisance parameters |
| **Effective dim ≤ 3** | System is intrinsically low-dimensional |
| **Sobol S1_sum > 0.8** | Main effects dominate; interactions negligible |
| **6D policy generalizes** | SAC learned principles, not memorization |

**Conclusion:** The Kosmic corridor is a **low-dimensional manifold** in high-dimensional space, confirming it's not a 3-parameter artifact.

### 5.2 Updated Claims for Paper

**Before (v0.5 only):**
> "We identified a corridor in 3D parameter space where coherence emerges."

**After (with 5D/6D extensions):**
> "The corridor is a robust, low-dimensional attractor (effective dimension ≤3) that persists when embedded in 5-6D space. Policies trained on this manifold generalize to unseen dimensions, confirming the corridor captures fundamental organizing principles rather than parameter-specific artifacts."

---

## 6. Implementation Checklist

### Phase 1: Setup (1-2 days)
- [ ] Verify v0.5 reproduction successful
- [ ] Install additional dependencies (SALib for Sobol)
- [ ] Create new config files (5D, 6D)
- [ ] Test on minimal examples (10 points)

### Phase 2: 5D Robustness (3-4 days)
- [ ] Generate 75 Sobol samples
- [ ] Run 750 simulations (parallelized)
- [ ] Compute marginalized corridor
- [ ] Active subspace analysis
- [ ] Sobol sensitivity analysis
- [ ] Generate all figures
- [ ] Write up results

### Phase 3: 6D Policy Gen (2-3 days)
- [ ] Load trained SAC policy
- [ ] Extend to 6D action space
- [ ] Generate 120 test configs
- [ ] Run 1800 evaluations
- [ ] Compute pass rates
- [ ] Visualize performance
- [ ] Compare to baselines

### Phase 4: Integration (1 day)
- [ ] Cross-validate 5D & 6D findings
- [ ] Update v0.5 summary document
- [ ] Draft new sections for paper
- [ ] Preregister findings on OSF

---

## 7. Expected Outcomes & Contingencies

### 7.1 Best-Case Scenario

**All targets met:**
- 5D: Jaccard > 0.70, shift < 0.12, effective dim = 3, S1 > 0.8
- 6D: Pass rate > 70%

**Interpretation:** Corridor is a fundamental attractor. Publish with strong claims.

**Paper Section:** "Robustness Analysis: The Corridor as Low-Dimensional Manifold"

### 7.2 Partial Success

**5D passes, 6D marginal (50-70% pass rate):**
- Corridor is real, but policy needs domain randomization
- Still publishable, but temper generalization claims

**Action:** Retrain SAC with 6D action space (add 2 weeks)

**5D marginal (Jaccard 0.6-0.7), 6D passes:**
- Corridor shifts slightly with new dims, but policies robust
- Indicates corridor boundary is fuzzy but core is stable

**Action:** Report with uncertainty bands, emphasis on policy robustness

### 7.3 Failure

**Both fail (Jaccard < 0.6, pass rate < 50%):**
- 3D corridor was artifact or highly parameter-specific
- Major revision needed

**Action:**
1. Try different nuisance dims (maybe rewire/jitter are too influential)
2. Increase K-index flexibility (add interaction terms)
3. Rethink parameter selection (use Sobol to identify true drivers)

**Timeline Extension:** +4 weeks to revise and re-run

---

## 8. Cost-Benefit Analysis

### 8.1 Computational Cost

| Experiment | Runtime | Cloud Cost |
|------------|---------|------------|
| 5D Robustness | 24 GPU-hrs | ~$200 |
| 6D Policy Gen | 12 GPU-hrs | ~$100 |
| **Total** | **36 GPU-hrs** | **~$300** |

### 8.2 Scientific Value

**High ROI:**
- Addresses key criticism ("just a 3D artifact?")
- Demonstrates low-dimensional structure (highly publishable)
- Validates policy learning (important for AI applications)

**Impact on Paper:**
- Elevates from "interesting finding" to "robust phenomenon"
- Enables stronger claims about universality
- Likely increases citation potential

---

## 9. Preregistration & Reporting

### 9.1 Preregister Hypotheses

**Before running experiments, lock:**

**H5D.1:** Jaccard(5D_marginalized, 3D_v05) ≥ 0.70  
**H5D.2:** Top-3 PCA eigenvalues explain ≥ 80% variance  
**H5D.3:** Sobol first-order sum ≥ 0.80

**H6D.1:** Median TAT ≥ 0.75 across 120 configs  
**H6D.2:** Pass rate (all criteria) ≥ 70%

Upload to OSF with date stamp.

### 9.2 Report All Results

**Even if hypotheses fail:**
- Report exact Jaccard, eigenvalue breakdown, pass rates
- Include all figures (not just successful ones)
- Discuss implications transparently

### 9.3 Deviations

If you deviate from protocol (e.g., increase from 75 to 150 points):
- Document reason
- Report both original and revised results
- Justify as "sensitivity analysis"

---

## 10. Next Steps After Completion

**If successful (GO on both 5D and 6D):**

1. **Paper Integration:**
   - Add "High-Dimensional Robustness" section
   - Update abstract claims
   - Submit to *Physical Review E* or *Nature Communications*

2. **Follow-Up Experiments:**
   - Test 7D, 8D (find true effective dimension limit)
   - Try different nuisance dims (observation noise, agent count)
   - Cross-validate on different K-index weightings

3. **Application Development:**
   - Use active subspace for real-time corridor tracking
   - Deploy 6D SAC in Holochain prototype
   - Build "corridor navigator" web tool

**If partial/failure:**

1. **Revise K-Index:**
   - Reweight based on 5D Sobol analysis
   - Add interaction terms
   - Test on toy systems first

2. **Expand Search:**
   - Try 10D Latin Hypercube with aggressive PCA
   - Use surrogate models (Gaussian Process) to explore cheaply

3. **Pivot Narrative:**
   - Emphasize "corridor core" is stable even if boundary fuzzy
   - Frame as "dimension-dependent phenomenon" (still interesting)

### Reproducibility Checklist

- Keep the `seed_base` value identical across all sweeps, surrogate fittings,
  and policy evaluations. Derive subsequent seeds deterministically (e.g.,
  `seed_base + idx`).
- Persist raw metrics, surrogates, and figures under
  `outputs/highdim/<experiment_name>/` (e.g., `outputs/highdim/5d_robustness/`,
  `outputs/highdim/6d_policy/`, `outputs/highdim/surrogates/`) and store the
  matching config hash to enable exact reruns.

---

**END OF HIGH-DIMENSIONAL EXTENSION DESIGN**

*This document provides complete specifications for testing corridor robustness beyond 3D. These experiments are critical for establishing the corridor as a fundamental attractor rather than a parameter-specific artifact.*

**Document Version:** 1.0  
**Prerequisites:** Successful v0.5 reproduction  
**Status:** Ready for Execution
