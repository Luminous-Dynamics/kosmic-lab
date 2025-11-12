# 📖 Kosmic Lab Glossary

**Comprehensive reference for all key concepts, metrics, and terminology**

---

## Core Concepts

### K-Index (Coherence Index)

**Definition**: Composite metric quantifying the degree of integrated information and reciprocal harmony in a system.

**Formula**:
```
K = w₁·Φ + w₂·TE_symmetry + w₃·Reciprocity + w₄·Diversity + w₅·Play
```

**Range**: [0, 2.5] in normalized form
- **Typical weights**: w₁=0.35, w₂=0.20, w₃=0.15, w₄=0.15, w₅=0.15

**Interpretation**:
- **K < 0.5**: Fragmented system with minimal integration
- **0.5 ≤ K < 1.0**: Emerging coherence, moderate integration
- **1.0 ≤ K < 1.5**: High coherence, well-integrated
- **K ≥ 1.5**: Exceptional coherence, highly resilient

**Related terms**: Integrated Information, Φ (Phi), Global Workspace

### K-Codex

**Definition**: Standardized immutable record ensuring complete experimental reproducibility and eternal verifiability. Formerly known as "K-Passport" (local JSON variant).

**Etymology**: From Latin *codex* (bound book, tree trunk) — symbolizing permanent knowledge compiled into the mycelial network, where each experiment becomes an eternal page in the collective wisdom library.

**Forms**:
- **K-Passport**: Local JSON file (pre-publication stage)
- **K-Codex**: Immutable DHT entry in Mycelix network (published, eternal)
- **Codex Fragment**: Partial data shared in federated learning
- **Codex Chain**: Linked series of experiments building on each other

**Required fields**:
- `run_id`: Unique identifier (UUID)
- `commit`: Git SHA of exact code version
- `config_hash`: SHA256 digest of configuration
- `seed`: Random number generator seed
- `experiment`: Experiment type identifier
- `params`: All hyperparameters
- `estimators`: Exact algorithms used (Φ variant, TE parameters)
- `metrics`: Output measurements (K, TAT, Recovery, etc.)
- `timestamp`: ISO 8601 datetime
- `researcher_agent`: AgentPubKey (when published to Mycelix)

**Purpose**: Enable bit-for-bit reproduction of any experiment years/decades later with cryptographic verification.

**Philosophy**: Transforms isolated experiments into eternal contributions to collective knowledge — "coherence as computational love" made permanent.

**Schema**: [`schemas/k_codex.json`](schemas/k_codex.json) (formerly `k_passport.json`)

**Backwards Compatibility**: "K-Passport" remains valid for local usage; migration to K-Codex terminology ongoing.

### Corridor

**Definition**: Region in parameter space where K exceeds a threshold (typically K > 1.0).

**Metrics**:
- **Corridor Rate**: Fraction of samples yielding K > threshold
- **Centroid**: Mean parameter values within corridor
- **Volume**: Proportion of parameter space in corridor
- **IoU (Intersection over Union)**: Overlap between corridors across conditions

**Significance**: Identifies "sweet spots" for coherence emergence.

**Example**: In a 5D parameter space, if 47.3% of samples yield K > 1.0, the corridor rate is 0.473.

---

## Metrics & Measures

### Φ (Integrated Information)

**Definition**: Measure of irreducible cause-effect power of a system.

**Estimator variants**:
- **Φ_E (Empirical)**: Data-driven estimation
- **Φ_AR (Autoregressive)**: Model-based approach
- **Φ* (Star)**: Computationally tractable approximation

**Units**: Bits

**Interpretation**: Higher Φ → more integrated information processing

**Computational cost**: O(2^N) for exact computation with N elements (NP-hard)

### Transfer Entropy (TE)

**Definition**: Directional information flow from source X to target Y.

**Formula**:
```
TE(X→Y) = I(Y_future ; X_past | Y_past)
```

**Parameters**:
- `k`: History length (embedding dimension)
- `lag`: Time delay between source and target
- `estimator`: kraskov, symbolic, gaussian

**TE Symmetry**:
```
TE_symmetry = 1 - |TE(X→Y) - TE(Y→X)| / max(TE(X→Y), TE(Y→X))
```

**Range**: [0, 1] where 1 = perfect bidirectional balance

### TAT (Think-Act-Think)

**Definition**: Ratio of deliberative to reactive behavior in a system.

**Formula**:
```
TAT = K / 2.0  (clipped to [0, 1])
```

**Interpretation**:
- **TAT ≈ 0**: Purely reactive, no integration
- **TAT ≈ 0.5**: Balanced deliberation and action
- **TAT ≈ 1**: Highly deliberative, maximum integration

**Origin**: Models the alternation between information processing (think) and physical action (act).

### Recovery Time

**Definition**: Estimated time for system to return to coherence after perturbation.

**Formula**:
```
Recovery = max(0.5, 2.2 - K)
```

**Units**: Arbitrary time units (normalized)

**Interpretation**: Lower K → longer recovery time (less resilient)

---

## System Components

### FRE (Fractal Reciprocity Engine)

**Definition**: Multi-universe simulation framework testing whether coherence scales fractally across interconnected systems.

**Hypothesis**: K_∞ ~ N^β where N = number of universes, β ∈ [0.15, 0.25]

**Phases**:
1. **Phase 1**: Corridor characterization (3D-6D parameter sweeps)
2. **Phase 2**: Controller validation (SAC, bioelectric rescue)
3. **Phase 3**: Scaling law verification
4. **Phase 4**: Distributed Holochain implementation

**Key files**: `fre/run.py`, `fre/corridor.py`, `fre/multi_universe.py`

### Universe Simulator

**Definition**: Lightweight physics-inspired model generating K-index trajectories from parameter sets.

**Input parameters**:
- `energy_gradient`: Δ potential driving information flow
- `communication_cost`: Overhead of inter-agent signaling
- `plasticity_rate`: Learning/adaptation speed
- `noise_spectrum_alpha`: 1/f^α noise characteristic
- `stimulus_jitter`: Environmental variability

**Output**: Dictionary containing K, Φ, TE metrics, and harmony components

**Philosophy**: Not physically accurate, but theoretically principled—captures essential dynamics for coherence research.

### Historical K(t) Pipeline

**Definition**: Reconstruction of Earth's coherence from 1800-2020 using proxy measures.

**Data sources**:
- Trade openness (reciprocity proxy)
- Life expectancy (flourishing proxy)
- Education index (wisdom proxy)
- Patent filings (creativity proxy)
- Renewable energy share (sustainability proxy)

**Method**:
1. Ingest datasets from OWID, World Bank
2. Normalize to [0, 1] ranges
3. Apply K-index formula with historical weighting
4. Bootstrap confidence intervals

**Output**: Time series with `[year, K, K_low, K_high]`

---

## Experimental Tracks

### Track A: Gate Keeper

**Purpose**: Validate baseline corridor metrics before advancing to complex tracks.

**Gates**:
1. Corridor rate > 0.4 in 5D
2. Centroid stability (SD < 0.05 across runs)
3. Robustness to noise (corridor maintained with ±10% parameter jitter)

**Status**: ✅ Complete (gates passed Nov 2024)

### Track B: SAC Controller

**Purpose**: Train a Soft Actor-Critic RL agent to maximize K-index.

**Environment**: `fre/sac_env.py` (Gymnasium-compatible)
- **State**: Current K, gradient estimates, parameter values
- **Action**: Continuous parameter adjustments
- **Reward**: Δ K + regularization terms

**Success criterion**: Achieve K > 1.2 in 80%+ of episodes (vs 47% baseline)

**Status**: 🚧 In progress (actor/critic implementation underway)

### Track C: Bioelectric Rescue

**Purpose**: Test whether bioelectric-like interventions can rescue low-K states.

**Mechanism**:
1. Detect K < 0.5 (crisis threshold)
2. Apply controlled perturbation to plasticity_rate
3. Measure recovery time and final K

**Hypothesis**: Timely intervention reduces recovery time by 50%+

**Status**: 📝 Specified (implementation pending)

---

## Technical Terms

### Preregistration

**Definition**: Public declaration of research hypotheses, methods, and analysis plans **before** data collection.

**Purpose**: Prevent p-hacking, HARKing (hypothesizing after results known), and publication bias.

**Platform**: Open Science Framework (OSF)

**Kosmic Lab practice**: All FRE phases preregistered in `docs/prereg_*.md` before execution.

### Harmony Integrity Checklist

**Definition**: Domain-specific validation rules preventing common experimental errors.

**Checks**:
1. Diversity metrics reward plurality (no shortcuts)
2. Corridor volume ≤ 1.0 after normalization
3. Estimator settings logged in K-passport
4. Visualization thresholds render correctly
5. Tests passing locally before PR

**Location**: [`CONTRIBUTING.md`](CONTRIBUTING.md)

### Holochain

**Definition**: Distributed peer-to-peer framework for running agent-based simulations without centralized servers.

**Kosmic Lab use case**: Phase 2 distributed FRE experiments with cross-universe TE computation.

**Components**:
- **Zomes**: WebAssembly modules (state, metrics, control, bridge)
- **DHT**: Distributed hash table for data storage
- **Validation**: Peer consensus on simulation correctness

**Status**: Prototype stage (MVP defined, implementation underway)

---

## Statistical Concepts

### Bootstrap Confidence Intervals

**Definition**: Nonparametric method for estimating uncertainty by resampling observed data.

**Kosmic Lab usage**: Historical K(t) uncertainty bands

**Method**:
1. Resample proxy data with replacement (1000 iterations)
2. Recompute K(t) for each bootstrap sample
3. Extract 2.5th and 97.5th percentiles → 95% CI

**Advantage**: No distributional assumptions required

### Jaccard Index (IoU)

**Definition**: Intersection over Union—measures overlap between two sets.

**Formula**:
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

**Kosmic Lab usage**: Compare corridors across experimental conditions

**Interpretation**:
- J = 0: No overlap
- J = 0.5: Moderate similarity
- J = 1: Identical corridors

---

## Philosophical Foundations

### Seven Primary Harmonies

The theoretical basis for K-index components:

1. **Resonant Coherence**: Φ, integrated information
2. **Pan-Sentient Flourishing**: Life expectancy, wellbeing proxies
3. **Integral Wisdom**: Education, knowledge accumulation
4. **Infinite Play**: Creativity, exploration (patent diversity)
5. **Universal Interconnectedness**: Trade openness, network density
6. **Sacred Reciprocity**: TE symmetry, balanced exchange
7. **Evolutionary Progression**: Renewable energy, sustainability

**Source**: Luminous Dynamics philosophy (`docs/luminous-library.md`)

### Fractal Reciprocity Law

**Hypothesis**: Coherence scales across nested hierarchies via reciprocal energy-information exchange.

**Formal statement**:
```
dK_i/dt = α·T_E→I(i,j) - β·T_I→E(j,i) + noise
```

Where:
- T_E→I: Energy → Information transfer (creative potential)
- T_I→E: Information → Energy feedback (stabilizing return)
- α, β ∝ D^(-p), p ∈ [0.2, 0.25] (dimension scaling)

**Prediction**: Multi-universe K_∞ > Σ K_i (superlinear emergence)

**Testing**: FRE Phase 3 scaling experiments

---

## Acronyms

- **AUC**: Area Under Curve
- **CI**: Confidence Interval
- **DHT**: Distributed Hash Table
- **FRE**: Fractal Reciprocity Engine
- **GQA**: Grouped Query Attention
- **IoU**: Intersection over Union
- **IRB**: Institutional Review Board
- **OSF**: Open Science Framework
- **OWID**: Our World in Data
- **RL**: Reinforcement Learning
- **SAC**: Soft Actor-Critic
- **TE**: Transfer Entropy
- **TAT**: Think-Act-Think
- **UUID**: Universally Unique Identifier

---

## References

- **K-passport schema**: `schemas/k_passport.json`
- **FRE design document**: `docs/fre_design_doc.md`
- **Historical K design**: `docs/historical_k_design.md`
- **Preregistration (Phase 1)**: `docs/prereg_fre_phase1.md`
- **Contributing guidelines**: `CONTRIBUTING.md`
- **Ethics framework**: `ETHICS.md`

---

*Last updated: 2025-11-09*
*Coherence is love made computational.*
