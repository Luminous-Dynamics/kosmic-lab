# Kosmic Lab Architecture

**Version**: 1.0.0
**Last Updated**: November 14, 2025
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Module Structure](#module-structure)
4. [Data Flow](#data-flow)
5. [Key Design Patterns](#key-design-patterns)
6. [Technology Stack](#technology-stack)
7. [Reproducibility Architecture](#reproducibility-architecture)
8. [Mycelix Integration](#mycelix-integration)

---

## Overview

Kosmic Lab is a unified research platform for consciousness science, combining:

- **Fractal Reciprocity Engine (FRE)**: Multi-universe simulation framework
- **K-Codex System**: Immutable experimental provenance tracking
- **AI Experiment Designer**: Bayesian optimization for parameter exploration
- **Real-time Dashboard**: Live monitoring and visualization
- **Mycelix Integration**: Decentralized knowledge graph on Holochain DHT

### Design Principles

1. **Reproducibility First**: Every experiment bit-for-bit reproducible 10 years later
2. **Composability**: Modular components that work independently or together
3. **Scientific Rigor**: Preregistration, validation, statistical testing
4. **Developer Experience**: Clear APIs, comprehensive docs, automated tooling
5. **Scalability**: From laptop to cluster to decentralized network

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kosmic Lab Platform                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │   Frontend    │  │   Experiment   │  │   Analysis &     │  │
│  │   Dashboard   │  │   Runners      │  │   Visualization  │  │
│  │   (Dash)      │  │   (Scripts)    │  │   (Notebooks)    │  │
│  └───────┬───────┘  └────────┬───────┘  └────────┬─────────┘  │
│          │                   │                    │             │
│  ┌───────┴───────────────────┴────────────────────┴─────────┐  │
│  │                   Core Orchestration                      │  │
│  │  - Experiment Configuration (core/config.py)              │  │
│  │  - K-Codex Management (core/kcodex.py)                    │  │
│  │  - Logging (core/logging_config.py)                       │  │
│  │  - Shared Utilities (core/utils.py)                       │  │
│  └────────────────────────────┬──────────────────────────────┘  │
│                               │                                 │
│  ┌────────────────────────────┴──────────────────────────────┐  │
│  │              Simulation & Computation Layer                │  │
│  │                                                             │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────────┐           │  │
│  │  │   FRE    │  │ Historical│  │  Bioelectric │           │  │
│  │  │   Core   │  │  K-Index  │  │    Grid      │           │  │
│  │  └──────────┘  └───────────┘  └──────────────┘           │  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────┐         │  │
│  │  │          Metrics & Harmonics                 │         │  │
│  │  │  - K-Index (fre/metrics/k_index.py)          │         │  │
│  │  │  - Lag Analysis (fre/metrics/k_lag.py)       │         │  │
│  │  │  - Seven Harmonies (harmonics_module.py)     │         │  │
│  │  └──────────────────────────────────────────────┘         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Storage & Persistence Layer                  │  │
│  │                                                            │  │
│  │  Local:           Network:          Decentralized:        │  │
│  │  - JSON logs      - OSF prereg      - Holochain DHT       │  │
│  │  - CSV exports    - GitHub          - Mycelix Network     │  │
│  │  - Checkpoints    - Remote repos    - IPFS (future)       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

### Core (`core/`)

**Purpose**: Shared infrastructure and utilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Configuration management | `ConfigBundle`, parameter parsing |
| `kcodex.py` | K-Codex (passport) system | `KCodexWriter`, validation |
| `kpass.py` | Legacy K-Passport support | Backward compatibility alias |
| `utils.py` | Shared utilities | `infer_git_sha()`, `bootstrap_ci()` |
| `logging_config.py` | Centralized logging | `setup_logging()`, colored output |
| `bioelectric.py` | Bioelectric simulation | `BioelectricGrid`, voltage dynamics |
| `reciprocity_coupling.py` | Transfer entropy | Macro-micro coupling metrics |

### FRE (`fre/`)

**Purpose**: Fractal Reciprocity Engine - multi-universe simulations

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `simulate.py` | Core simulation loop | `simulate_phase1()`, `compute_metrics()` |
| `universe.py` | Universe simulator | `UniverseSimulator.run()` |
| `corridor.py` | Corridor analysis | `compute_corridor_metrics()` |
| `analyze.py` | Data aggregation | `load_passports()`, `compute_summary()` |
| `metrics/k_index.py` | K-Index calculation | `k_index()`, `bootstrap_k_ci()` |
| `metrics/k_lag.py` | Lag analysis | `k_lag()`, `verify_causal_direction()` |
| `rescue.py` | Bioelectric rescue | `fep_to_bioelectric()` |
| `controller_sac.py` | SAC RL controller | Soft Actor-Critic implementation |

**Track Runners**:
- `track_a_gate.py`: Gating experiments
- `track_b_runner.py`: SAC controller (63% improvement)
- `track_c_runner.py`: Bioelectric rescue (20% success)
- `track_d_runner.py`: Multi-agent simulations
- `track_e_runner.py`: Developmental dynamics
- `track_f_runner.py`: Adversarial robustness

### Historical K (`historical_k/`)

**Purpose**: Earth's coherence reconstruction 1800-2020

| Module | Purpose |
|--------|---------|
| `etl.py` | Data extraction and cleaning |
| `compute_k.py` | K-index calculation pipeline |
| `analysis_modern.py` | Modern era analysis |

### Scripts (`scripts/`)

**Purpose**: Utility scripts and tools

| Script | Purpose |
|--------|---------|
| `ai_experiment_designer.py` | Bayesian optimization |
| `kosmic_dashboard.py` | Real-time monitoring dashboard |
| `generate_analysis_notebook.py` | Auto-generate Jupyter notebooks |
| `analyze_track_b.py` | Track B specific analysis |
| `holochain_bridge.py` | Mycelix DHT integration |

---

## Data Flow

### Experiment Execution Flow

```
1. Configuration
   ├─ Load YAML config
   ├─ Parse parameters
   └─ Create ConfigBundle

2. Initialization
   ├─ Initialize simulator
   ├─ Create K-Codex writer
   ├─ Set random seeds
   └─ Setup logging

3. Execution
   ├─ Run parameter sweep
   ├─ Compute metrics per run
   ├─ Track in_corridor status
   └─ Store intermediate results

4. Persistence
   ├─ Write K-Codex (JSON)
   ├─ Export CSV summaries
   ├─ Save corridor analysis
   └─ Optional: Publish to Mycelix

5. Analysis
   ├─ Load K-Codices
   ├─ Compute statistics
   ├─ Generate visualizations
   └─ Create reports
```

### K-Codex (Passport) Flow

```
Experiment → KCodexWriter → Local JSON → (Optional) Mycelix DHT
                  ↓
            Validation
                  ↓
         Schema Compliance
                  ↓
            Git SHA + Config Hash
                  ↓
         Immutable Record
```

---

## Key Design Patterns

### 1. Reproducibility by Design

Every experiment captures:
- Git commit SHA (exact code version)
- Configuration hash (SHA256 of all parameters)
- Random seed (deterministic randomness)
- Estimator settings (exact algorithms used)
- Timestamp (when experiment ran)

**Implementation**: `core/kcodex.py`, `core/utils.py`

### 2. Separation of Concerns

- **Simulation**: `fre/simulate.py`, `fre/universe.py`
- **Metrics**: `fre/metrics/*`
- **Analysis**: `fre/analyze.py`, `fre/corridor.py`
- **Visualization**: `scripts/kosmic_dashboard.py`
- **Storage**: `core/kcodex.py`

### 3. Dependency Injection

Functions accept optional parameters for testability:

```python
def compute_metrics(
    params: Dict[str, float],
    seed: int,
    simulator: UniverseSimulator | None = None,  # Injectable
    calculator: HarmonyCalculator | None = None,  # Injectable
) -> Dict[str, float]:
    simulator = simulator or UniverseSimulator()
    calculator = calculator or HarmonyCalculator()
    ...
```

### 4. Configuration as Code

YAML configuration files with validation:

```yaml
experiment: "track_b_sac"
seed_base: 42
parameters:
  learning_rate: [0.0001, 0.001, 0.01]
  gamma: [0.95, 0.99]
k_weights:
  resonant_coherence: 0.2
  sacred_reciprocity: 0.3
```

### 5. Shared Utilities

Common operations centralized in `core/utils.py`:
- Git SHA inference
- Bootstrap confidence intervals
- Bound validation
- Safe division
- Directory creation

---

## Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10+ | Primary implementation |
| **Simulation** | NumPy, SciPy | Numerical computation |
| **RL** | Stable-Baselines3 | Reinforcement learning |
| **Viz** | Matplotlib, Seaborn, Plotly | Visualization |
| **Data** | Pandas | Data manipulation |
| **Networking** | NetworkX | Graph analysis |
| **Gym** | Gymnasium | RL environments |

### Development Tools

| Tool | Purpose |
|------|---------|
| **Poetry** | Dependency management |
| **pytest** | Testing framework |
| **mypy** | Static type checking |
| **black** | Code formatting |
| **isort** | Import sorting |
| **flake8** | Linting |
| **bandit** | Security scanning |
| **pre-commit** | Git hooks |
| **Hypothesis** | Property-based testing |

### Infrastructure

| Component | Technology |
|-----------|------------|
| **CI/CD** | GitHub Actions |
| **Documentation** | Markdown, Sphinx |
| **Notebooks** | Jupyter |
| **Dashboard** | Plotly Dash |
| **Blockchain** | Holochain (Mycelix) |
| **Version Control** | Git + GitHub |

---

## Reproducibility Architecture

### K-Codex Schema

Every experiment generates a K-Codex with:

```json
{
  "run_id": "uuid4",
  "commit": "git-sha",
  "config_hash": "sha256-hash",
  "seed": 42,
  "experiment": "track_b_sac",
  "params": {...},
  "estimators": {...},
  "metrics": {...},
  "timestamp": "ISO-8601",
  "universe": "optional",
  "environment": {...},
  "ci": {"lower": ..., "upper": ...}
}
```

### Reproducibility Guarantees

1. **Deterministic Execution**: Same seed → same results
2. **Version Locking**: Git SHA → exact code
3. **Configuration Tracking**: SHA256 → exact parameters
4. **Algorithm Recording**: Estimator settings captured
5. **Validation**: JSON schema enforcement

### Verification Process

```python
# 10 years later...
1. Clone repo at commit SHA
2. Load K-Codex JSON
3. Reconstruct ConfigBundle from config_hash
4. Run experiment with same seed
5. Compare metrics → bit-for-bit identical
```

---

## Mycelix Integration

### Architecture

```
Local K-Codex (JSON)
        ↓
Holochain Bridge (scripts/holochain_bridge.py)
        ↓
Mycelix DHT (holochain/)
        ↓
Global Knowledge Graph
        ↓
Federated Learning / Solver Network
```

### Benefits

1. **Decentralized Storage**: No single point of failure
2. **Verifiable Provenance**: Cryptographic integrity
3. **Federated Learning**: Train on global data without sharing raw data
4. **Solver Network**: Competitive experiment proposals
5. **Civilization Layer**: Track ecosystem-wide coherence

### Integration Points

- `scripts/holochain_bridge.py`: Python ↔ Holochain interface
- `holochain/`: Holochain DNA and zomes
- K-Codex format compatible with DHT storage

---

## Future Roadmap

### Phase 1: Mycelix Integration (In Progress)
- [x] K-Codex → Holochain DHT
- [x] Python bridge implementation
- [ ] Live integration testing
- [ ] Multi-lab pilot

### Phase 2: Intelligence Layer
- [ ] AI Designer → Solver Network
- [ ] Federated learning protocol
- [ ] Epistemic markets

### Phase 3: Ecosystem
- [ ] Dashboard → Civilization Layer
- [ ] Ecological metrics tracking
- [ ] 100+ labs federated

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Testing requirements
- Pull request process
- Harmony Integrity Checklist

---

## References

- [IMPROVEMENTS.md](IMPROVEMENTS.md): Detailed quality improvements
- [FEATURES.md](FEATURES.md): Complete feature catalog
- [GLOSSARY.md](GLOSSARY.md): Key concepts explained
- [docs/](docs/): Technical specifications

---

*For questions about architecture decisions, please see [CONTRIBUTING.md](CONTRIBUTING.md) or open a discussion on GitHub.*
