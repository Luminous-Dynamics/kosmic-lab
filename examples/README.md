# Kosmic Lab Examples

This directory contains hands-on examples demonstrating key features of Kosmic Lab. Each example is self-contained and includes detailed comments.

## Quick Start

All examples can be run with Poetry:

```bash
poetry run python examples/01_hello_kosmic.py
poetry run python examples/02_advanced_k_index.py
poetry run python examples/03_multi_universe.py
poetry run python examples/04_bioelectric_rescue.py
poetry run python examples/05_neuroscience_eeg_analysis.py
poetry run python examples/06_ai_model_coherence.py
```

## Examples Overview

### 01_hello_kosmic.py - Getting Started Tutorial

**Difficulty**: Beginner
**Runtime**: ~5 seconds
**Topics**: K-Index, K-Codex, logging, basic workflow

The perfect starting point! This example walks through:
- Setting up logging
- Generating synthetic data
- Computing K-Index metrics
- Creating K-Codex for reproducibility
- Analyzing results

**Key Learning**:
- How to use the core APIs
- K-Index computation and interpretation
- Reproducibility tracking with K-Codex

**Output**:
- `logs/hello_kosmic.log` - Detailed execution log
- `logs/hello_kosmic_kcodex.json` - Reproducibility record

---

### 02_advanced_k_index.py - Advanced Statistical Analysis

**Difficulty**: Intermediate
**Runtime**: ~30 seconds
**Topics**: Bootstrap CI, K-Lag, statistical power, visualization

Deep dive into K-Index analysis techniques:
- Bootstrap confidence intervals
- Multiple scenario comparison
- Time series analysis with K-Lag
- Statistical power analysis
- Publication-quality visualizations

**Key Learning**:
- How to estimate uncertainty
- Temporal relationship detection
- Sample size requirements
- Interpreting correlation vs K-Index

**Output**:
- `logs/examples/02_advanced_k_index_analysis.png` - Visualizations
- Console output with statistical summaries

**Requirements**:
```bash
poetry add matplotlib  # Optional, for visualizations
```

---

### 03_multi_universe.py - Parameter Space Exploration

**Difficulty**: Intermediate
**Runtime**: ~2 minutes (serial), ~30 seconds (parallel)
**Topics**: Parameter sweeps, parallel execution, optimization

Explore multiple universe configurations:
- Grid search over parameter space
- Parallel execution for performance
- Identifying optimal parameters
- Visualization of parameter landscape

**Key Learning**:
- How to run systematic experiments
- Parameter optimization strategies
- Scaling up simulations
- Result aggregation and analysis

**Output**:
- `logs/examples/03_parameter_space.png` - Parameter visualization
- `logs/multi_universe/multi_universe_results.json` - Full results

**Usage**:
```bash
# Serial execution
poetry run python examples/03_multi_universe.py

# Parallel execution (faster, more CPU)
poetry run python examples/03_multi_universe.py --parallel

# Custom configuration
poetry run python examples/03_multi_universe.py --timesteps 200 --seeds 5
```

---

### 04_bioelectric_rescue.py - Bioelectric Intervention

**Difficulty**: Advanced
**Runtime**: ~10 seconds
**Topics**: Bioelectric circuits, rescue mechanisms, FEP, consciousness dynamics

Demonstrates bioelectric rescue for consciousness collapse:
- Simulating consciousness collapse
- Detecting collapse via FEP errors
- Applying bioelectric interventions
- Comparing rescued vs baseline outcomes
- K-Codex tracking for rescue experiments

**Key Learning**:
- Bioelectric principles (inspired by Michael Levin)
- Free Energy Principle in practice
- Rescue mechanism design
- Time series intervention analysis

**Output**:
- `logs/examples/04_bioelectric_rescue.png` - Rescue trajectories
- `logs/bioelectric_rescue/rescue_experiment.json` - K-Codex record

**Theoretical Background**:
This example is inspired by:
- Michael Levin's work on bioelectric control of pattern
- Karl Friston's Free Energy Principle
- Active inference and homeostasis

---

### 05_neuroscience_eeg_analysis.py - Neuroscience Application

**Difficulty**: Intermediate
**Runtime**: ~15 seconds
**Topics**: EEG analysis, consciousness monitoring, clinical applications, K-Lag

Real-world neuroscience application demonstrating:
- EEG signal processing and feature extraction
- Consciousness level prediction from EEG data
- K-Index for validating prediction coherence
- Temporal dynamics analysis with K-Lag
- Clinical interpretation and decision-making

**Key Learning**:
- Applying K-Index to real-world neuroscience
- EEG-based consciousness assessment
- Temporal delay detection (prediction lag)
- Clinical threshold determination
- Multi-level consciousness analysis

**Output**:
- `outputs/example_05_eeg_analysis.png` - Comprehensive visualization
- `logs/example_05_eeg_kcodex.json` - K-Codex records for 4 consciousness levels

**Use Cases**:
- Anesthesia depth monitoring
- ICU sedation assessment
- Sleep stage classification
- Disorders of consciousness diagnosis

---

### 06_ai_model_coherence.py - AI Interpretability

**Difficulty**: Intermediate
**Runtime**: ~20 seconds
**Topics**: AI alignment, interpretability, internal coherence, overconfidence detection

Analyze internal coherence in neural networks:
- Train a simple MLP classifier
- Extract hidden layer representations
- Measure coherence between internals and predictions
- Detect model overconfidence
- Validate alignment between "understanding" and output

**Key Learning**:
- Applying K-Index to AI interpretability
- Analyzing model internal representations
- Detecting misalignment between internals and predictions
- Calibration analysis
- Multi-level coherence (representation → prediction → truth)

**Output**:
- `outputs/example_06_ai_coherence.png` - Model analysis visualization
- `logs/example_06_ai_coherence_kcodex.json` - K-Codex record

**Use Cases**:
- Model debugging and validation
- Alignment research (AI safety)
- Detecting when models are uncertain but confident
- Understanding representation quality
- Trustworthy AI development

---

### interactive_tutorial.ipynb - Complete Interactive Tutorial

**Difficulty**: Beginner to Advanced
**Runtime**: Self-paced (~1 hour)
**Topics**: All major features with executable code

Complete hands-on Jupyter notebook covering:
- All core utilities and metrics
- K-Index, K-Lag, K-Codex workflows
- Bioelectric and FRE simulations
- Visualization and analysis
- Interactive exercises and challenges

**Key Learning**:
- End-to-end workflows
- Best practices
- Common patterns
- Troubleshooting

**Usage**:
```bash
jupyter notebook examples/interactive_tutorial.ipynb
# or
jupyter lab examples/interactive_tutorial.ipynb
```

---

## Example Progression Path

We recommend following this learning path:

```
01_hello_kosmic.py              (Start here - 5 min)
    ↓
02_advanced_k_index.py          (Statistical depth - 15 min)
    ↓
interactive_tutorial.ipynb      (Interactive learning - 60 min) [OPTIONAL]
    ↓
03_multi_universe.py            (Scaling up - 20 min)
    ↓
04_bioelectric_rescue.py        (Advanced concepts - 20 min)
    ↓
05_neuroscience_eeg_analysis.py (Real-world neuroscience - 15 min)
    ↓
06_ai_model_coherence.py        (AI interpretability - 20 min)
```

**Total time**: ~2 hours (or ~3 hours with interactive tutorial)

**Quick Path** (essentials only): 01 → 02 → 05 or 06 (~35 min)

## Common Patterns

### Pattern 1: Setup and Logging

All examples use consistent setup:

```python
from core.logging_config import setup_logging, get_logger

setup_logging(level="INFO", log_file="logs/mylog.log")
logger = get_logger(__name__)
```

### Pattern 2: K-Codex Tracking

For reproducibility:

```python
from core.kcodex import KCodexWriter

kcodex = KCodexWriter("logs/experiment.json")
kcodex.log_experiment(
    experiment_name="my_experiment",
    params={"param1": value1},
    metrics={"metric1": result1},
    seed=42
)
```

### Pattern 3: Metrics Computation

Standard workflow:

```python
from fre.metrics.k_index import k_index, bootstrap_k_ci

# Compute K-Index
k = k_index(observed, actual)

# Get confidence interval
ci_lower, ci_upper = bootstrap_k_ci(
    observed, actual,
    n_bootstrap=1000,
    confidence_level=0.95
)
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Ensure you're in the project root
cd /path/to/kosmic-lab

# Activate Poetry environment
poetry install
poetry shell

# Run example
python examples/01_hello_kosmic.py
```

### Missing Dependencies

Some examples have optional dependencies (e.g., matplotlib):

```bash
# Install optional visualization dependencies
poetry add matplotlib

# Or run without visualizations - examples will skip gracefully
```

### Performance Issues

For slow execution:

```bash
# Use parallel mode (where supported)
python examples/03_multi_universe.py --parallel

# Reduce iterations
python examples/03_multi_universe.py --timesteps 50 --seeds 2
```

## Next Steps

After completing the examples:

1. **Run Benchmarks**: `poetry run python benchmarks/run_benchmarks.py`
2. **Launch Dashboard**: `make dashboard` or `python scripts/kosmic_dashboard.py`
3. **Read Architecture**: See `ARCHITECTURE.md` for system design
4. **Write Your Own**: Create experiments in `experiments/` directory
5. **Contribute**: See `CONTRIBUTING.md` for guidelines

## Additional Resources

- **Main README**: `../README.md` - Project overview
- **Architecture**: `../ARCHITECTURE.md` - System design
- **Troubleshooting**: `../TROUBLESHOOTING.md` - Common issues
- **API Docs**: Coming soon - auto-generated API documentation
- **Dashboard**: `../scripts/kosmic_dashboard.py` - Real-time monitoring

## Questions?

- Check `TROUBLESHOOTING.md` for common issues
- Review `ARCHITECTURE.md` for system understanding
- Open an issue on GitHub for bugs or feature requests
- See `CONTRIBUTING.md` for development guidelines

---

**Happy Experimenting!** 🌊🔬✨
