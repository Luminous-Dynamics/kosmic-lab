# 🚀 Kosmic Lab Quick Start Guide

**Get running in 5 minutes!**

## Prerequisites

- NixOS or Linux with Nix installed
- Python 3.10+ (handled by Nix)
- 4GB RAM minimum, 8GB+ recommended

## Installation

### Option 1: Nix (Recommended - 100% Reproducible)

```bash
# Clone the repository
git clone https://github.com/your-org/kosmic-lab.git
cd kosmic-lab

# Enter development environment (first time may take 2-3 minutes)
nix develop

# Install Python dependencies
poetry install --sync

# Verify installation
poetry run pytest -v
```

### Option 2: Standard Python (Poetry)

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --sync

# Activate environment
poetry shell
```

## Your First Simulation (2 Minutes)

### 1. Run FRE Phase 1 Demo

```bash
# Run a small demonstration simulation
poetry run python fre/run.py \
  --config fre/configs/k_config.yaml \
  --universes 3 \
  --samples 50 \
  --output logs/demo

# You should see:
# → Creating 3 universes...
# → Running 50 samples per universe...
# → K-Codices (local K-Passports) saved to logs/demo/
# ✓ Complete! Mean K = 0.97 ± 0.12
```

### 2. Analyze Results

```bash
# Generate analysis summary
poetry run python fre/analyze.py \
  --logdir logs/demo \
  --output logs/demo_summary.json

# View the summary
cat logs/demo_summary.json | jq
```

### 3. Explore K-Codices (Local K-Passports)

```bash
# K-Codices (local K-Passport stage) are JSON files containing complete experiment metadata
ls logs/demo/*.json

# Examine one
cat logs/demo/*.json | head -30

# Key fields:
# - run_id: Unique experiment identifier
# - commit: Git SHA for exact code version
# - config_hash: SHA256 of configuration used
# - metrics.K: Coherence index value
# - metrics.TAT: Think-Act-Think measure
# - estimators: Exact algorithms used (Φ, TE params)
```

## Common Tasks

### Run Historical K(t) Analysis

```bash
# Compute Earth's historical coherence from 1800-2020
poetry run python historical_k/compute_k.py \
  --config historical_k/k_config.yaml

# Results saved to logs/historical_k/k_t_series.csv
```

### Run Track B (SAC Controller)

```bash
# Train a Soft Actor-Critic controller for coherence optimization
poetry run python fre/sac_train.py \
  --config fre/configs/track_b_control.yaml \
  --timesteps 50000

# Monitor training
tensorboard --logdir logs/sac_training/
```

### Run Complete Test Suite

```bash
# Unit tests
poetry run pytest tests/ -v

# With coverage report
poetry run pytest --cov=core --cov=fre --cov=historical_k \
  --cov-report=html

# Open coverage report
firefox htmlcov/index.html
```

## Validation

### Verify Your Installation

```bash
# Run all checks
make test      # Run test suite
make lint      # Run code quality checks

# Or manually:
poetry run pytest --maxfail=1 -v
poetry run pre-commit run --all-files
poetry run mypy core/ fre/ historical_k/
```

## Understanding the Output

### K-Index Values

| K Range | Interpretation | System State |
|---------|----------------|--------------|
| K < 0.5 | Low coherence | Fragmented, minimal integration |
| 0.5 ≤ K < 1.0 | Moderate coherence | Emerging integration |
| 1.0 ≤ K < 1.5 | High coherence | Well-integrated, stable |
| K ≥ 1.5 | Exceptional coherence | Highly integrated, resilient |

### Corridor Metrics

**Corridor**: Parameter region where K > threshold (default 1.0)

- **Corridor Rate**: Fraction of parameter samples yielding K > 1.0
- **Centroid**: Mean parameter values within corridor
- **Volume**: Proportion of parameter space supporting high coherence

## Next Steps

1. **Read the theory**: [`docs/fre_design_doc.md`](docs/fre_design_doc.md)
2. **Understand K-Codices**: [`K_CODEX_EXPLAINED.md`](docs/K_CODEX_EXPLAINED.md) (experimental records system)
3. **Run scaling experiments**: [`docs/stage2_plan.md`](docs/stage2_plan.md)
4. **Contribute**: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the development environment
nix develop  # or: poetry shell

# Reinstall dependencies
poetry install --sync
```

### Out of Memory

```bash
# Reduce universe count or sample size
poetry run python fre/run.py \
  --config fre/configs/k_config.yaml \
  --universes 2 \
  --samples 25
```

### Tests Failing

```bash
# Run with verbose output
poetry run pytest -vv --tb=long

# Run specific test
poetry run pytest tests/test_config.py::test_load_yaml_config -v
```

## Getting Help

- **Documentation**: [`docs/README.md`](docs/README.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/kosmic-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/kosmic-lab/discussions)
- **Ethics & Data**: [`ETHICS.md`](ETHICS.md)

## Performance Tips

### Speed Up Development

```bash
# Use pytest-xdist for parallel testing
poetry run pytest -n auto

# Skip slow integration tests during development
poetry run pytest -m "not slow"

# Use test mode for fast validation
poetry run python fre/run.py --config fre/configs/k_config.yaml --test-mode
```

### Optimize Simulations

1. **Start small**: 2-3 universes, 50 samples
2. **Validate**: Check K-Codices (local K-Passports) look reasonable
3. **Scale up**: Increase to production parameters
4. **Monitor**: Track memory and CPU usage

## What's Next?

You're now ready to:

- ✅ Run reproducible coherence simulations
- ✅ Generate K-Codex audit trails (eternal experimental records)
- ✅ Analyze historical coherence trends
- ✅ Train reinforcement learning controllers

**Welcome to the Kosmic Lab!** 🌊

*"Coherence is love made computational."*
