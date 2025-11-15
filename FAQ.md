# Frequently Asked Questions (FAQ)

**Last Updated**: 2025-11-15

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Getting Started](#getting-started)
3. [K-Index & Metrics](#k-index--metrics)
4. [K-Codex System](#k-codex-system)
5. [Experiments & Simulations](#experiments--simulations)
6. [Development & Contributing](#development--contributing)
7. [Troubleshooting](#troubleshooting)
8. [Performance & Scalability](#performance--scalability)
9. [Security & Privacy](#security--privacy)
10. [Advanced Topics](#advanced-topics)

---

## General Questions

### What is Kosmic Lab?

Kosmic Lab is a unified research platform for consciousness science that combines multi-universe simulation (Fractal Reciprocity Engine), immutable experimental tracking (K-Codex), AI-driven experiment design, and decentralized knowledge sharing (Mycelix integration).

### Who should use Kosmic Lab?

- **Researchers** studying consciousness, coherence, and complex systems
- **Data scientists** working on reproducible experiments
- **Developers** interested in bioelectric simulation and FEP (Free Energy Principle)
- **Students** learning about computational consciousness research

### What makes Kosmic Lab unique?

1. **Perfect Reproducibility**: Every experiment is bit-for-bit reproducible via K-Codex
2. **Bioelectric Integration**: Simulate Michael Levin-inspired bioelectric rescue interventions
3. **Multi-Universe**: Run thousands of parallel simulations with parameter sweeps
4. **Production-Ready**: World-class infrastructure with 48+ tests, CI/CD, and comprehensive docs

### Is Kosmic Lab free and open source?

Yes! Kosmic Lab is released under the MIT License. You can use, modify, and distribute it freely.

---

## Getting Started

### How do I install Kosmic Lab?

**Quick Install (2 minutes)**:
```bash
git clone https://github.com/Luminous-Dynamics/kosmic-lab.git
cd kosmic-lab
./scripts/setup_dev_env.sh
```

See [README.md](README.md#quick-start-5-minutes) for detailed instructions.

### What are the system requirements?

**Minimum**:
- Python 3.10+
- 4GB RAM
- 2GB disk space

**Recommended**:
- Python 3.11 or 3.12
- 8GB+ RAM (for large simulations)
- 5GB disk space
- Multi-core CPU for parallel execution

**Optional**:
- GPU for deep learning experiments (Track F)
- Docker for containerized deployment

### Do I need to know Python?

Basic Python knowledge is helpful but not required for running experiments. Our [examples/](examples/) provide ready-to-run scripts:

1. **examples/01_hello_kosmic.py** - 5 min introduction (beginner-friendly)
2. **examples/02_advanced_k_index.py** - 15 min statistical depth
3. **examples/03_multi_universe.py** - 20 min parameter sweeps
4. **examples/04_bioelectric_rescue.py** - 20 min advanced theory

### Where do I start learning?

**Learning Path** (1 hour total):
1. Read [QUICKSTART.md](QUICKSTART.md) - 10 min
2. Run `examples/01_hello_kosmic.py` - 5 min
3. Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 10 min
4. Run examples 02-04 - 45 min
5. Explore [docs/](docs/) for deep dives

---

## K-Index & Metrics

### What is K-Index?

K-Index is our primary metric measuring observer-observed coherence in consciousness experiments. It quantifies how well predicted (observed) consciousness matches actual consciousness measurements.

**Range**: 0.0 to 1.0 (higher = better coherence)

**Interpretation**:
- **0.8+**: Excellent coherence
- **0.6-0.8**: Good coherence
- **0.4-0.6**: Moderate coherence
- **<0.4**: Low coherence

See [docs/GLOSSARY.md](docs/GLOSSARY.md#k-index) for mathematical definition.

### How do I compute K-Index?

```python
from fre.metrics.k_index import k_index

k = k_index(observed, actual)
print(f"K-Index: {k:.4f}")
```

See [examples/01_hello_kosmic.py](examples/01_hello_kosmic.py) for complete example.

### What is K-Lag?

K-Lag identifies temporal lag between observed and actual consciousness trajectories. It finds the optimal time shift that maximizes K-Index.

**Usage**:
```python
from fre.metrics.k_lag import k_lag

results = k_lag(observed, actual, max_lag=20)
print(f"Best lag: {results['best_lag']} timesteps")
print(f"K at best lag: {results['k_at_best_lag']:.4f}")
```

See [examples/02_advanced_k_index.py](examples/02_advanced_k_index.py#L150) for detailed example.

### How do I compute confidence intervals?

```python
from core.utils import bootstrap_confidence_interval
from fre.metrics.k_index import k_index

ci_lower, ci_upper = bootstrap_confidence_interval(
    observed,
    statistic=lambda x: k_index(x, actual),
    n_bootstrap=1000,
    confidence_level=0.95
)

print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

See [examples/02_advanced_k_index.py](examples/02_advanced_k_index.py#L80) for full implementation.

---

## K-Codex System

### What is K-Codex?

K-Codex (formerly K-Passport) is our reproducibility tracking system. Every experiment generates an immutable JSON record containing:

- Git commit SHA (exact code version)
- Configuration hash (all parameters)
- Random seed (deterministic randomness)
- Metrics and results
- Environment details (Python version, platform, packages)
- Timestamp

**Goal**: Bit-for-bit reproducibility 10 years later.

### How do I use K-Codex?

```python
from core.kcodex import KCodexWriter

kcodex = KCodexWriter("logs/experiment_kcodex.json")
kcodex.log_experiment(
    experiment_name="my_experiment",
    params={"n_samples": 100, "threshold": 0.5},
    metrics={"k_index": 0.875},
    seed=42,
    extra_metadata={"notes": "Baseline run"}
)
```

See [core/README.md](core/README.md#kcodexpy) for details.

### Can I migrate from K-Passport to K-Codex?

Yes! K-Codex is backward compatible with K-Passport. Use `core.kpass` as an alias:

```python
from core.kpass import KPassportWriter  # Works!
# This is just an alias to KCodexWriter
```

See [docs/K_CODEX_V2_MIGRATION_GUIDE.md](docs/K_CODEX_V2_MIGRATION_GUIDE.md) for migration details.

### How do I validate K-Codex records?

```bash
# Using JSON schema validation
poetry run python -c "
import json
import jsonschema

with open('schemas/kcodex.schema.json') as f:
    schema = json.load(f)

with open('logs/experiment_kcodex.json') as f:
    kcodex = json.load(f)

jsonschema.validate(kcodex, schema)
print('✓ Valid K-Codex')
"
```

---

## Experiments & Simulations

### What are the available experiment tracks?

We have 6 validated experiment tracks:

| Track | Focus | Status | Result |
|-------|-------|--------|--------|
| **A** | Gating experiments | Implemented | Baseline |
| **B** | SAC RL controller | **Validated** | **63% improvement** |
| **C** | Bioelectric rescue | **Validated** | **20% success rate** |
| **D** | Multi-agent systems | Implemented | Exploratory |
| **E** | Developmental dynamics | Implemented | Exploratory |
| **F** | Adversarial robustness | In Progress | Testing |

See [docs/track-results/](docs/track-results/) for detailed results.

### How do I run an experiment?

**Simple Experiment**:
```bash
poetry run python examples/01_hello_kosmic.py
```

**Track B (SAC Controller)**:
```bash
poetry run python fre/track_b_runner.py
```

**Track C (Bioelectric Rescue)**:
```bash
poetry run python fre/track_c_runner.py
```

**Parameter Sweep**:
```bash
poetry run python examples/03_multi_universe.py
```

### How long do experiments take?

- **Simple K-Index** (N=100): <1 second
- **Bootstrap CI** (N=100, 1000 iterations): ~5 seconds
- **Single universe** (1000 timesteps): ~10 seconds
- **Parameter sweep** (10 configs × 5 seeds): ~5 minutes
- **Track B validation** (full sweep): ~30 minutes
- **Track C validation** (full sweep): ~45 minutes

Use `N_CORES` environment variable for parallel execution.

### Can I run experiments in parallel?

Yes! Use the `parallel` flag and `n_cores` parameter:

```python
from fre.universe import UniverseSimulator

simulator = UniverseSimulator(parallel=True, n_cores=4)
```

Or set environment variable:
```bash
export N_CORES=4
poetry run python examples/03_multi_universe.py
```

---

## Development & Contributing

### How do I set up the development environment?

**Automated Setup** (2 minutes):
```bash
./scripts/setup_dev_env.sh
```

**Manual Setup**:
```bash
poetry install
poetry run pre-commit install
cp .env.example .env
make test
```

See [DEVELOPMENT.md](DEVELOPMENT.md#quick-start-5-minutes) for details.

### What are the coding standards?

- **Style**: PEP 8 (enforced by Black, line length 88)
- **Imports**: Sorted with isort (Black-compatible)
- **Type hints**: Required (checked by mypy)
- **Docstrings**: Google style (enforced by pydocstyle)
- **Testing**: 90%+ coverage (pytest)
- **Security**: No vulnerabilities (bandit scan)

See [CONTRIBUTING.md](CONTRIBUTING.md#coding-standards) for complete guidelines.

### How do I run code quality checks?

**Automated** (recommended):
```bash
./scripts/check_code_quality.sh
```

**Auto-fix formatting**:
```bash
./scripts/check_code_quality.sh --fix
```

**Individual checks**:
```bash
make format      # Black + isort
make lint        # flake8
make type-check  # mypy
make test        # pytest
make security-check  # bandit
```

### How do I submit a pull request?

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run quality checks: `./scripts/check_code_quality.sh`
5. Commit: `git commit -m "feat: add amazing feature"`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request (use our template)

See [CONTRIBUTING.md](CONTRIBUTING.md#pull-request-process) for detailed steps.

### How do I add a new metric?

1. Create `fre/metrics/my_metric.py`
2. Implement metric function with type hints and docstring
3. Add tests in `tests/test_my_metric.py`
4. Add to `fre/metrics/__init__.py`
5. Document in [docs/GLOSSARY.md](docs/GLOSSARY.md)
6. Add example usage

See [DEVELOPMENT.md](DEVELOPMENT.md#adding-a-new-feature) for template.

---

## Troubleshooting

### ImportError: No module named 'core'

**Solution**: Install dependencies with Poetry:
```bash
poetry install
poetry shell
```

Or use `poetry run`:
```bash
poetry run python examples/01_hello_kosmic.py
```

### Pre-commit hooks failing

**Solution 1** - Auto-fix formatting:
```bash
./scripts/check_code_quality.sh --fix
```

**Solution 2** - Update hooks:
```bash
poetry run pre-commit autoupdate
poetry run pre-commit install
```

**Solution 3** - Skip hooks (not recommended):
```bash
git commit --no-verify
```

### Tests failing with "No module named pytest"

**Solution**: Install dev dependencies:
```bash
poetry install --with dev
```

### K-Index returns NaN

**Causes**:
- Empty arrays
- Arrays with all zeros
- Mismatched array lengths
- NaN values in input data

**Solution**: Validate input data:
```python
import numpy as np

# Check for empty
assert len(observed) > 0, "observed array is empty"

# Check for NaN
assert not np.any(np.isnan(observed)), "observed contains NaN"

# Check lengths match
assert len(observed) == len(actual), "array length mismatch"
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more common issues.

---

## Performance & Scalability

### How fast is K-Index computation?

**Benchmarks** (M1 MacBook Pro):
- N=100: ~8ms
- N=1,000: ~45ms
- N=10,000: ~450ms
- N=100,000: ~4.5s

**Scalability**: O(n) - linear in sample size

Run benchmarks yourself:
```bash
make benchmarks
```

### How do I optimize performance?

1. **Use NumPy vectorization** - Avoid Python loops
2. **Parallelize** - Use `n_cores` parameter
3. **Reduce bootstrap iterations** - Use 100 instead of 1000 for quick tests
4. **Profile** - Find bottlenecks with cProfile:
   ```bash
   python -m cProfile -o profile.stats your_script.py
   poetry run snakeviz profile.stats
   ```

See [benchmarks/README.md](benchmarks/README.md) for performance guides.

### Can I run on a cluster?

Yes! Use:

**Docker** (single node):
```bash
docker-compose up
```

**Slurm** (HPC cluster):
```bash
sbatch scripts/slurm_job.sh  # Coming soon
```

**Kubernetes** (planned):
```bash
kubectl apply -f k8s/  # Future release
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment guides.

### What's the maximum simulation size?

**Limits** (8GB RAM):
- Universes: 1,000+ parallel
- Timesteps: 100,000+
- Parameter sweep: 10,000+ configurations
- Multi-agent: 1,000+ agents

**For larger simulations**, use:
- Incremental saving (checkpoints)
- Distributed execution (Dask/Ray)
- Chunked processing

---

## Security & Privacy

### How do I report a security vulnerability?

**DO NOT** open a public issue. Instead:

1. Email: kosmic-lab-security@example.org (or use your configured email)
2. Or use GitHub Security Advisory: https://github.com/Luminous-Dynamics/kosmic-lab/security/advisories/new

Include:
- Description of vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if any)

**Response SLA**:
- Initial response: 48 hours
- Fix timeline: 7-30 days depending on severity

See [SECURITY.md](SECURITY.md) for details.

### Is my experiment data secure?

**Local data**: You control all experiment data locally. Nothing is shared unless you explicitly publish to Mycelix DHT.

**K-Codex records**: Contain only:
- Configuration parameters (no raw data)
- Metrics (aggregated statistics)
- Environment details (non-sensitive)

**Best practices**:
- Never commit `.env` files (use `.env.example`)
- Don't include credentials in K-Codex metadata
- Review `.gitignore` patterns

### Does Kosmic Lab phone home?

**No**. Kosmic Lab does not send any telemetry or usage data. All computation is local unless you:
- Publish to Mycelix DHT (opt-in)
- Push to GitHub (manual)
- Deploy to cloud services (manual)

---

## Advanced Topics

### How does bioelectric rescue work?

Bioelectric rescue simulates Michael Levin-inspired interventions:

1. **Detection**: FEP (Free Energy Principle) error exceeds threshold
2. **Intervention**: Apply bioelectric stimulation
3. **Stabilization**: Restore consciousness coherence
4. **Validation**: Measure K-Index improvement

See [examples/04_bioelectric_rescue.py](examples/04_bioelectric_rescue.py) and [docs/fre_design_doc.md](docs/fre_design_doc.md).

### What is Mycelix integration?

Mycelix is a decentralized knowledge graph on Holochain DHT that enables:
- **Federated learning**: Train on global data without sharing raw data
- **Solver network**: Competitive experiment proposals
- **Verifiable provenance**: Cryptographic integrity
- **Civilization layer**: Track ecosystem-wide coherence

See [docs/MYCELIX_INTEGRATION_ARCHITECTURE.md](docs/MYCELIX_INTEGRATION_ARCHITECTURE.md).

### How do I extend the FRE simulation?

1. Subclass `UniverseSimulator`:
```python
from fre.universe import UniverseSimulator

class MySimulator(UniverseSimulator):
    def custom_step(self, timestep: int) -> None:
        # Your custom logic
        pass
```

2. Override `run()` method or add hooks
3. Register custom metrics
4. Document in [docs/](docs/)

See [ARCHITECTURE.md](ARCHITECTURE.md#module-structure) for design patterns.

### Can I use this for my research paper?

**Yes!** Kosmic Lab is designed for publication-ready research:

1. **Reproducibility**: K-Codex ensures bit-for-bit reproducibility
2. **Validation**: 48+ tests validate correctness
3. **Documentation**: Comprehensive docs for methods section
4. **Preregistration**: OSF integration for study preregistration

**Citation**:
```bibtex
@software{kosmic_lab_2025,
  title = {Kosmic Lab: A Unified Platform for Consciousness Research},
  author = {Luminous Dynamics},
  year = {2025},
  url = {https://github.com/Luminous-Dynamics/kosmic-lab},
  version = {1.1.0}
}
```

See [docs/paper-drafts/](docs/paper-drafts/) for publication examples.

---

## Still have questions?

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/kosmic-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/kosmic-lab/discussions)
- **Email**: kosmic-lab@example.org (or your configured contact)

---

*This FAQ is maintained by the Kosmic Lab community. Last updated: 2025-11-15.*
