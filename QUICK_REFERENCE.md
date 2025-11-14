# Quick Reference Cheatsheet

Fast reference for common Kosmic Lab commands and patterns.

## Essential Commands

```bash
# Setup
poetry install                  # Install dependencies
poetry shell                    # Activate environment
make init                       # Full setup with pre-commit

# Development
make test                       # Run tests
make format                     # Auto-format code
make lint                       # Run all linters
make type-check                 # Type checking
make coverage                   # Test coverage
make ci-local                   # Full CI pipeline locally

# Documentation
make docs                       # Build docs
make docs-serve                 # Serve docs at localhost:8000
make docs-clean                 # Clean docs build

# Performance
make benchmarks                 # Run benchmarks
make benchmarks-save            # Save benchmark results

# Running Experiments
python examples/01_hello_kosmic.py                    # Basic tutorial
python examples/02_advanced_k_index.py                # Advanced K-Index
python examples/03_multi_universe.py --parallel       # Multi-universe
python examples/04_bioelectric_rescue.py              # Bioelectric rescue

# Dashboard
make dashboard                  # Launch monitoring dashboard
```

## Code Patterns

### Imports

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.logging_config import setup_logging, get_logger
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index, bootstrap_k_ci
```

### Logging Setup

```python
from core.logging_config import setup_logging, get_logger

# Setup once at module start
setup_logging(level="INFO", log_file="logs/experiment.log")
logger = get_logger(__name__)

# Use throughout code
logger.info("Starting experiment")
logger.debug(f"Parameter value: {param}")
logger.warning("Unusual condition detected")
logger.error(f"Failed: {error}")
```

### K-Index Computation

```python
import numpy as np
from fre.metrics.k_index import k_index, bootstrap_k_ci

# Basic K-Index
observed = np.abs(np.random.randn(100))
actual = np.abs(np.random.randn(100))
k = k_index(observed, actual)
print(f"K-Index: {k:.4f}")

# With confidence interval
ci_lower, ci_upper = bootstrap_k_ci(
    observed, actual,
    n_bootstrap=1000,
    confidence_level=0.95
)
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

### K-Lag Analysis

```python
from fre.metrics.k_lag import k_lag

# Analyze temporal lags
results = k_lag(observed, actual, max_lag=20)

print(f"Best lag: {results['best_lag']}")
print(f"K-Index at best lag: {results['k_at_best_lag']:.4f}")
print(f"Correlations: {results['correlations']}")
```

### K-Codex Tracking

```python
from core.kcodex import KCodexWriter

kcodex = KCodexWriter("logs/experiment.json")
kcodex.log_experiment(
    experiment_name="my_experiment",
    params={
        "n_samples": 100,
        "threshold": 0.5,
        "learning_rate": 0.01
    },
    metrics={
        "k_index": 0.875,
        "accuracy": 0.95,
        "loss": 0.05
    },
    seed=42,
    extra_metadata={
        "notes": "Baseline run",
        "dataset": "synthetic"
    }
)
```

### Universe Simulation

```python
from fre.simulate import UniverseSimulator, compute_metrics

# Create simulator
sim = UniverseSimulator(n_agents=50, seed=42)

# Define parameters
params = {
    "consciousness": 0.7,
    "coherence": 0.8,
    "fep": 0.5
}

# Run simulation
for t in range(100):
    sim.step(params)

# Get metrics
metrics = compute_metrics(params, seed=42)
print(f"Harmony: {metrics['harmony']:.4f}")
```

### Bioelectric Rescue

```python
from core.bioelectric import BioelectricCircuit
from fre.rescue import (
    detect_consciousness_collapse,
    apply_bioelectric_rescue
)

# Detect collapse
fep_error = 0.7
coherence = 0.3
collapsed = detect_consciousness_collapse(fep_error, coherence)

if collapsed:
    # Apply rescue
    correction = apply_bioelectric_rescue(
        current_voltage=-50.0,
        target_voltage=-70.0,
        fep_error=fep_error,
        momentum=0.9
    )
    print(f"Correction: {correction:.2f} mV")
```

### Bootstrap Confidence Intervals

```python
from core.utils import bootstrap_confidence_interval
import numpy as np

data = np.random.randn(100)

# CI for any statistic
ci_lower, ci_upper = bootstrap_confidence_interval(
    data,
    statistic=np.mean,  # or np.median, np.std, etc.
    n_bootstrap=1000,
    confidence_level=0.95
)

print(f"Mean: {np.mean(data):.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
```

## Testing Patterns

### Basic Test

```python
import pytest
import numpy as np
from fre.metrics.k_index import k_index

def test_k_index_basic():
    """Test basic K-Index computation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = k_index(data, data)
    assert k == pytest.approx(1.0, abs=0.01)
```

### Parametrized Test

```python
@pytest.mark.parametrize("n_samples,expected_time", [
    (100, 0.01),
    (1000, 0.1),
    (10000, 1.0),
])
def test_scaling(n_samples, expected_time):
    """Test performance scaling."""
    import time
    data = np.random.randn(n_samples)

    start = time.time()
    result = k_index(data, data)
    elapsed = time.time() - start

    assert elapsed < expected_time
```

### Test Fixtures

```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Generate sample data for tests."""
    rng = np.random.default_rng(42)
    return rng.random(100)

# Use in test
def test_with_fixture(sample_data):
    """Test using fixture."""
    assert len(sample_data) == 100
```

## Git Workflows

### Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/my-feature

# Make changes, commit
git add .
git commit -m "Add new feature: description"

# Push to remote
git push -u origin feature/my-feature

# Create pull request on GitHub
```

### Bug Fix

```bash
# Create fix branch
git checkout -b fix/issue-123

# Write test that fails
# Fix the bug
# Verify test passes

git commit -m "Fix #123: Description of fix"
git push -u origin fix/issue-123
```

### Update from Main

```bash
# Fetch latest changes
git fetch origin

# Merge main into your branch
git checkout feature/my-feature
git merge origin/main

# Or rebase
git rebase origin/main
```

## File Structure

```
kosmic-lab/
├── core/                      # Core infrastructure
│   ├── logging_config.py     # Centralized logging
│   ├── kcodex.py             # K-Codex system
│   ├── bioelectric.py        # Bioelectric circuits
│   └── utils.py              # Shared utilities
├── fre/                       # Free Energy Rescue
│   ├── metrics/
│   │   ├── k_index.py        # K-Index computation
│   │   └── k_lag.py          # K-Lag analysis
│   ├── simulate.py           # Universe simulation
│   ├── rescue.py             # Rescue mechanisms
│   └── analyze.py            # Batch analysis
├── scripts/                   # User-facing scripts
│   ├── kosmic_dashboard.py   # Monitoring dashboard
│   └── ai_experiment_designer.py
├── examples/                  # Tutorial examples
│   ├── 01_hello_kosmic.py
│   ├── 02_advanced_k_index.py
│   ├── 03_multi_universe.py
│   └── 04_bioelectric_rescue.py
├── tests/                     # Test suite
├── benchmarks/                # Performance benchmarks
├── docs/                      # Sphinx documentation
└── logs/                      # Experiment logs (gitignored)
```

## Common Issues

### Import Errors

```bash
# Ensure in project root
cd /path/to/kosmic-lab

# Activate environment
poetry shell

# Install dependencies
poetry install
```

### Test Failures

```bash
# Run specific test
poetry run pytest tests/test_k_index.py::test_k_index_basic -v

# Stop on first failure
poetry run pytest -x

# Show print statements
poetry run pytest -s
```

### Type Errors

```bash
# Check specific file
poetry run mypy core/logging_config.py

# Ignore specific error
# Add to code:
# type: ignore[error-code]
```

### Format Issues

```bash
# Auto-fix formatting
make format

# Check what would change
poetry run black --check --diff core/
```

## Performance Tips

```python
# ✅ Good: Vectorized
result = np.sum(array)

# ❌ Bad: Loop
result = sum(x for x in array)

# ✅ Good: In-place
array *= 2

# ❌ Bad: Creates new array
array = array * 2

# ✅ Good: Generator
total = sum(x**2 for x in range(1000000))

# ❌ Bad: List
total = sum([x**2 for x in range(1000000)])
```

## Type Hints

```python
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# Basic types
def func(x: int, y: str) -> bool:
    pass

# Collections
def func(items: List[int]) -> Dict[str, float]:
    pass

# Optional
def func(x: Optional[int] = None) -> str:
    pass

# NumPy arrays
def func(data: NDArray[np.float64]) -> NDArray[np.float64]:
    pass

# Union
def func(x: Union[int, str]) -> Any:
    pass

# Callable
from typing import Callable
def func(callback: Callable[[int], str]) -> None:
    pass
```

## Logging Levels

```python
logger.debug("Detailed diagnostic info")      # Development only
logger.info("General informational message")  # Normal operations
logger.warning("Warning about potential issue")  # Needs attention
logger.error("Error occurred, operation failed")  # Serious problem
logger.critical("Critical error, system unstable")  # Immediate action
```

## Environment Variables

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Python unbuffered output
export PYTHONUNBUFFERED=1

# Disable warnings
export PYTHONWARNINGS="ignore"
```

## Useful One-Liners

```bash
# Find all TODOs
grep -r "TODO" --include="*.py" core/ fre/

# Count lines of code
find . -name "*.py" -not -path "./.*" | xargs wc -l

# Find large files
find . -type f -size +1M

# List test coverage
poetry run pytest --cov=core --cov-report=term-missing

# Profile script
python -m cProfile -o profile.stats script.py

# Format imports
poetry run isort --diff core/

# Find unused imports
poetry run pylint core/ --disable=all --enable=unused-import
```

---

## Quick Links

- **Examples**: `examples/01_hello_kosmic.py` - Start here
- **Tests**: `make test` - Run all tests
- **Docs**: `make docs-serve` - View documentation
- **Dashboard**: `make dashboard` - Real-time monitoring
- **Architecture**: `ARCHITECTURE.md` - System design
- **Contributing**: `CONTRIBUTING.md` - How to contribute
- **Development**: `DEVELOPMENT.md` - Full dev guide
- **Troubleshooting**: `TROUBLESHOOTING.md` - Common issues

---

**Print this and keep it handy!** 📋✨
