# Development Guide

Complete guide for developing and contributing to Kosmic Lab.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance](#performance)
- [Debugging](#debugging)
- [Common Workflows](#common-workflows)
- [Best Practices](#best-practices)
- [Release Process](#release-process)

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone repository
git clone https://github.com/Luminous-Dynamics/kosmic-lab.git
cd kosmic-lab

# 2. Install dependencies
poetry install

# 3. Activate environment
poetry shell

# 4. Run tests
make test

# 5. Try an example
python examples/01_hello_kosmic.py
```

## Development Environment

### Prerequisites

**Required:**
- Python 3.10 or higher
- Poetry 1.7+ (`curl -sSL https://install.python-poetry.org | python3 -`)
- Git

**Optional but recommended:**
- pyenv (Python version management)
- pre-commit (automatic code quality checks)
- VS Code with Python extension

### Initial Setup

```bash
# Install dependencies
poetry install

# Install pre-commit hooks (auto-format on commit)
poetry run pre-commit install

# Verify installation
make test
```

### Virtual Environment

Poetry manages the virtual environment automatically:

```bash
# Activate shell
poetry shell

# Run commands without activating
poetry run python examples/01_hello_kosmic.py
poetry run pytest

# Show environment info
poetry env info
```

### IDE Configuration

#### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Black Formatter
- isort

Settings (`.vscode/settings.json`):
```json
{
  "python.formatting.provider": "black",
  "python.linting.mypyEnabled": true,
  "python.linting.enabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

1. Open project in PyCharm
2. File → Settings → Project → Python Interpreter
3. Select Poetry environment
4. Enable "Black" formatter in Tools → Black
5. Enable "mypy" in Tools → External Tools

## Code Style

### Style Guide

We follow **PEP 8** with these additions:

- Line length: **88 characters** (Black default)
- Import sorting: **isort** with Black compatibility
- Type hints: **Required** for all functions
- Docstrings: **Google style** for all public APIs

### Formatting

Auto-format with Black and isort:

```bash
# Format all code
make format

# Check formatting without changing
poetry run black --check core/ fre/ scripts/ tests/
```

### Type Checking

We use mypy for static type checking:

```bash
# Run type checks
make type-check

# Type check specific module
poetry run mypy core/logging_config.py
```

### Linting

```bash
# Run all linters
make lint

# Individual linters
poetry run black --check .
poetry run isort --check-only .
poetry run flake8 core/ fre/
```

### Example: Well-Styled Function

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def compute_metrics(
    data: np.ndarray,
    threshold: float = 0.5,
    *,
    normalize: bool = True,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute statistical metrics from data array.

    This function computes mean, std, and threshold-based metrics
    with optional normalization and weighting.

    Args:
        data: Input data array
        threshold: Threshold for binary classification
        normalize: Whether to normalize output metrics
        weights: Optional weights for weighted metrics

    Returns:
        Dictionary containing:
            - mean: Mean value
            - std: Standard deviation
            - above_threshold: Count above threshold

    Raises:
        ValueError: If data is empty or threshold out of range

    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> metrics = compute_metrics(data, threshold=3.0)
        >>> print(metrics["mean"])
        3.0
    """
    if len(data) == 0:
        raise ValueError("Data array cannot be empty")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    # Compute basic statistics
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))

    # Apply weights if provided
    if weights is not None:
        if len(weights) != len(data):
            raise ValueError("Weights must match data length")
        mean_val = float(np.average(data, weights=weights))

    # Count above threshold
    above_count = int(np.sum(data > threshold))

    results = {
        "mean": mean_val,
        "std": std_val,
        "above_threshold": above_count,
    }

    # Normalize if requested
    if normalize and std_val > 0:
        results["mean"] = results["mean"] / std_val

    return results
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Verbose output
make test-verbose

# With coverage
make coverage

# Specific test file
poetry run pytest tests/test_k_index.py

# Specific test function
poetry run pytest tests/test_k_index.py::test_k_index_basic

# Fast fail (stop on first failure)
poetry run pytest -x

# Run last failed tests
poetry run pytest --lf
```

### Writing Tests

Test structure:

```
tests/
├── core/
│   ├── test_logging_config.py
│   ├── test_kcodex.py
│   └── test_bioelectric.py
├── fre/
│   ├── test_k_index.py
│   ├── test_k_lag.py
│   └── test_simulate.py
└── conftest.py  # Shared fixtures
```

Example test:

```python
"""Tests for K-Index computation."""

from __future__ import annotations

import numpy as np
import pytest

from fre.metrics.k_index import k_index, bootstrap_k_ci


class TestKIndex:
    """Test suite for K-Index metrics."""

    def test_k_index_perfect_correlation(self):
        """Test K-Index with perfectly correlated data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k = k_index(data, data)
        assert k == pytest.approx(1.0, abs=0.01)

    def test_k_index_no_correlation(self):
        """Test K-Index with uncorrelated data."""
        rng = np.random.default_rng(42)
        obs = rng.random(100)
        act = rng.random(100)
        k = k_index(obs, act)
        assert -0.5 < k < 0.5  # Expect near zero

    def test_k_index_validation(self):
        """Test input validation for K-Index."""
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            k_index(np.array([[1, 2]]), np.array([1, 2]))

        with pytest.raises(ValueError, match="must have same length"):
            k_index(np.array([1, 2]), np.array([1, 2, 3]))

    @pytest.mark.parametrize("n_samples", [10, 50, 100, 500])
    def test_k_index_scaling(self, n_samples: int):
        """Test K-Index computation scales with sample size."""
        rng = np.random.default_rng(42)
        obs = rng.random(n_samples)
        act = rng.random(n_samples)

        # Should complete quickly
        import time
        start = time.time()
        k = k_index(obs, act)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be fast
        assert isinstance(k, float)


def test_bootstrap_ci():
    """Test bootstrap confidence intervals."""
    rng = np.random.default_rng(42)
    obs = rng.random(100)
    act = rng.random(100)

    ci_lower, ci_upper = bootstrap_k_ci(obs, act, n_bootstrap=100)

    assert ci_lower < ci_upper
    assert isinstance(ci_lower, float)
    assert isinstance(ci_upper, float)
```

### Test Fixtures

Use fixtures for common test data:

```python
# conftest.py
import pytest
import numpy as np


@pytest.fixture
def random_data():
    """Generate random test data."""
    rng = np.random.default_rng(42)
    return rng.random(100)


@pytest.fixture
def correlated_data():
    """Generate correlated test data."""
    rng = np.random.default_rng(42)
    x = rng.random(100)
    y = x + rng.random(100) * 0.1  # High correlation
    return x, y
```

### Coverage Goals

- **Target**: 90%+ coverage
- **Minimum**: 80% for new code
- **Critical paths**: 100% coverage

```bash
# Generate coverage report
make coverage

# View in browser
open htmlcov/index.html
```

## Documentation

### Docstrings

All public functions/classes need Google-style docstrings:

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """
    One-line summary.

    Detailed description with multiple paragraphs if needed.
    Explain what the function does, not how it does it.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to "default".

    Returns:
        True if successful, False otherwise.

    Raises:
        ValueError: If param1 is negative
        TypeError: If param2 is not a string

    Example:
        >>> result = my_function(42, "test")
        >>> print(result)
        True

    Note:
        Additional notes or caveats.

    See Also:
        related_function: Similar functionality
    """
    pass
```

### Building Documentation

```bash
# Build HTML docs
make docs

# Serve docs locally
make docs-serve  # Opens http://localhost:8000

# Clean docs
make docs-clean
```

### Adding Documentation

1. Write comprehensive docstrings
2. Add module to `docs/api/`
3. Build and verify: `make docs`
4. Check for warnings

## Performance

### Benchmarking

```bash
# Run benchmarks
make benchmarks

# Save results
make benchmarks-save
```

### Profiling

```python
# Profile with cProfile
python -m cProfile -o profile.stats examples/02_advanced_k_index.py

# Visualize with snakeviz
poetry add --group dev snakeviz
poetry run snakeviz profile.stats
```

### Performance Tips

1. **Use NumPy vectorization** instead of loops
2. **Profile before optimizing** - measure first
3. **Cache expensive computations** with functools.lru_cache
4. **Use appropriate data structures** - sets for membership, dicts for lookups
5. **Minimize I/O** - batch file operations

## Debugging

### Logging

```python
from core.logging_config import get_logger

logger = get_logger(__name__)

# Use appropriate levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning - potential issue")
logger.error("Error occurred")
logger.critical("Critical error - system unstable")
```

### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in (Python 3.7+)
breakpoint()
```

### VS Code Debugging

Launch configuration (`.vscode/launch.json`):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Python: Run Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v"],
      "console": "integratedTerminal"
    }
  ]
}
```

## Common Workflows

### Adding a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Write code with tests
vim core/my_new_module.py
vim tests/test_my_new_module.py

# 3. Format and check
make format
make lint
make test

# 4. Commit (pre-commit hooks run automatically)
git add .
git commit -m "Add new feature: description"

# 5. Push and create PR
git push origin feature/my-new-feature
```

### Fixing a Bug

```bash
# 1. Write failing test that demonstrates bug
vim tests/test_bugfix.py

# 2. Verify test fails
make test

# 3. Fix the bug
vim core/module_with_bug.py

# 4. Verify test passes
make test

# 5. Commit with issue reference
git commit -m "Fix #123: Description of bug fix"
```

### Running an Experiment

```bash
# 1. Create experiment script
cp examples/01_hello_kosmic.py experiments/my_experiment.py

# 2. Modify parameters
vim experiments/my_experiment.py

# 3. Run experiment
poetry run python experiments/my_experiment.py

# 4. Analyze results
python scripts/fre_analyzer.py --logdir logs/my_experiment

# 5. Visualize
make dashboard
```

## Best Practices

### Code Organization

```python
# Standard library imports
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from core.logging_config import get_logger
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index
```

### Error Handling

```python
# Good: Specific exceptions with helpful messages
def process_data(data: np.ndarray) -> float:
    if len(data) == 0:
        raise ValueError(
            "Data array is empty. Expected at least 1 element."
        )

    if np.any(np.isnan(data)):
        raise ValueError(
            "Data contains NaN values. Use np.nan_to_num() to handle."
        )

    try:
        result = compute_expensive_operation(data)
    except RuntimeError as e:
        raise RuntimeError(
            f"Computation failed: {e}. Check data range and dtype."
        ) from e

    return result
```

### Reproducibility

```python
# Always use seeds for randomness
rng = np.random.default_rng(seed=42)

# Track experiments with K-Codex
kcodex = KCodexWriter("logs/experiment.json")
kcodex.log_experiment(
    experiment_name="my_experiment",
    params=params,
    metrics=metrics,
    seed=42
)

# Log git SHA
from core.utils import infer_git_sha
sha = infer_git_sha()
logger.info(f"Running experiment on commit {sha}")
```

### Resource Management

```python
# Use context managers
from pathlib import Path

# Good
with open("data.json") as f:
    data = json.load(f)

# Better - use Path
data_path = Path("data.json")
with data_path.open() as f:
    data = json.load(f)
```

## Release Process

### Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

```bash
# Update version in pyproject.toml
poetry version patch  # 1.0.0 → 1.0.1
poetry version minor  # 1.0.1 → 1.1.0
poetry version major  # 1.1.0 → 2.0.0
```

### Creating a Release

```bash
# 1. Update CHANGELOG.md
vim CHANGELOG.md

# 2. Bump version
poetry version minor

# 3. Commit version bump
git add pyproject.toml CHANGELOG.md
git commit -m "Release v1.1.0"

# 4. Tag release
git tag -a v1.1.0 -m "Release version 1.1.0"

# 5. Push
git push origin main --tags

# 6. Create GitHub release from tag
```

### Pre-release Checklist

- [ ] All tests passing: `make test`
- [ ] Code formatted: `make format`
- [ ] Type checks pass: `make type-check`
- [ ] Docs build: `make docs`
- [ ] Benchmarks run: `make benchmarks`
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] No security issues: `make security-check`

---

## Getting Help

- **Documentation**: See `docs/` and run `make docs`
- **Examples**: Check `examples/` directory
- **Troubleshooting**: See `TROUBLESHOOTING.md`
- **Contributing**: See `CONTRIBUTING.md`
- **Architecture**: See `ARCHITECTURE.md`

---

**Happy Developing!** 🚀✨
