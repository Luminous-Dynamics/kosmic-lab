# Contributing to Kosmic Lab

Thank you for your interest in contributing to Kosmic Lab! We welcome contributions from researchers, developers, and anyone interested in advancing consciousness research through AI-accelerated science.

This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

### Our Standards

We are committed to providing a welcoming and inclusive environment. We expect all contributors to:

- **Be respectful** and considerate in communication
- **Be collaborative** and open to feedback
- **Focus on what's best** for the scientific community
- **Show empathy** towards other community members
- **Be patient** with newcomers and questions

### Unacceptable Behavior

- Harassment, discrimination, or exclusionary behavior
- Trolling, insulting comments, or personal attacks
- Publishing others' private information
- Other conduct inappropriate for a professional setting

If you experience or witness unacceptable behavior, please contact the maintainers.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.10+** installed
- **Poetry 1.7+** for dependency management
- **Git** for version control
- **GitHub account** for pull requests

### Environment Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/kosmic-lab.git
cd kosmic-lab

# 2. Install dependencies
poetry install

# 3. Install pre-commit hooks
poetry run pre-commit install

# 4. Verify installation
make test

# 5. Run an example
poetry run python examples/01_hello_kosmic.py
```

### Understanding the Codebase

Before making changes, familiarize yourself with:

- **README.md** - Project overview and quick start
- **ARCHITECTURE.md** - System design and structure
- **DEVELOPMENT.md** - Detailed development guide
- **examples/** - Working code examples

## How to Contribute

There are many ways to contribute to Kosmic Lab:

### 1. Report Bugs

Found a bug? Help us fix it!

- Check [existing issues](https://github.com/Luminous-Dynamics/kosmic-lab/issues)
- If not reported, create a new issue using the bug report template
- Include:
  - Clear description of the bug
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version)
  - K-Codex information if applicable

### 2. Suggest Features

Have an idea for a new feature?

- Check if it's already suggested in issues
- Create a feature request using the template
- Explain:
  - The problem it solves
  - Proposed solution
  - Alternative approaches considered
  - Impact on existing functionality

### 3. Improve Documentation

Documentation improvements are always welcome!

- Fix typos and clarify confusing sections
- Add more examples and tutorials
- Improve API documentation
- Write guides for specific use cases

### 4. Write Code

Ready to contribute code?

- **Bug fixes** - See "good first issue" labels
- **New features** - Discuss in an issue first
- **Performance improvements** - Include benchmarks
- **Tests** - Help improve coverage
- **Examples** - Add new tutorial examples

### 5. Review Pull Requests

Help review open pull requests:

- Test the changes locally
- Provide constructive feedback
- Check code quality and tests
- Verify documentation updates

## Development Workflow

### 1. Create an Issue

For any significant change:

1. Search existing issues
2. Create a new issue if needed
3. Discuss approach with maintainers
4. Get consensus before starting work

### 2. Fork and Branch

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/kosmic-lab.git
cd kosmic-lab

# Add upstream remote
git remote add upstream https://github.com/Luminous-Dynamics/kosmic-lab.git

# Create a feature branch
git checkout -b feature/my-awesome-feature
```

### 3. Make Changes

Follow our coding standards (see below) and:

```bash
# Write code
vim core/my_new_module.py

# Write tests
vim tests/test_my_new_module.py

# Format code
make format

# Run tests
make test

# Check types
make type-check

# Run full CI locally
make ci-local
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit (pre-commit hooks will run)
git commit -m "feat: add new feature description"
```

#### Commit Message Convention

Use semantic commit messages:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or fixes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes (formatting)
- `chore:` - Build/tooling changes
- `ci:` - CI/CD changes

Examples:
```
feat: add K-Lag analysis for time series data
fix: correct K-Index computation for edge cases
docs: improve quick start guide in README
test: add integration tests for bioelectric rescue
perf: optimize bootstrap CI calculation
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/my-awesome-feature

# Create pull request on GitHub
```

## Coding Standards

### Style Guide

We follow **PEP 8** with these specifics:

- **Line length**: 88 characters (Black default)
- **Import order**: Standard library → Third party → Local (enforced by isort)
- **Type hints**: Required for all functions
- **Docstrings**: Google style for all public APIs

### Code Quality Tools

All code must pass:

```bash
make format      # Black + isort formatting
make lint        # All linters
make type-check  # mypy type checking
make test        # All tests
```

### Type Hints

All functions require comprehensive type hints:

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


def compute_metrics(
    data: NDArray[np.float64],
    threshold: float = 0.5,
    *,
    normalize: bool = True,
    weights: Optional[NDArray[np.float64]] = None,
) -> Dict[str, Any]:
    """Compute statistical metrics from data."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """
    One-line summary of function purpose.

    Detailed explanation of what the function does, including
    any important details about behavior, edge cases, etc.

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
    """
    ...
```

### Error Handling

Use specific exceptions with helpful messages:

```python
if len(data) == 0:
    raise ValueError(
        "Data array is empty. Expected at least 1 element."
    )

if not 0 <= threshold <= 1:
    raise ValueError(
        f"Threshold {threshold} out of range. Must be in [0, 1]."
    )
```

### Logging

Use structured logging:

```python
from core.logging_config import get_logger

logger = get_logger(__name__)

# Use appropriate levels
logger.debug(f"Processing {len(data)} samples")
logger.info("Experiment started successfully")
logger.warning(f"Unusual value detected: {value}")
logger.error(f"Computation failed: {error}")
```

## Testing Requirements

### Test Coverage

- **Minimum**: 80% coverage for new code
- **Target**: 90%+ coverage overall
- **Critical paths**: 100% coverage

### Writing Tests

All new features must include tests:

```python
import pytest
import numpy as np
from fre.metrics.k_index import k_index


class TestKIndex:
    """Test suite for K-Index computation."""

    def test_basic_computation(self):
        """Test basic K-Index computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k = k_index(data, data)
        assert k == pytest.approx(1.0, abs=0.01)

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            k_index(np.array([[1, 2]]), np.array([1, 2]))

    @pytest.mark.parametrize("n_samples", [10, 50, 100, 500])
    def test_scaling(self, n_samples: int):
        """Test performance scaling."""
        data = np.random.randn(n_samples)
        k = k_index(data, data)
        assert isinstance(k, float)
```

### Running Tests

```bash
# All tests
make test

# Specific test file
poetry run pytest tests/test_k_index.py

# With coverage
make coverage

# Fast fail
poetry run pytest -x
```

## Documentation

### Code Documentation

- All public functions/classes need docstrings
- Include usage examples in docstrings
- Document all parameters and return values
- List all exceptions that can be raised

### User Documentation

When adding features:

1. **Update README.md** if it changes quickstart
2. **Add examples** to `examples/` directory
3. **Update ARCHITECTURE.md** for architectural changes
4. **Create guides** in `docs/` if needed
5. **Update CHANGELOG.md** with your changes

### Building Docs

```bash
# Build documentation
make docs

# Serve locally
make docs-serve  # Opens at localhost:8000

# Check for warnings
cd docs && make html
```

## Pull Request Process

### Before Submitting

Checklist:

- [ ] Code follows style guide
- [ ] All tests pass: `make test`
- [ ] Type checks pass: `make type-check`
- [ ] Code is formatted: `make format`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts with main
- [ ] Commits are clean and descriptive

### PR Description

Include in your PR:

1. **What** - What does this PR do?
2. **Why** - Why is this change needed?
3. **How** - How does it work?
4. **Testing** - How was it tested?
5. **Screenshots** - For UI changes
6. **Breaking changes** - List any breaking changes
7. **Related issues** - Link related issues

Example:
```markdown
## Summary
Adds K-Lag analysis for temporal correlations in time series data.

## Motivation
Users need to identify temporal delays between observed and predicted
consciousness metrics. Closes #42.

## Changes
- Added `fre/metrics/k_lag.py` with k_lag function
- Added comprehensive tests in `tests/test_k_lag.py`
- Added example usage in `examples/02_advanced_k_index.py`
- Updated CHANGELOG.md

## Testing
- All tests pass with 95% coverage
- Benchmarked on arrays up to N=10,000
- Validated against manual calculations

## Breaking Changes
None
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Maintainer review** required
3. **Address feedback** promptly
4. **Squash commits** if requested
5. **Merge** when approved

## Issue Guidelines

### Bug Reports

Use the bug report template and include:

- **Description**: Clear and concise bug description
- **Reproduction**: Minimal steps to reproduce
- **Expected**: What should happen
- **Actual**: What actually happens
- **Environment**: OS, Python version, dependencies
- **K-Codex**: Include K-Codex record if applicable
- **Logs**: Relevant log excerpts

### Feature Requests

Use the feature request template and include:

- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Other approaches considered
- **Impact**: Who benefits and how?
- **Implementation**: Ideas for implementation

### Questions

For questions:

- Check **documentation** first
- Search **existing issues**
- Check **examples/** directory
- See **TROUBLESHOOTING.md**
- If still stuck, create an issue with "question" label

## Community

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **Pull Requests** - Code review and discussion
- **Discussions** - General questions and ideas

### Getting Help

- **Examples**: Start with `examples/01_hello_kosmic.py`
- **Documentation**: Run `make docs-serve`
- **Troubleshooting**: See `TROUBLESHOOTING.md`
- **Development**: See `DEVELOPMENT.md`
- **Quick Reference**: See `QUICK_REFERENCE.md`

### Recognition

Contributors are recognized in:

- **CHANGELOG.md** - For significant contributions
- **GitHub insights** - Automatic contribution tracking
- **Project documentation** - For major features

## License

By contributing to Kosmic Lab, you agree that your contributions will be licensed under the same license as the project.

---

## Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/kosmic-lab.git
cd kosmic-lab

# 2. Setup
poetry install
poetry run pre-commit install

# 3. Create branch
git checkout -b feature/my-feature

# 4. Make changes
# ... edit code ...

# 5. Test
make format
make test
make type-check

# 6. Commit
git commit -m "feat: add new feature"

# 7. Push and PR
git push origin feature/my-feature
# Create PR on GitHub
```

## Questions?

If you have questions about contributing:

- Check this guide and other documentation
- Search existing issues
- Create a new issue with your question
- Maintainers are happy to help!

---

**Thank you for contributing to Kosmic Lab!** 🌊✨

We're excited to work with you to advance consciousness research through AI-accelerated science!
