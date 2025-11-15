# Core Module

Shared infrastructure and utilities for Kosmic Lab.

## Overview

The `core` module provides fundamental building blocks used throughout the project:

- **Logging**: Centralized logging configuration
- **K-Codex**: Reproducibility tracking system
- **Bioelectric**: Bioelectric circuit simulation
- **K-Pass**: Multi-universe passage tracking
- **Utilities**: Common helper functions

## Modules

### `logging_config.py`

Centralized logging configuration with colored output and file logging.

**Usage**:
```python
from core.logging_config import setup_logging, get_logger

# Setup logging once at application start
setup_logging(level="INFO", log_file="logs/experiment.log", colored=True)

# Get logger in any module
logger = get_logger(__name__)
logger.info("Experiment started")
logger.debug(f"Processing {len(data)} samples")
```

**Features**:
- Colored console output (with color codes)
- File logging with rotation
- Customizable log levels
- Module-specific loggers

---

### `kcodex.py`

K-Codex reproducibility tracking system (formerly K-Passport).

**Usage**:
```python
from core.kcodex import KCodexWriter

kcodex = KCodexWriter("logs/experiment_kcodex.json")
kcodex.log_experiment(
    experiment_name="baseline_experiment",
    params={"n_samples": 100, "threshold": 0.5},
    metrics={"k_index": 0.875, "accuracy": 0.95},
    seed=42,
    extra_metadata={"notes": "Baseline run"}
)
```

**Features**:
- Git SHA tracking (automatic)
- Configuration hashing
- Timestamp recording
- Environment details (Python version, platform)
- Extensible metadata

**K-Codex Record Contains**:
- Experiment name and timestamp
- Git commit SHA
- Configuration hash
- Parameters and metrics
- Random seed
- Environment information
- Optional metadata and notes

---

### `bioelectric.py`

Bioelectric circuit simulation inspired by Michael Levin's work.

**Usage**:
```python
from core.bioelectric import BioelectricCircuit

# Create circuit
circuit = BioelectricCircuit(
    n_cells=100,
    resting_voltage=-70.0,  # mV
)

# Apply stimulus
circuit.apply_stimulus(amplitude=0.5, duration=10)

# Step simulation
for t in range(100):
    circuit.step(dt=0.01)

# Get state
voltages = circuit.get_voltages()
currents = circuit.get_currents()
```

**Features**:
- Membrane voltage dynamics
- Gap junction coupling
- Ion channel simulation
- Stimulus application
- Pattern formation

---

### `kpass.py`

Multi-universe passage tracking (K-Pass system).

**Usage**:
```python
from core.kpass import KPassManager

manager = KPassManager()

# Record universe transition
manager.record_passage(
    from_universe="universe_A",
    to_universe="universe_B",
    passage_metrics={"k_index": 0.8, "harmony": 0.9},
    timestamp=datetime.now()
)

# Query passages
passages = manager.get_passages_for_universe("universe_A")
```

**Features**:
- Universe transition tracking
- Passage metrics recording
- Temporal sequencing
- Query capabilities

---

### `utils.py`

Common utility functions used across modules.

**Usage**:
```python
from core.utils import (
    infer_git_sha,
    hash_config,
    bootstrap_confidence_interval
)

# Get current git commit SHA
git_sha = infer_git_sha()

# Hash configuration for reproducibility
config = {"param1": 1, "param2": 2}
config_hash = hash_config(config)

# Bootstrap confidence interval
import numpy as np
data = np.random.randn(100)
ci_lower, ci_upper = bootstrap_confidence_interval(
    data,
    statistic=np.mean,
    n_bootstrap=1000,
    confidence_level=0.95
)
```

**Functions**:
- `infer_git_sha()`: Get current git commit SHA
- `hash_config()`: Hash configuration for reproducibility
- `bootstrap_confidence_interval()`: Compute bootstrap CI for any statistic
- `ensure_dir()`: Create directory if it doesn't exist
- `save_json()`: Save data to JSON with proper formatting
- `load_json()`: Load data from JSON

---

## Design Principles

### 1. Separation of Concerns

Each module has a single, well-defined purpose:
- Logging handles all logging
- K-Codex handles all reproducibility
- Bioelectric handles all circuit simulation
- Utils provides cross-cutting utilities

### 2. Reproducibility First

All modules support reproducibility:
- Logging preserves execution traces
- K-Codex tracks all experiments
- Utils provides hashing and git tracking
- Random operations accept seeds

### 3. Type Safety

All modules use comprehensive type hints:
```python
def my_function(param: int, optional: Optional[str] = None) -> Dict[str, Any]:
    ...
```

### 4. Error Handling

All modules provide clear error messages:
```python
if len(data) == 0:
    raise ValueError(
        "Data array is empty. Expected at least 1 element."
    )
```

### 5. Documentation

All public functions have Google-style docstrings:
```python
def my_function(param1: int, param2: str) -> bool:
    """
    One-line summary.

    Detailed description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description

    Raises:
        ValueError: When ...

    Example:
        >>> result = my_function(42, "test")
        >>> print(result)
        True
    """
```

## Common Patterns

### Pattern 1: Logging Setup

```python
from core.logging_config import setup_logging, get_logger

# At module/script start
setup_logging(level="INFO", log_file="logs/experiment.log")
logger = get_logger(__name__)

# Throughout code
logger.info("Starting processing")
logger.debug(f"Processing item {i}")
logger.warning("Unusual condition detected")
logger.error(f"Operation failed: {error}")
```

### Pattern 2: Experiment Tracking

```python
from core.kcodex import KCodexWriter
from core.utils import infer_git_sha

# Create K-Codex writer
kcodex = KCodexWriter("logs/experiment.json")

# Run experiment
params = {"n_samples": 100}
results = run_experiment(params, seed=42)

# Log to K-Codex
kcodex.log_experiment(
    experiment_name="my_experiment",
    params=params,
    metrics=results,
    seed=42
)
```

### Pattern 3: Configuration Hashing

```python
from core.utils import hash_config

config = {
    "model": "baseline",
    "params": {"lr": 0.01, "epochs": 100},
    "seed": 42
}

config_hash = hash_config(config)
print(f"Config hash: {config_hash}")  # Deterministic
```

## Testing

All core modules have comprehensive tests:

```bash
# Run core module tests
pytest tests/test_utils.py
pytest tests/test_logging_config.py
pytest tests/test_kcodex.py
pytest tests/test_bioelectric.py
```

See `tests/` directory for examples.

## API Reference

For detailed API documentation, see:
- `docs/api/core.rst` - Sphinx API documentation
- Module docstrings - In-code documentation
- `examples/` - Usage examples

## Contributing

When adding to core:

1. **Single responsibility**: One module, one purpose
2. **Type hints**: All functions need type hints
3. **Documentation**: Google-style docstrings
4. **Tests**: Comprehensive test coverage
5. **Reproducibility**: Support deterministic behavior
6. **Error handling**: Clear, helpful error messages

See `CONTRIBUTING.md` for detailed guidelines.

---

**Last Updated**: 2025-11-15
