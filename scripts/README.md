# Scripts

Utility scripts and automation tools for Kosmic Lab.

## Overview

This directory contains helper scripts for common development and operational tasks. All scripts are designed to be run from the project root directory.

## Available Scripts

### Development Scripts

#### `setup_dev_env.sh`

**Purpose**: Automated development environment setup

**Usage**:
```bash
./scripts/setup_dev_env.sh
```

**What it does**:
- Checks prerequisites (Python, Poetry, Git)
- Installs dependencies with Poetry
- Sets up pre-commit hooks
- Creates `.env` file from template
- Creates necessary directories
- Runs tests to verify setup
- Displays helpful next steps

**Time**: ~2 minutes

---

#### `check_code_quality.sh`

**Purpose**: Comprehensive code quality validation

**Usage**:
```bash
# Check only
./scripts/check_code_quality.sh

# Auto-fix formatting issues
./scripts/check_code_quality.sh --fix

# Fail on first error
./scripts/check_code_quality.sh --strict
```

**Checks performed** (8 total):
1. Black formatting
2. isort import sorting
3. flake8 linting
4. mypy type checking
5. bandit security scanning
6. pytest unit tests
7. Sphinx documentation build
8. Common issues (TODO, print statements, large files)

**Time**: ~30 seconds

---

### Analysis Scripts

#### `kosmic_dashboard.py`

**Purpose**: Real-time monitoring dashboard

**Usage**:
```bash
# Launch with defaults
poetry run python scripts/kosmic_dashboard.py

# Custom configuration
poetry run python scripts/kosmic_dashboard.py --logdir logs --port 8050

# Or use Makefile
make dashboard
```

**Features**:
- Live monitoring with 5-second auto-refresh
- Interactive parameter exploration
- Export publication figures
- Team collaboration via shared URL

**Port**: 8050 (default)

---

#### `ai_experiment_designer.py`

**Purpose**: AI-powered experiment suggestions

**Usage**:
```bash
# Get suggestions
poetry run python scripts/ai_experiment_designer.py \
    --hypothesis "Test bioelectric rescue effectiveness" \
    --output experiments/ai_designed.py

# Or use Makefile
make ai-suggest
```

**Features**:
- Bayesian optimization for experiment design
- 70% fewer experiments needed
- Transfer learning from historical data
- Uncertainty quantification

---

#### `generate_analysis_notebook.py`

**Purpose**: Auto-generate analysis Jupyter notebooks

**Usage**:
```bash
# Generate notebook
poetry run python scripts/generate_analysis_notebook.py \
    --logdir logs \
    --output analysis/auto_analysis.ipynb

# Or use Makefile
make notebook
```

**Features**:
- 2 hours → 30 seconds for analysis
- Statistical summaries
- Publication-ready plots
- LaTeX snippets
- Completely reproducible

---

#### `fre_analyzer.py`

**Purpose**: Batch FRE experiment analysis

**Usage**:
```bash
poetry run python scripts/fre_analyzer.py --logdir logs --output results.json
```

**Features**:
- Batch processing of experiment logs
- Aggregated statistics
- Trend analysis
- Export results to JSON

---

### Integration Scripts

#### `holochain_bridge.py`

**Purpose**: Mycelix/Holochain DHT integration

**Usage**:
```bash
# Publish K-Codex records to DHT
poetry run python scripts/holochain_bridge.py --publish logs

# Query corridor passports
poetry run python scripts/holochain_bridge.py --query --min-k 1.0 --max-k 1.5

# Verify passport integrity
poetry run python scripts/holochain_bridge.py --verify QmXXX

# Or use Makefile
make holochain-publish
make holochain-query
make holochain-verify HASH=QmXXX
```

**Features**:
- Decentralized storage on DHT
- Verifiable provenance
- Federated learning support
- Immutable audit trail

---

## Script Conventions

### Location

All scripts should be in the `scripts/` directory.

### Naming

- Use snake_case: `my_script.py`
- Be descriptive: `generate_analysis_notebook.py` not `gen_nb.py`
- Add `.sh` extension for shell scripts

### Structure

Python scripts should include:

```python
#!/usr/bin/env python3
"""
Script description.

Usage:
    poetry run python scripts/my_script.py [OPTIONS]

Examples:
    poetry run python scripts/my_script.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--option", help="Option description")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info("Script started")

    # Script logic here

    logger.info("Script completed")


if __name__ == "__main__":
    main()
```

### Documentation

Each script should have:

- Docstring with description
- Usage examples
- Argument documentation
- Entry in this README

### Error Handling

Scripts should:

- Exit with code 0 on success
- Exit with code 1 on error
- Log errors clearly
- Provide helpful error messages

## Adding New Scripts

1. **Create the script** in `scripts/`
2. **Make it executable** (if shell script): `chmod +x scripts/my_script.sh`
3. **Add shebang**: `#!/usr/bin/env python3` or `#!/usr/bin/env bash`
4. **Write docstring**: Describe purpose and usage
5. **Add to Makefile** (if commonly used)
6. **Document here**: Add section to this README
7. **Test thoroughly**: Ensure it works from project root

## Testing Scripts

Test scripts before committing:

```bash
# Run from project root
./scripts/my_script.sh

# With various options
./scripts/my_script.sh --option1 --option2

# Error cases
./scripts/my_script.sh --invalid-option
```

## Makefile Integration

Common scripts should have Makefile targets:

```makefile
my-task:  # Description of task
	poetry run python scripts/my_script.py
```

See `Makefile` for examples.

## Shell Scripts Best Practices

For shell scripts (`.sh`):

1. **Use strict mode**:
   ```bash
   set -e  # Exit on error
   set -u  # Exit on undefined variable
   ```

2. **Add help text**:
   ```bash
   if [[ "${1:-}" == "--help" ]]; then
       echo "Usage: $0 [OPTIONS]"
       exit 0
   fi
   ```

3. **Use functions**:
   ```bash
   function do_something() {
       local arg=$1
       echo "Doing something with $arg"
   }
   ```

4. **Color output**:
   ```bash
   RED='\033[0;31m'
   GREEN='\033[0;32m'
   NC='\033[0m'  # No Color
   echo -e "${GREEN}Success${NC}"
   ```

5. **Check prerequisites**:
   ```bash
   if ! command -v python3 &> /dev/null; then
       echo "Python 3 is required"
       exit 1
   fi
   ```

## Python Scripts Best Practices

For Python scripts (`.py`):

1. **Use argparse** for command-line arguments
2. **Setup logging** early in main()
3. **Use type hints** throughout
4. **Follow project code style** (Black, isort)
5. **Handle KeyboardInterrupt** gracefully
6. **Provide progress indicators** for long-running tasks
7. **Use pathlib.Path** for file operations
8. **Validate inputs** before processing

## Resources

- **argparse**: https://docs.python.org/3/library/argparse.html
- **logging**: https://docs.python.org/3/library/logging.html
- **pathlib**: https://docs.python.org/3/library/pathlib.html
- **Bash best practices**: https://google.github.io/styleguide/shellguide.html

---

**Last Updated**: 2025-11-15
