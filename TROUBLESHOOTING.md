# Troubleshooting Guide

**Quick Links**: [Installation](#installation-issues) | [Testing](#testing-issues) | [Experiments](#experiment-issues) | [Performance](#performance-issues) | [Reproducibility](#reproducibility-issues) | [CI/CD](#cicd-issues)

---

## Installation Issues

### Problem: Poetry install fails

**Symptoms**:
```bash
$ poetry install
Error: No module named 'poetry.core'
```

**Solutions**:

1. **Update Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   poetry --version  # Should be 1.7.1+
   ```

2. **Clear cache and retry**:
   ```bash
   poetry cache clear . --all
   poetry install --sync
   ```

3. **Use specific Python version**:
   ```bash
   poetry env use python3.11
   poetry install
   ```

### Problem: Dependency conflicts

**Symptoms**:
```
SolverProblemError: Because package-a depends on package-b (>=2.0)...
```

**Solutions**:

1. **Update lock file**:
   ```bash
   poetry lock --no-update
   poetry install
   ```

2. **Fresh environment**:
   ```bash
   poetry env remove python
   poetry install --sync
   ```

### Problem: Import errors after installation

**Symptoms**:
```python
ModuleNotFoundError: No module named 'core'
```

**Solutions**:

1. **Ensure you're in poetry shell or using poetry run**:
   ```bash
   poetry shell
   python your_script.py
   # OR
   poetry run python your_script.py
   ```

2. **Check PYTHONPATH**:
   ```bash
   export PYTHONPATH=/home/user/kosmic-lab:$PYTHONPATH
   ```

3. **Verify installation**:
   ```bash
   poetry run python -c "import core; import fre; print('✅ Imports work!')"
   ```

---

## Testing Issues

### Problem: Tests fail with "No module named 'pytest'"

**Solution**:
```bash
poetry install --with dev
poetry run pytest
```

### Problem: Import errors in tests

**Symptoms**:
```
ImportError: attempted relative import with no known parent package
```

**Solutions**:

1. **Run tests with poetry**:
   ```bash
   poetry run pytest
   ```

2. **Run from project root**:
   ```bash
   cd /home/user/kosmic-lab
   poetry run pytest tests/
   ```

### Problem: Test coverage fails

**Symptoms**:
```
coverage.py warning: No data was collected
```

**Solutions**:

1. **Install coverage dependencies**:
   ```bash
   poetry install --with dev
   poetry run pytest --cov=core --cov=fre
   ```

2. **Check source paths**:
   ```bash
   # Verify pyproject.toml has correct paths
   poetry run pytest --cov=. --cov-report=term
   ```

---

## Experiment Issues

### Problem: K-Codex validation fails

**Symptoms**:
```
KCodexError: K-Codex validation failed: 'metrics' is a required property
```

**Solutions**:

1. **Check schema file exists**:
   ```bash
   ls schemas/k_passport.json
   # OR
   ls schemas/k_codex.json
   ```

2. **Verify JSON structure**:
   ```python
   import json
   with open('logs/experiment.json') as f:
       data = json.load(f)
   assert 'metrics' in data, "Missing metrics field"
   assert 'K' in data['metrics'], "Missing K-index"
   ```

3. **Use KCodexWriter correctly**:
   ```python
   from core.kcodex import KCodexWriter
   from pathlib import Path

   writer = KCodexWriter(Path("schemas/k_codex.json"))
   codex = writer.build_record(
       experiment="my_experiment",
       params={"lr": 0.001},
       estimators={"method": "pearson"},
       metrics={"K": 1.23},  # ← Must include metrics!
       seed=42
   )
   ```

### Problem: Simulation produces NaN values

**Symptoms**:
```
Warning: K-Index = nan
```

**Solutions**:

1. **Check observation/action arrays**:
   ```python
   assert not np.any(np.isnan(obs_norms)), "NaN in observations"
   assert not np.any(np.isnan(act_norms)), "NaN in actions"
   assert len(obs_norms) > 1, "Need at least 2 samples"
   ```

2. **Verify data ranges**:
   ```python
   from core.utils import validate_bounds
   validate_bounds(k_value, 0.0, 2.0, "K-index")
   ```

3. **Check for zero division**:
   ```python
   from core.utils import safe_divide
   ratio = safe_divide(numerator, denominator, default=0.0)
   ```

### Problem: No corridor hits (corridor_rate = 0%)

**Symptoms**:
```
corridor_hits: 0
corridor_rate: 0.0
```

**Solutions**:

1. **Check threshold**:
   ```yaml
   # In config file
   corridor_threshold: 1.0  # Lower if too strict
   ```

2. **Inspect K-Index distribution**:
   ```python
   import pandas as pd
   df = pd.read_csv('logs/summary.csv')
   print(df['K'].describe())
   # If max K < threshold, adjust parameters
   ```

3. **Review parameter ranges**:
   ```yaml
   parameters:
     learning_rate: [0.0001, 0.001, 0.01]  # Broader sweep
     gamma: [0.9, 0.95, 0.99]
   ```

---

## Performance Issues

### Problem: Simulations running too slow

**Symptoms**:
- Taking hours instead of minutes
- High CPU usage
- Memory growing

**Solutions**:

1. **Reduce parameter sweep size**:
   ```yaml
   parameters:
     learning_rate: [0.001]  # Start with single value
     gamma: [0.99]
   seeds_per_point: 1  # Reduce from 10
   ```

2. **Use vectorized operations**:
   ```python
   # Instead of loops
   result = np.array([compute(x) for x in data])
   # Use vectorization
   result = np.vectorize(compute)(data)
   ```

3. **Profile code**:
   ```bash
   poetry run python -m cProfile -o profile.stats your_script.py
   poetry run python -m pstats profile.stats
   ```

4. **Check for memory leaks**:
   ```bash
   poetry run python -m memory_profiler your_script.py
   ```

### Problem: Dashboard not loading

**Symptoms**:
```
Dash is running on http://127.0.0.1:8050/
But browser shows "Unable to connect"
```

**Solutions**:

1. **Check port availability**:
   ```bash
   lsof -i :8050
   # If occupied, use different port
   make dashboard PORT=8051
   ```

2. **Verify data exists**:
   ```bash
   ls logs/fre_phase1/*.json
   # Should show K-Codex files
   ```

3. **Check browser console** for JavaScript errors

---

## Reproducibility Issues

### Problem: Can't reproduce experiment from K-Codex

**Symptoms**:
- Different results with same K-Codex
- Git SHA doesn't match

**Solutions**:

1. **Verify git checkout**:
   ```bash
   git rev-parse HEAD  # Should match K-Codex 'commit' field
   git checkout <sha-from-kcodex>
   ```

2. **Check config hash**:
   ```python
   from core.config import ConfigBundle
   config = ConfigBundle.from_dict(params)
   assert config.sha256 == kcodex['config_hash']
   ```

3. **Use exact seed**:
   ```python
   seed = kcodex['seed']  # Don't use different seed!
   np.random.seed(seed)
   ```

4. **Verify Python version**:
   ```bash
   python --version  # Should match original environment
   ```

### Problem: Git SHA shows "unknown"

**Symptoms**:
```json
{
  "commit": "unknown"
}
```

**Solutions**:

1. **Ensure you're in git repository**:
   ```bash
   git status  # Should work
   ```

2. **Check .git directory exists**:
   ```bash
   ls -la .git/
   ```

3. **Use subprocess fallback**:
   ```python
   from core.utils import infer_git_sha
   sha = infer_git_sha()  # Uses subprocess as backup
   ```

---

## CI/CD Issues

### Problem: GitHub Actions failing

**Symptoms**:
- Red X on commits
- Email notifications of failures

**Solutions**:

1. **Run CI locally first**:
   ```bash
   make ci-local
   # Fix any issues before pushing
   ```

2. **Check specific job**:
   - Go to GitHub Actions tab
   - Click on failed workflow
   - Expand failed job to see errors

3. **Common CI failures**:

   **Black formatting**:
   ```bash
   make format
   git add -u
   git commit --amend --no-edit
   ```

   **Type checking**:
   ```bash
   make type-check
   # Fix reported issues
   ```

   **Tests**:
   ```bash
   poetry run pytest -x  # Stop on first failure
   # Debug and fix
   ```

### Problem: Pre-commit hooks blocking commit

**Symptoms**:
```
black....................................................................Failed
```

**Solutions**:

1. **Let hooks auto-fix**:
   ```bash
   poetry run pre-commit run --all-files
   git add -u
   git commit
   ```

2. **Skip hooks temporarily** (not recommended):
   ```bash
   git commit --no-verify
   ```

3. **Update hooks**:
   ```bash
   poetry run pre-commit autoupdate
   ```

---

## Mycelix Integration Issues

### Problem: Holochain bridge fails

**Symptoms**:
```
Error: Holochain conductor not running
```

**Solutions**:

1. **Check Holochain installation**:
   ```bash
   holochain --version
   ```

2. **Start conductor**:
   ```bash
   cd holochain/
   holochain -c conductor-config.yaml
   ```

3. **Verify network connectivity**:
   ```bash
   curl http://localhost:8888
   ```

---

## Common Error Messages

### "FileNotFoundError: No passport JSON files found"

**Cause**: No experiments have been run yet

**Solution**:
```bash
make fre-run  # Run experiments first
make fre-summary  # Then analyze
```

### "ValueError: K-index = 3.5 is outside valid range [0.0, 2.0]"

**Cause**: Algorithm error or data corruption

**Solution**:
1. Check correlation calculation
2. Verify input data is normalized
3. Add assertion in code:
   ```python
   assert 0 <= k <= 2, f"Invalid K: {k}"
   ```

### "Permission denied: logs/"

**Cause**: Insufficient permissions

**Solution**:
```bash
mkdir -p logs
chmod 755 logs
```

### "OSError: [Errno 24] Too many open files"

**Cause**: File handle leak

**Solution**:
```bash
ulimit -n 4096  # Increase limit
# Fix code to close files:
with open(path) as f:
    data = f.read()  # Auto-closes
```

---

## Getting Help

If you still have issues after trying these solutions:

1. **Check existing issues**: https://github.com/Luminous-Dynamics/kosmic-lab/issues
2. **Search discussions**: https://github.com/Luminous-Dynamics/kosmic-lab/discussions
3. **Create new issue**: Use our templates
4. **Include**:
   - Error messages (full stack trace)
   - Environment info (OS, Python version, poetry version)
   - Minimal reproduction code
   - K-Codex JSON (if relevant)
   - What you've already tried

---

## Quick Diagnostics

Run this script to check your environment:

```bash
poetry run python -c "
import sys
import numpy as np
import pandas as pd
from core.utils import infer_git_sha
from core.kcodex import KCodexWriter
print('✅ Python:', sys.version)
print('✅ NumPy:', np.__version__)
print('✅ Pandas:', pd.__version__)
print('✅ Git SHA:', infer_git_sha())
print('✅ All imports successful!')
"
```

Expected output:
```
✅ Python: 3.11.x
✅ NumPy: 1.26.x
✅ Pandas: 2.1.x
✅ Git SHA: <40-char-hash>
✅ All imports successful!
```

---

**Last Updated**: 2025-11-14
**For More Help**: See [ARCHITECTURE.md](ARCHITECTURE.md) for system details
