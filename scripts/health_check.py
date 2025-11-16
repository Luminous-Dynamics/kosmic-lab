#!/usr/bin/env python3
"""
Kosmic Lab Health Check - Validate installation and system health

This script performs a comprehensive health check of the Kosmic Lab installation:
1. Python version check
2. Dependency checks
3. Import tests
4. Core functionality tests
5. Performance smoke tests
6. File system checks

Usage:
    python scripts/health_check.py

    or

    make health-check

Author: Kosmic Lab Team
"""

import sys
import importlib
from pathlib import Path
import platform

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class HealthCheck:
    """Comprehensive health check for Kosmic Lab."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []

    def check(self, name: str, func, critical: bool = True):
        """
        Run a health check.

        Args:
            name: Check name
            func: Function to run
            critical: If True, failure is critical
        """
        try:
            func()
            self.checks_passed += 1
            print(f"{Colors.GREEN}✓{Colors.RESET} {name}")
            return True
        except Exception as e:
            if critical:
                self.checks_failed += 1
                print(f"{Colors.RED}✗{Colors.RESET} {name}: {e}")
            else:
                self.warnings.append(f"{name}: {e}")
                print(f"{Colors.YELLOW}⚠{Colors.RESET} {name}: {e}")
            return False

    def check_python_version(self):
        """Check Python version."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            raise RuntimeError(f"Python 3.9+ required, got {version.major}.{version.minor}")

    def check_dependencies(self):
        """Check that all required dependencies are installed."""
        required_deps = [
            'numpy',
            'scipy',
            'pytest',
            'jsonschema',
        ]

        for dep in required_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                raise ImportError(f"Missing required dependency: {dep}")

    def check_optional_dependencies(self):
        """Check optional dependencies."""
        optional_deps = {
            'matplotlib': 'Visualization support',
            'plotly': 'Interactive plots',
            'jupyter': 'Notebook support',
        }

        missing = []
        for dep, purpose in optional_deps.items():
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(f"{dep} ({purpose})")

        if missing:
            raise Warning(f"Missing optional: {', '.join(missing)}")

    def check_core_imports(self):
        """Check that core modules can be imported."""
        try:
            from core.logging_config import setup_logging, get_logger
            from core.kcodex import KCodexWriter
            from fre.metrics.k_index import k_index, bootstrap_k_ci
            from fre.metrics.k_lag import k_lag
        except ImportError as e:
            raise ImportError(f"Core import failed: {e}")

    def check_k_index_computation(self):
        """Test basic K-Index computation."""
        import numpy as np
        from fre.metrics.k_index import k_index

        # Simple test
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        act = np.array([1.1, 2.1, 2.9, 4.1, 5.0])

        k = k_index(obs, act)

        if not (0.0 <= k <= 3.0):  # Reasonable range
            raise ValueError(f"K-Index value unreasonable: {k}")

    def check_bootstrap_ci(self):
        """Test bootstrap CI computation."""
        import numpy as np
        from fre.metrics.k_index import bootstrap_k_ci

        np.random.seed(42)
        obs = np.random.randn(100)
        act = np.random.randn(100)

        k, ci_low, ci_high = bootstrap_k_ci(obs, act, n_bootstrap=10, seed=42)

        if not (ci_low <= k <= ci_high):
            raise ValueError("K-Index not within confidence interval")

    def check_kcodex_logging(self):
        """Test K-Codex logging."""
        from core.kcodex import KCodexWriter
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            kcodex = KCodexWriter(f.name)
            kcodex.log_experiment(
                experiment_name="health_check",
                params={"test": True},
                metrics={"k": 1.0},
                seed=42
            )

            # Verify file exists
            path = Path(f.name)
            if not path.exists():
                raise FileNotFoundError("K-Codex file not created")

            # Clean up
            path.unlink()

    def check_performance(self):
        """Quick performance smoke test."""
        import numpy as np
        import time
        from fre.metrics.k_index import k_index

        np.random.seed(42)
        obs = np.random.randn(10000)
        act = np.random.randn(10000)

        start = time.time()
        k = k_index(obs, act)
        elapsed = time.time() - start

        # Should complete in <100ms
        if elapsed > 0.1:
            raise Warning(f"Performance degraded: {elapsed*1000:.1f}ms for N=10k")

    def check_directories(self):
        """Check that expected directories exist."""
        required_dirs = [
            'core',
            'fre',
            'examples',
            'tests',
            'benchmarks',
            'scripts',
        ]

        for dirname in required_dirs:
            path = project_root / dirname
            if not path.exists():
                raise FileNotFoundError(f"Directory missing: {dirname}")

    def check_examples_exist(self):
        """Check that examples exist."""
        examples_dir = project_root / 'examples'
        examples = list(examples_dir.glob('*.py'))

        # Filter out __init__.py
        examples = [e for e in examples if e.name != '__init__.py']

        if len(examples) < 5:
            raise Warning(f"Few examples found: {len(examples)}")

    def run_all_checks(self):
        """Run all health checks."""
        print(f"\n{Colors.BOLD}{'='*70}")
        print(f"🏥 KOSMIC LAB HEALTH CHECK")
        print(f"{'='*70}{Colors.RESET}\n")

        print(f"{Colors.BOLD}System Information:{Colors.RESET}")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {platform.system()} {platform.release()}")
        print(f"  Architecture: {platform.machine()}")
        print()

        print(f"{Colors.BOLD}Running Checks:{Colors.RESET}\n")

        # Critical checks
        self.check("Python version (3.9+)", self.check_python_version, critical=True)
        self.check("Required dependencies", self.check_dependencies, critical=True)
        self.check("Core module imports", self.check_core_imports, critical=True)
        self.check("Project directories", self.check_directories, critical=True)

        print()

        # Functionality checks
        self.check("K-Index computation", self.check_k_index_computation, critical=True)
        self.check("Bootstrap CI", self.check_bootstrap_ci, critical=True)
        self.check("K-Codex logging", self.check_kcodex_logging, critical=True)

        print()

        # Performance and optional checks
        self.check("Performance (smoke test)", self.check_performance, critical=False)
        self.check("Optional dependencies", self.check_optional_dependencies, critical=False)
        self.check("Examples exist", self.check_examples_exist, critical=False)

        # Summary
        print(f"\n{Colors.BOLD}{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}{Colors.RESET}\n")

        total = self.checks_passed + self.checks_failed
        print(f"✅ {Colors.GREEN}Passed:{Colors.RESET} {self.checks_passed}/{total}")

        if self.checks_failed > 0:
            print(f"❌ {Colors.RED}Failed:{Colors.RESET} {self.checks_failed}/{total}")

        if self.warnings:
            print(f"⚠️  {Colors.YELLOW}Warnings:{Colors.RESET} {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   • {warning}")

        print()

        # Overall status
        if self.checks_failed == 0:
            print(f"{Colors.GREEN}{'='*70}")
            print(f"✅ SYSTEM HEALTHY - All critical checks passed!")
            print(f"{'='*70}{Colors.RESET}\n")

            if self.warnings:
                print(f"{Colors.YELLOW}Note: Some optional features unavailable (see warnings above){Colors.RESET}\n")

            return True
        else:
            print(f"{Colors.RED}{'='*70}")
            print(f"❌ SYSTEM UNHEALTHY - {self.checks_failed} critical check(s) failed!")
            print(f"{'='*70}{Colors.RESET}\n")

            print(f"{Colors.BOLD}Troubleshooting:{Colors.RESET}")
            print(f"  1. Run: poetry install --sync")
            print(f"  2. Check: poetry run python --version")
            print(f"  3. See: TROUBLESHOOTING.md")
            print(f"  4. See: INSTALL.md")
            print()

            return False


def main():
    """Main execution."""
    checker = HealthCheck()
    healthy = checker.run_all_checks()

    # Exit code
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
