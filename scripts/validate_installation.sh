#!/bin/bash
#
# Comprehensive Installation Validation Script
# Validates that Kosmic Lab is correctly installed and configured
#
# Usage: ./scripts/validate_installation.sh [--verbose] [--strict]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Options
VERBOSE=false
STRICT=false

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --strict|-s)
            STRICT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--strict]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v    Show detailed output"
            echo "  --strict, -s     Fail on warnings"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}ℹ${NC} $1"
    fi
}

log_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
    if [ "$STRICT" = true ]; then
        exit 1
    fi
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
    if [ "$STRICT" = true ]; then
        exit 1
    fi
}

section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Start validation
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Kosmic Lab Installation Validator v1.1.0          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# 1. System Requirements
section "1. System Requirements"

log_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        log_pass "Python $PYTHON_VERSION (≥3.10 required)"
    else
        log_fail "Python $PYTHON_VERSION (≥3.10 required, found $PYTHON_VERSION)"
    fi
else
    log_fail "Python 3 not found"
fi

log_info "Checking Poetry..."
if command -v poetry &> /dev/null; then
    POETRY_VERSION=$(poetry --version | awk '{print $3}' | tr -d '()')
    log_pass "Poetry $POETRY_VERSION installed"
else
    log_fail "Poetry not found (install from https://python-poetry.org/)"
fi

log_info "Checking Git..."
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    log_pass "Git $GIT_VERSION installed"
else
    log_fail "Git not found"
fi

# 2. Project Structure
section "2. Project Structure"

REQUIRED_DIRS=(
    "core"
    "fre"
    "fre/metrics"
    "tests"
    "scripts"
    "examples"
    "docs"
    "schemas"
    "benchmarks"
)

log_info "Checking required directories..."
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        log_pass "Directory exists: $dir/"
    else
        log_fail "Missing directory: $dir/"
    fi
done

REQUIRED_FILES=(
    "pyproject.toml"
    "README.md"
    "CHANGELOG.md"
    "LICENSE"
    ".gitignore"
    ".pre-commit-config.yaml"
    ".editorconfig"
    "Makefile"
)

log_info "Checking required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_pass "File exists: $file"
    else
        log_fail "Missing file: $file"
    fi
done

# 3. Python Environment
section "3. Python Environment"

log_info "Checking virtual environment..."
if poetry env info &> /dev/null; then
    VENV_PATH=$(poetry env info --path)
    log_pass "Virtual environment: $VENV_PATH"
else
    log_warn "No virtual environment found (run: poetry install)"
fi

log_info "Checking Python dependencies..."
if poetry run python -c "import numpy" 2>/dev/null; then
    log_pass "Core dependencies installed (numpy)"
else
    log_fail "Dependencies not installed (run: poetry install)"
fi

# 4. Core Modules
section "4. Core Modules Import Test"

CORE_MODULES=(
    "core.logging_config"
    "core.kcodex"
    "core.utils"
    "fre.metrics.k_index"
    "fre.metrics.k_lag"
)

log_info "Testing module imports..."
for module in "${CORE_MODULES[@]}"; do
    if poetry run python -c "import $module" 2>/dev/null; then
        log_pass "Import successful: $module"
    else
        log_fail "Import failed: $module"
    fi
done

# 5. K-Index Computation
section "5. K-Index Computation Test"

log_info "Testing K-Index computation..."
K_INDEX_TEST=$(poetry run python -c "
import numpy as np
from fre.metrics.k_index import k_index

rng = np.random.default_rng(42)
x = rng.random(100)
y = rng.random(100)
k = k_index(x, y)
assert 0 <= k <= 1, f'K-Index out of range: {k}'
print(f'{k:.4f}')
" 2>&1)

if [ $? -eq 0 ]; then
    log_pass "K-Index computation: $K_INDEX_TEST"
else
    log_fail "K-Index computation failed: $K_INDEX_TEST"
fi

# 6. Git Configuration
section "6. Git Configuration"

log_info "Checking Git repository..."
if git rev-parse --git-dir &> /dev/null; then
    log_pass "Git repository initialized"

    # Check for uncommitted changes
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        log_pass "Working directory clean"
    else
        log_warn "Uncommitted changes in working directory"
    fi
else
    log_warn "Not a Git repository"
fi

# 7. Pre-commit Hooks
section "7. Pre-commit Hooks"

log_info "Checking pre-commit installation..."
if [ -f ".git/hooks/pre-commit" ]; then
    log_pass "Pre-commit hooks installed"
else
    log_warn "Pre-commit hooks not installed (run: poetry run pre-commit install)"
fi

# 8. Documentation
section "8. Documentation"

DOCS=(
    "README.md"
    "QUICKSTART.md"
    "DEVELOPMENT.md"
    "CONTRIBUTING.md"
    "CHANGELOG.md"
    "FAQ.md"
    "DEPLOYMENT.md"
    "SECURITY.md"
)

log_info "Checking documentation files..."
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        log_pass "Documentation: $doc"
    else
        log_warn "Missing documentation: $doc"
    fi
done

# 9. Examples
section "9. Examples"

EXAMPLES=(
    "examples/01_hello_kosmic.py"
    "examples/02_advanced_k_index.py"
    "examples/03_multi_universe.py"
    "examples/04_bioelectric_rescue.py"
)

log_info "Checking example files..."
for example in "${EXAMPLES[@]}"; do
    if [ -f "$example" ]; then
        log_pass "Example: $example"
    else
        log_warn "Missing example: $example"
    fi
done

# 10. Configuration Files
section "10. Configuration Files"

log_info "Checking .env.example..."
if [ -f ".env.example" ]; then
    log_pass "Environment template: .env.example"

    if [ -f ".env" ]; then
        log_pass "Environment file: .env (configured)"
    else
        log_warn "Environment file .env not found (copy from .env.example)"
    fi
else
    log_fail "Missing .env.example"
fi

log_info "Checking JSON schemas..."
if [ -f "schemas/kcodex.schema.json" ] && [ -f "schemas/experiment_config.schema.json" ]; then
    log_pass "JSON schemas present"
else
    log_warn "Some JSON schemas missing"
fi

# 11. Tests
section "11. Test Suite"

log_info "Checking test files..."
TEST_COUNT=$(find tests -name "test_*.py" | wc -l)
if [ "$TEST_COUNT" -gt 0 ]; then
    log_pass "Found $TEST_COUNT test files"
else
    log_warn "No test files found"
fi

log_info "Running quick test..."
if poetry run pytest tests/ -q --collect-only &> /dev/null; then
    TEST_TOTAL=$(poetry run pytest tests/ -q --collect-only 2>&1 | tail -1 | awk '{print $1}')
    log_pass "Test suite: $TEST_TOTAL tests collected"
else
    log_warn "Could not collect tests (dependencies may be missing)"
fi

# 12. Makefile Targets
section "12. Makefile Targets"

log_info "Checking Makefile targets..."
if [ -f "Makefile" ]; then
    TARGETS=$(grep "^[a-zA-Z_-]*:" Makefile | cut -d: -f1 | wc -l)
    log_pass "Makefile with $TARGETS targets"
else
    log_fail "Makefile not found"
fi

# Summary
section "Summary"

echo ""
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${RED}Failed:${NC}   $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

# Final verdict
if [ "$FAILED" -eq 0 ]; then
    if [ "$WARNINGS" -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ Installation validated successfully!                   ║${NC}"
        echo -e "${GREEN}║    Kosmic Lab is ready to use.                            ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        exit 0
    else
        echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  ⚠ Installation validated with warnings                   ║${NC}"
        echo -e "${YELLOW}║    Review warnings above for optimal setup.               ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
        exit 0
    fi
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ✗ Installation validation failed                          ║${NC}"
    echo -e "${RED}║    Fix errors above before using Kosmic Lab.               ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Quick fixes:"
    echo "  1. Install dependencies: poetry install"
    echo "  2. Install pre-commit: poetry run pre-commit install"
    echo "  3. Create .env: cp .env.example .env"
    echo "  4. Run tests: make test"
    exit 1
fi
