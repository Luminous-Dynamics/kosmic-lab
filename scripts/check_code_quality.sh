#!/usr/bin/env bash
# Comprehensive code quality check script
# Runs all quality checks: formatting, linting, type checking, security

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED++))
}

failure() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED++))
}

run_check() {
    local name=$1
    shift
    local cmd="$@"

    echo ""
    info "Running: $name"
    echo "Command: $cmd"
    echo "---"

    if eval "$cmd"; then
        success "$name passed"
        return 0
    else
        failure "$name failed"
        return 1
    fi
}

# Banner
echo "================================================"
echo "  Kosmic Lab Code Quality Check"
echo "================================================"
echo ""

# Parse arguments
FIX=false
STRICT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX=true
            shift
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fix      Auto-fix issues where possible"
            echo "  --strict   Exit on first failure"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Configure based on flags
if [ "$STRICT" = true ]; then
    set -e  # Already set, but being explicit
fi

# 1. Black formatting
if [ "$FIX" = true ]; then
    run_check "Black formatting (auto-fix)" "poetry run black core/ fre/ historical_k/ scripts/ tests/ benchmarks/ examples/"
else
    run_check "Black formatting (check)" "poetry run black --check core/ fre/ historical_k/ scripts/ tests/ benchmarks/ examples/"
fi

# 2. isort import sorting
if [ "$FIX" = true ]; then
    run_check "isort import sorting (auto-fix)" "poetry run isort core/ fre/ historical_k/ scripts/ tests/ benchmarks/ examples/"
else
    run_check "isort import sorting (check)" "poetry run isort --check-only core/ fre/ historical_k/ scripts/ tests/ benchmarks/ examples/"
fi

# 3. flake8 linting
run_check "flake8 linting" "poetry run flake8 core/ fre/ historical_k/ scripts/ --max-line-length=88 --extend-ignore=E203,W503,E501"

# 4. mypy type checking
run_check "mypy type checking" "poetry run mypy core/ fre/ historical_k/ --config-file=pyproject.toml --ignore-missing-imports"

# 5. bandit security check
run_check "bandit security scan" "poetry run bandit -r core/ fre/ historical_k/ -f screen -ll"

# 6. pytest tests
run_check "pytest unit tests" "poetry run pytest tests/ -v --tb=short -m 'not slow and not integration'"

# 7. Documentation build (if available)
if [ -f "docs/conf.py" ]; then
    run_check "Sphinx documentation build" "cd docs && poetry run make html && cd .."
fi

# 8. Check for common issues
echo ""
info "Checking for common issues..."

# Check for TODO/FIXME comments
if grep -r "TODO\|FIXME" core/ fre/ historical_k/ 2>/dev/null | grep -v "Binary file"; then
    failure "Found TODO/FIXME comments (not a failure, just FYI)"
else
    success "No TODO/FIXME comments found"
fi

# Check for print statements (should use logging)
if grep -r "^\s*print(" core/ fre/ historical_k/ --include="*.py" 2>/dev/null | grep -v "Binary file"; then
    failure "Found print() statements (use logging instead)"
else
    success "No print() statements found"
fi

# Check for large files
LARGE_FILES=$(find . -type f -size +1M -not -path "./.venv/*" -not -path "./.git/*" -not -path "./logs/*" -not -path "./data/*" 2>/dev/null || true)
if [ -n "$LARGE_FILES" ]; then
    failure "Found large files (>1MB):"
    echo "$LARGE_FILES"
else
    success "No large files found in repository"
fi

# Summary
echo ""
echo "================================================"
echo "  Code Quality Summary"
echo "================================================"
echo ""
echo -e "Checks passed: ${GREEN}$PASSED${NC}"
echo -e "Checks failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    success "All checks passed! Code quality is excellent. 🎉"
    exit 0
else
    failure "Some checks failed. Please review the output above."
    echo ""
    echo "Suggestions:"
    if [ "$FIX" = false ]; then
        echo "  - Run with --fix to auto-fix formatting issues"
    fi
    echo "  - Review failed checks and make necessary corrections"
    echo "  - See DEVELOPMENT.md for coding standards"
    echo ""
    exit 1
fi
