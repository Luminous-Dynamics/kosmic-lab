#!/usr/bin/env bash
# Setup development environment for Kosmic Lab
# This script automates the initial setup process

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "================================================"
echo "  Kosmic Lab Development Environment Setup"
echo "================================================"
echo ""

# Check prerequisites
info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
info "Found Python $PYTHON_VERSION"

# Check Poetry
if ! command -v poetry &> /dev/null; then
    error "Poetry is not installed."
    echo "Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

POETRY_VERSION=$(poetry --version | cut -d' ' -f3 | tr -d '()')
info "Found Poetry $POETRY_VERSION"

# Check Git
if ! command -v git &> /dev/null; then
    error "Git is not installed. Please install Git."
    exit 1
fi

GIT_VERSION=$(git --version | cut -d' ' -f3)
info "Found Git $GIT_VERSION"

success "All prerequisites met!"
echo ""

# Install dependencies
info "Installing Python dependencies with Poetry..."
poetry install --with dev
success "Dependencies installed!"
echo ""

# Setup pre-commit hooks
info "Setting up pre-commit hooks..."
poetry run pre-commit install
success "Pre-commit hooks installed!"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    info "Creating .env file from template..."
    cp .env.example .env
    success ".env file created! Please update it with your configuration."
else
    warning ".env file already exists. Skipping creation."
fi
echo ""

# Create necessary directories
info "Creating project directories..."
mkdir -p logs
mkdir -p data
mkdir -p experiments
mkdir -p benchmarks/results
success "Directories created!"
echo ""

# Run tests to verify setup
info "Running tests to verify setup..."
if poetry run pytest tests/ -v --maxfail=3; then
    success "Tests passed! Environment is ready."
else
    warning "Some tests failed. Please review the output above."
fi
echo ""

# Display helpful information
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Review and update .env file with your configuration"
echo "  2. Activate Poetry shell: poetry shell"
echo "  3. Run an example: python examples/01_hello_kosmic.py"
echo "  4. View available commands: make help"
echo ""
echo "Useful commands:"
echo "  make test          - Run tests"
echo "  make format        - Format code"
echo "  make lint          - Run linters"
echo "  make docs          - Build documentation"
echo "  make dashboard     - Launch monitoring dashboard"
echo ""
echo "Documentation:"
echo "  README.md          - Project overview"
echo "  DEVELOPMENT.md     - Development guide"
echo "  CONTRIBUTING.md    - Contributing guide"
echo "  QUICK_REFERENCE.md - Quick reference"
echo ""
success "Happy coding! 🚀"
