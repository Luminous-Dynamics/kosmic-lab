# Kosmic Lab Makefile - Powerful shortcuts for 10/10 productivity

PYTHON ?= python3
LOGDIR ?= logs/fre_phase1

.PHONY: help init lint test fre-run historical-run docs docs-serve docs-clean
.PHONY: dashboard notebook ai-suggest coverage demo clean benchmarks
.PHONY: holochain-publish holochain-query holochain-verify mycelix-demo
.PHONY: format type-check security-check ci-local review-improvements
.PHONY: validate-install profile check-all migrate-v1.1 update-deps
.PHONY: benchmark-parallel benchmark-suite performance-check profile-k-index profile-bootstrap

help:  # Show all available targets
	@echo "🌊 Kosmic Lab - Available Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

init:  # Bootstrap poetry environment and pre-commit hooks
	poetry install --sync
	poetry run pre-commit install
	@echo "✅ Environment ready! Run 'make test' to verify."

lint:  # Run static analysis (black, flake8, mypy)
	poetry run pre-commit run --all-files

test:  # Run unit + integration tests
	poetry run pytest --maxfail=1 --disable-warnings -q

test-verbose:  # Run tests with full output
	poetry run pytest -vv --tb=long

coverage:  # Generate test coverage report (HTML)
	poetry run pytest --cov=core --cov=fre --cov=historical_k --cov-report=html --cov-report=term
	@echo "📊 Coverage report: htmlcov/index.html"

test-property:  # Run property-based tests (hypothesis)
	poetry run pytest tests/test_property_based.py -v

fre-run:  # Execute FRE Phase 1 with default config
	poetry run python fre/run.py --config fre/configs/k_config.yaml

fre-summary:  # Aggregate FRE K-passports
	poetry run python fre/analyze.py --logdir $(LOGDIR) --output $(LOGDIR)/summary.json
	@cat $(LOGDIR)/summary.json | jq

historical-run:  # Compute Historical K(t) from 1800-2020
	poetry run python historical_k/compute_k.py --config historical_k/k_config.yaml

dashboard:  # Launch real-time interactive dashboard
	poetry run python scripts/kosmic_dashboard.py --logdir $(LOGDIR) --port 8050

notebook:  # Auto-generate analysis Jupyter notebook
	poetry run python scripts/generate_analysis_notebook.py \
		--logdir $(LOGDIR) \
		--output analysis/auto_analysis_$(shell date +%Y%m%d).ipynb
	@echo "📓 Notebook: analysis/auto_analysis_$(shell date +%Y%m%d).ipynb"

ai-suggest:  # Get AI-powered experiment suggestions
	poetry run python scripts/ai_experiment_designer.py \
		--train $(LOGDIR) \
		--model models/designer.pkl \
		--suggest 10 \
		--target-k 1.5 \
		--output configs/ai_suggestions_$(shell date +%Y%m%d).yaml
	@echo "🧠 Suggestions: configs/ai_suggestions_$(shell date +%Y%m%d).yaml"

demo:  # Run quick demo (5 min)
	@echo "🚀 Running Kosmic Lab demo..."
	poetry run python fre/run.py --config fre/configs/k_config.yaml --universes 2 --samples 25
	make notebook LOGDIR=logs
	@echo "✅ Demo complete! Check logs/ and analysis/"

docs:  # Build Sphinx documentation
	cd docs && poetry run make html
	@echo "📚 Docs: docs/_build/html/index.html"

docs-serve:  # Build and serve docs locally
	cd docs && poetry run make html
	@echo "📚 Starting documentation server..."
	@echo "📖 Open http://localhost:8000 in your browser"
	cd docs/_build/html && python -m http.server 8000

docs-clean:  # Clean documentation build artifacts
	cd docs && make clean
	@echo "🧹 Documentation build cleaned!"

benchmarks:  # Run performance benchmarks
	@echo "⚡ Running performance benchmarks..."
	poetry run python benchmarks/run_benchmarks.py
	@echo "✅ Benchmarks complete!"

benchmarks-save:  # Run and save benchmark results
	@echo "⚡ Running and saving benchmarks..."
	poetry run python benchmarks/run_benchmarks.py --save benchmarks/results/benchmark_$(shell date +%Y%m%d_%H%M%S).json
	@echo "✅ Benchmarks saved!"

benchmark-parallel:  # Compare serial vs parallel performance (Phase 14)
	@echo "⚡ Comparing serial vs parallel performance..."
	poetry run python benchmarks/suite.py --compare-parallel
	@echo "✅ Parallel comparison complete!"

benchmark-suite:  # Run comprehensive benchmark suite (Phase 14)
	@echo "⚡ Running comprehensive benchmark suite..."
	poetry run python benchmarks/suite.py
	@echo "✅ Benchmark suite complete!"

performance-check:  # Quick performance validation (smoke test)
	@echo "⚡ Running quick performance check..."
	@echo "Testing K-Index performance..."
	poetry run python -c "from fre.metrics.k_index import k_index; import numpy as np; import time; \
		np.random.seed(42); obs = np.random.randn(10000); act = np.random.randn(10000); \
		start = time.time(); k = k_index(obs, act); elapsed = time.time() - start; \
		print(f'K-Index (N=10k): {elapsed*1000:.2f} ms'); \
		assert elapsed < 0.1, 'Performance regression detected!'"
	@echo "✅ Performance check passed!"

profile-k-index:  # Profile K-Index computation
	@echo "⚡ Profiling K-Index computation..."
	poetry run python -m cProfile -o profiling/k_index.stats -c \
		"from fre.metrics.k_index import k_index; import numpy as np; \
		np.random.seed(42); obs = np.random.randn(100000); act = np.random.randn(100000); \
		k = k_index(obs, act)"
	@echo "📊 Profile saved: profiling/k_index.stats"
	@echo "View with: python -m pstats profiling/k_index.stats"

profile-bootstrap:  # Profile bootstrap CI computation
	@echo "⚡ Profiling bootstrap CI..."
	poetry run python -m cProfile -o profiling/bootstrap_ci.stats -c \
		"from fre.metrics.k_index import bootstrap_k_ci; import numpy as np; \
		np.random.seed(42); obs = np.random.randn(1000); act = np.random.randn(1000); \
		k, ci_low, ci_high = bootstrap_k_ci(obs, act, n_bootstrap=100, n_jobs=1)"
	@echo "📊 Profile saved: profiling/bootstrap_ci.stats"
	@echo "View with: python -m pstats profiling/bootstrap_ci.stats"

clean:  # Remove generated files
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	rm -rf logs/*.json analysis/*.ipynb
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "🧹 Cleaned!"

validate:  # Run all checks (tests, lint, coverage)
	@echo "🔍 Running full validation..."
	make lint
	make test
	make coverage
	@echo "✅ All checks passed!"

quick:  # Quick validation (tests only)
	poetry run pytest -x --tb=short -q

format:  # Auto-format code with black and isort
	@echo "🎨 Formatting code..."
	poetry run black core/ fre/ historical_k/ scripts/ tests/
	poetry run isort core/ fre/ historical_k/ scripts/ tests/
	@echo "✅ Code formatted!"

type-check:  # Run mypy type checking
	@echo "🔍 Running type checks..."
	poetry run mypy core/ fre/ historical_k/ --config-file=pyproject.toml
	@echo "✅ Type checks complete!"

security-check:  # Run security scan with bandit
	@echo "🔒 Running security scan..."
	poetry run bandit -r core/ fre/ historical_k/ -f screen
	@echo "✅ Security scan complete!"

ci-local:  # Run full CI pipeline locally
	@echo "🚀 Running CI pipeline locally..."
	@echo ""
	@echo "Step 1: Format check"
	poetry run black --check core/ fre/ historical_k/ scripts/ tests/
	@echo ""
	@echo "Step 2: Import sorting check"
	poetry run isort --check-only core/ fre/ historical_k/ scripts/ tests/
	@echo ""
	@echo "Step 3: Type checking"
	make type-check
	@echo ""
	@echo "Step 4: Tests with coverage"
	make coverage
	@echo ""
	@echo "✅ CI pipeline complete!"

review-improvements:  # View code quality improvements
	@echo "📊 Code Quality Improvements Summary"
	@echo "===================================="
	@cat IMPROVEMENTS.md | head -50
	@echo ""
	@echo "📖 Full report: IMPROVEMENTS.md"
	@echo "🏗️  Architecture: ARCHITECTURE.md"

# ========== Mycelix Integration ==========

holochain-publish:  # Publish K-passports to Mycelix DHT
	@echo "🌊 Publishing K-passports to Holochain DHT..."
	poetry run python scripts/holochain_bridge.py --publish $(LOGDIR)

holochain-query:  # Query corridor passports from DHT
	@echo "🔍 Querying Mycelix corridor..."
	poetry run python scripts/holochain_bridge.py --query --min-k 1.0 --max-k 1.5

holochain-verify:  # Verify passport integrity (usage: make holochain-verify HASH=QmXXX)
	@echo "🔐 Verifying passport..."
	poetry run python scripts/holochain_bridge.py --verify $(HASH)

mycelix-demo:  # Complete Mycelix integration demo
	@echo "🚀 Kosmic-Lab ↔ Mycelix Integration Demo"
	@echo "=========================================="
	@echo ""
	@echo "Step 1: Generate K-passports"
	make fre-run
	@echo ""
	@echo "Step 2: Publish to Holochain DHT"
	make holochain-publish
	@echo ""
	@echo "Step 3: Query corridor"
	make holochain-query
	@echo ""
	@echo "✅ Demo complete! K-passports are now on Mycelix DHT."

# ========== Phase 10: Advanced Tooling ==========

validate-install:  # Comprehensive installation validation
	@echo "🔍 Validating Kosmic Lab installation..."
	@./scripts/validate_installation.sh --verbose

profile:  # Profile performance bottlenecks
	@echo "⚡ Profiling performance..."
	poetry run python scripts/profile_performance.py --format html
	@echo "📊 Profile results: profiling/*.html"

profile-all:  # Profile all functions with detailed output
	@echo "⚡ Profiling all functions..."
	poetry run python scripts/profile_performance.py --function all --samples 10000 --format html
	@echo "📊 Detailed profile: profiling/*.html"

check-all:  # Run ALL validation checks (installation, quality, tests)
	@echo "🚀 Running comprehensive validation..."
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "1. Installation Validation"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@./scripts/validate_installation.sh
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "2. Code Quality Checks"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@./scripts/check_code_quality.sh
	@echo ""
	@echo "✅ All validation checks passed!"

migrate-v1.1:  # Migrate from v1.0.0 to v1.1.0
	@echo "🔄 Migrating to v1.1.0..."
	@echo ""
	@echo "Step 1: Update dependencies"
	poetry install --sync
	@echo ""
	@echo "Step 2: Install pre-commit hooks"
	poetry run pre-commit install
	@echo ""
	@echo "Step 3: Create .env from template"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example"; \
	else \
		echo "⚠ .env already exists, skipping..."; \
	fi
	@echo ""
	@echo "Step 4: Run tests"
	make test
	@echo ""
	@echo "Step 5: Validate code quality"
	./scripts/check_code_quality.sh
	@echo ""
	@echo "✅ Migration complete!"
	@echo "📖 See CHANGELOG.md for full v1.1.0 details"

update-deps:  # Update all dependencies
	@echo "📦 Updating dependencies..."
	poetry update
	poetry lock
	@echo "✅ Dependencies updated!"
	@echo "⚠ Run 'make test' to verify compatibility"

watch-tests:  # Watch for changes and auto-run tests
	@echo "👀 Watching for changes..."
	@echo "Press Ctrl+C to stop"
	@poetry run pytest-watch tests/ --clear --nobeep

install-dev:  # Install development tools
	@echo "🛠️  Installing development tools..."
	poetry install --with dev
	poetry run pre-commit install
	@echo "✅ Development tools installed!"

docker-build:  # Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t kosmic-lab:latest .
	@echo "✅ Docker image built: kosmic-lab:latest"

docker-run:  # Run Kosmic Lab in Docker
	@echo "🐳 Running Kosmic Lab in Docker..."
	docker run -it --rm -v $(PWD)/data:/app/data -v $(PWD)/logs:/app/logs kosmic-lab:latest

docker-shell:  # Open shell in Docker container
	@echo "🐳 Opening shell in Docker..."
	docker run -it --rm -v $(PWD):/app kosmic-lab:latest /bin/bash

release-check:  # Pre-release validation checklist
	@echo "🚀 Release Validation Checklist"
	@echo "================================"
	@echo ""
	@echo "Running comprehensive checks..."
	@echo ""
	make check-all
	@echo ""
	@echo "Running benchmarks..."
	make benchmarks
	@echo ""
	@echo "Building documentation..."
	make docs
	@echo ""
	@echo "✅ Release validation complete!"
	@echo ""
	@echo "📋 Next steps:"
	@echo "  1. Update version in pyproject.toml"
	@echo "  2. Update CHANGELOG.md"
	@echo "  3. Review RELEASE_CHECKLIST.md"
	@echo "  4. Create git tag: git tag -a vX.Y.Z -m 'Release X.Y.Z'"
	@echo "  5. Push: git push && git push --tags"
