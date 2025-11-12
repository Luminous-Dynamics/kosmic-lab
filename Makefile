# Kosmic Lab Makefile - Powerful shortcuts for 10/10 productivity

PYTHON ?= python3
LOGDIR ?= logs/fre_phase1

.PHONY: help init lint test fre-run historical-run docs
.PHONY: dashboard notebook ai-suggest coverage demo clean
.PHONY: holochain-publish holochain-query holochain-verify mycelix-demo

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
