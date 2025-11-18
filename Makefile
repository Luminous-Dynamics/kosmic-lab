# Kosmic Lab Makefile - Powerful shortcuts for 10/10 productivity

PYTHON ?= python3
LOGDIR ?= logs/fre_phase1

.PHONY: help init lint test fre-run historical-run docs
.PHONY: dashboard notebook ai-suggest coverage demo clean
.PHONY: holochain-publish holochain-query holochain-verify mycelix-demo
.PHONY: checkpoint-info checkpoint-list checkpoint-copy config-register config-lookup config-diff track-g track-h log-tail log-validate archive-artifacts archive-summary archive-verify

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

historical-compare:  # Compare normalization strategies
	poetry run python historical_k/comparison.py

historical-contrib:  # Harmony rolling correlation contributions
	poetry run python historical_k/contributions.py

historical-report:  # Generate consolidated markdown report
	poetry run python historical_k/report.py

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

checkpoint-info:  # Inspect checkpoint metadata (CHECKPOINT=path/to.json)
	@if [ -z "$(CHECKPOINT)" ]; then echo "Set CHECKPOINT=..."; exit 1; fi
	poetry run python scripts/checkpoint_tool.py info --path "$(CHECKPOINT)"

checkpoint-list:  # List checkpoints in directory (DIR=logs/track_g/checkpoints)
	@if [ -z "$(DIR)" ]; then echo "Set DIR=..."; exit 1; fi
	poetry run python scripts/checkpoint_tool.py list --dir "$(DIR)"

checkpoint-copy:  # Copy checkpoint (SRC=... DEST=...)
	@if [ -z "$(SRC)" ] || [ -z "$(DEST)" ]; then echo "Set SRC=... and DEST=..."; exit 1; fi
	poetry run python scripts/checkpoint_tool.py copy --src "$(SRC)" --dest "$(DEST)" $(if $(OVERWRITE),--overwrite,)

config-register:  # Register config hash/label (CONFIG=... LABEL="...")
	@if [ -z "$(CONFIG)" ] || [ -z "$(LABEL)" ]; then echo "Set CONFIG=... and LABEL=..."; exit 1; fi
	poetry run python scripts/config_registry.py register --config "$(CONFIG)" --label "$(LABEL)" $(if $(NOTES),--notes "$(NOTES)",)

config-lookup:  # Lookup config info (HASH=... or CONFIG=...)
	@if [ -z "$(HASH)" ] && [ -z "$(CONFIG)" ]; then echo "Set HASH=... or CONFIG=..."; exit 1; fi
	poetry run python scripts/config_registry.py lookup $(if $(HASH),--hash "$(HASH)",) $(if $(CONFIG),--config "$(CONFIG)",)

config-diff:  # Diff two configs (A=..., B=...)
	@if [ -z "$(A)" ] || [ -z "$(B)" ]; then echo "Set A=... and B=..."; exit 1; fi
	poetry run python scripts/config_registry.py diff --config-a "$(A)" --config-b "$(B)"

log-tail:  # Tail JSONL episode log (PATH=logs/track_g/episodes/*.jsonl, LINES=20, FOLLOW=1)
	@if [ -z "$(PATH)" ]; then echo "Set PATH=..."; exit 1; fi
	poetry run python scripts/log_tool.py tail --path "$(PATH)" --lines $${LINES:-20} $(if $(FOLLOW),--follow,)

log-validate:  # Validate JSONL log structure (PATH=logs/track_g/episodes/*.jsonl)
	@if [ -z "$(PATH)" ]; then echo "Set PATH=..."; exit 1; fi
	poetry run python scripts/log_tool.py validate --path "$(PATH)"

archive-artifacts:  # Snapshot checkpoint + log + config (CHECKPOINT=..., LOG=..., CONFIG=..., OUTPUT optional)
	@if [ -z "$(CHECKPOINT)" ] || [ -z "$(CONFIG)" ]; then echo "Set CHECKPOINT=... CONFIG=... (LOG optional)"; exit 1; fi
	poetry run python scripts/archive_tool.py create --checkpoint "$(CHECKPOINT)" $(if $(LOG),--log "$(LOG)",) --config "$(CONFIG)" $(if $(OUTPUT),--output "$(OUTPUT)",)

archive-verify:  # Verify archive bundle (ARCHIVE=archives/*.tar.gz)
	@if [ -z "$(ARCHIVE)" ]; then echo "Set ARCHIVE=..."; exit 1; fi
	poetry run python scripts/archive_tool.py verify --archive "$(ARCHIVE)"

archive-summary:  # Print archive metadata summary (ARCHIVE=archives/*.tar.gz)
	@if [ -z "$(ARCHIVE)" ]; then echo "Set ARCHIVE=..."; exit 1; fi
	poetry run python scripts/archive_tool.py summary --archive "$(ARCHIVE)"

track-g:  # Run Track G phase (PHASE=g2, CONFIG=fre/configs/track_g_phase_g2.yaml)
	@PHASE=$${PHASE:-g2}; CONFIG=$${CONFIG:-fre/configs/track_g_phase_g2.yaml}; \
	ARGS="--config $$CONFIG --phase $$PHASE"; \
	if [ -n "$(WARM_LOAD)" ]; then ARGS="$$ARGS --warm-start-load '$(WARM_LOAD)'"; fi; \
	if [ -n "$(WARM_SAVE)" ]; then ARGS="$$ARGS --warm-start-save '$(WARM_SAVE)'"; fi; \
	if [ -n "$(ALLOW_MISMATCH)" ]; then ARGS="$$ARGS --warm-start-allow-mismatch"; fi; \
	if [ -n "$(DRY_RUN)" ]; then ARGS="$$ARGS --dry-run"; fi; \
	echo "🌊 Running Track G phase '$$PHASE' with $$CONFIG"; \
	poetry run python fre/track_g_runner.py $$ARGS

track-h:  # Run Track H memory integration (CONFIG=fre/configs/track_h_memory.yaml)
	@CONFIG=$${CONFIG:-fre/configs/track_h_memory.yaml}; \
	ARGS="--config $$CONFIG"; \
	if [ -n "$(WARM_LOAD)" ]; then ARGS="$$ARGS --warm-start-load '$(WARM_LOAD)'"; fi; \
	if [ -n "$(WARM_SAVE)" ]; then ARGS="$$ARGS --warm-start-save '$(WARM_SAVE)'"; fi; \
	if [ -n "$(DRY_RUN)" ]; then ARGS="$$ARGS --dry-run"; fi; \
	echo "🧠 Running Track H memory integration with $$CONFIG"; \
	poetry run python fre/track_h_runner.py $$ARGS

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
