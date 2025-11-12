# Contributing to kosmic-lab

Thank you for helping advance the Kosmic Simulation & Coherence Framework!

## Getting Started
- Install dependencies via `make init`.
- Run `make test` before submitting a PR.
- Each PR must include an updated K-Codex record (local K-Passport stage) for new experiments.

## Harmony Integrity Checklist
Ensure the following items are satisfied before requesting review:
1. Diversity metrics reward plurality (no shortcuts assigning homogeneity = 1).
2. Corridor volume remains normalized (≤ 1.0 after aggregations).
3. Estimator settings (Φ variant, TE params) logged in K-Codex (see `core/kcodex.py`).
4. Visualization thresholds render correctly (e.g., baseline lines visible).
5. Tests updated and passing locally.

## Coding Standards
- Type hints (`from __future__ import annotations`) required.
- Prefer functional decomposition; controllers and metrics go under `core/`.
- For historical data scripts, keep raw data immutable; write derivatives to `data/derived/` (git-ignored).

## Commit Message Guidance
Use concise summaries, e.g., `fre: add phase1 corridor simulator`.
