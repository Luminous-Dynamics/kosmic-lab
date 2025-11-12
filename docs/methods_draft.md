# Methods – Kosmic Lab Baseline Pipelines

## Fractal Reciprocity Engine (FRE) – Phase 1 Baseline

- **Configuration:** `fre/configs/k_config.yaml` (5D grid over energy/communication/plasticity/noise/jitter, 3 seeds per point, seed base 44).
- **Simulation:** `poetry run python fre/run.py --config fre/configs/k_config.yaml`
  - Universe dynamics simulated via `fre.universe.UniverseSimulator`.
  - Harmonies computed with `harmonics_module.HarmonyCalculator`.
- **Telemetry:** Every run creates a K-Codex JSON (`logs/fre_phase1/*.json`) with metrics:
  - `K`, `TAT`, `Recovery`, TE symmetry/mutual, harmony components (H1–H7), params, seed.
  - See `core/kcodex.py` for schema and implementation.
- **Analysis:** `fre/analyze.py`
  - Aggregates into `logs/fre_phase1_summary.json` (means, corridor rate, baseline comparison).
  - Generates plots (`logs/fre_phase1_plots/k_distribution.png`, `parameter_scatter.png`).
  - Supports baseline export (`logs/fre_phase1/corridor_points_baseline.csv`) and Jaccard/centroid comparisons.
- **Baseline result:** 972 runs, 47% corridor hits (`K>1`), `mean_K ≈ 0.97 ± 0.17`.
- **Ready-to-upload prereg bundle:** `osf_prereg_bundle_fre_phase1_v1.zip` (prereg text, config, baseline metrics, plots).

### Robustness & Generalization Sweeps
- **5D robustness (`fre/configs/experiment_5d_robustness.yaml`):** 3 840 runs, 50% corridor rate after marginalizing topology/jitter; centroid shift 0.024 vs baseline.
- **6D policy generalization (`fre/configs/experiment_6d_generalization.yaml`):** 5 184 runs, 65% corridor rate; Jaccard 0.57 vs 5D corridor; centroid shift 0.028. Results saved to `logs/fre_phase1_summary_5d.json`, `logs/fre_phase1_summary_6d.json`.

## Historical K-index v1 (1800–2010)

- **Data ingestion:**
  - `historical_k/data/build_from_owid.py` → energy-derived proxies (integration, reciprocity, play, wisdom).
  - `historical_k/data/build_from_worldbank.py` → trade share, net migration, remittance inflows (World Bank API).
  - All CSVs recorded with hashes in `docs/data_sources.md`.
- **K computation:** `historical_k/compute_k.py`
  - Loads proxies per harmony, computes K(t) series (`logs/historical_k/k_t_series.csv`).
  - Bootstrap CI (1 000 samples) stored in `logs/historical_k/k_t_summary.json`.
  - Plot saved to `logs/historical_k/k_t_plot.png` (corridor threshold overlay).
- **Current outcome:** `K̄ ≈ 0.23` (CI [0.19, 0.27]); post‑1940 mean ≈ 0.31 vs pre‑1940 ≈ 0.16 (expected upswing).

## Next steps

- Produce publication-ready visuals (∇K streamlines, 5D marginal corridor overlay, historical harmony radar).
- Upload `osf_prereg_bundle_fre_phase1_v1.zip` to OSF and lock hypotheses.
- Integrate additional data sources (trade/migration/reciprocity) as they become available and re-run the historical pipeline to refine K(t).
