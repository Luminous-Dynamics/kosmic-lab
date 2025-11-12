# Kosmic Lab Baseline Results – Draft

## Methods Overview

### Fractal Reciprocity Engine (FRE) Phase 1
- Configurations: `fre/configs/k_config.yaml` (5D grid), `experiment_5d_robustness.yaml`, `experiment_6d_generalization.yaml`.
-  Harmonics computed with `HarmonyCalculator`; Universe dynamics via `UniverseSimulator`.
-  Telemetry recorded as K-Codices (`logs/fre_phase1/*.json`) using `core/kcodex.py`.
-  Pre-registered metrics: K-index (`K > 1` corridor hits), TAT, Recovery, TE symmetry.

### Historical K(t) v1
-  Energy-derived proxies from OWID (`build_from_owid.py`).
-  Additional global indicators from World Bank (`build_from_worldbank.py`).
-  Decadal aggregation (1800–2010), bootstrap CI (1 000 resamples).

## Results

### FRE Phase 1 Baseline
-  972 runs, 460 corridor hits (47.3 %).
-  `mean_K = 0.97`, `std_K = 0.17`; centroid ≈ `(energy 0.48, comm 0.30, plasticity 0.13, jitter 0.05, noise −0.015)`.
-  Plots: `logs/fre_phase1_plots/k_distribution.png`, `parameter_scatter.png`.

### FRE Robustness/Generalization
-  **5D robustness** (`experiment_5d_robustness.yaml`): 3 840 runs, 50.1 % corridor rate, centroid shift 0.024 vs. baseline.
-  **6D generalization** (`experiment_6d_generalization.yaml`): 5 184 runs, 65.3 % corridor rate, Jaccard 0.57 vs. 5D corridor, centroid shift 0.028.
-  JSON summaries: `logs/fre_phase1_summary_5d.json`, `logs/fre_phase1_summary_6d.json`.

### Historical K(t)
-  Mean K ≈ 0.23 (95 % CI [0.19, 0.27]).
-  Post‑1940 mean ≈ 0.31 vs. pre‑1940 ≈ 0.16.
-  Figure: `logs/historical_k/k_t_plot.png`.

## Next Steps
-  Integrate Jaccard/centroid comparisons directly into publication tables.
-  Expand historical proxies (trade/migration/reciprocity datasets) and re-run pipeline.
-  Formalize Methods/Results text for submission.
