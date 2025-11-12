# FRE Phase 1 – Internal Summary Report

## Overview
This report captures all Phase 1 corridor experiments executed to date. Metrics are derived from `fre/analyze.py` outputs.

| Experiment | Total Runs | Corridor Hits | Corridor Rate | Mean K | Notes |
|------------|------------|---------------|---------------|--------|-------|
| Baseline 3D (Track A) | 486 | 230 | 0.473 | 0.970 | Config `fre/configs/k_config.yaml`, seeds_per_point=3 |
| 5D Robustness | 3,840 | 1,923 | 0.501 | 0.963 | Config `fre/configs/experiment_5d_robustness.yaml` |
| 6D Generalization | 5,184 | 3,386 | 0.653 | 0.983 | Config `fre/configs/experiment_6d_generalization.yaml` |

### Key Plots
- `logs/fre_phase1_plots/k_distribution.png`
- `logs/fre_phase1_plots/parameter_scatter.png`
- Additional corridor exports: `corridor_points_baseline.csv`, `corridor_points_5d.csv`, `corridor_points_6d.csv`

### Baseline Comparison
- 5D sweep centroid shift vs baseline: 0.024, Jaccard 0.00 (expected; additional dims).
- 6D sweep centroid shift vs 5D corridor: 0.028, Jaccard 0.57 (policy generalization retains ~57% overlap).

### Takeaways
1. Corridor persists under dimensional expansion with modest centroid movement.
2. Controller-free 6D evaluation maintains corridor rate > 65%, indicating robustness of underlying dynamics.
3. Outputs ready for inclusion in Stage 1 manuscript draft (`docs/paper_draft.md`).

## Files Checklist
- Summaries: `logs/fre_phase1_summary.json`, `logs/fre_phase1_summary_5d.json`, `logs/fre_phase1_summary_6d.json`
- Gate log: `logs/fre_phase1_trackA_gates.csv`
- Corridors: `logs/fre_phase1/corridor_points_{baseline,5d,6d}.csv`

## Next Steps
- Incorporate summary tables into paper draft.
- Use 5D results as baseline for future Jaccard comparisons.
- Prepare TE audit outputs (see `fre/te_audit.py`).
