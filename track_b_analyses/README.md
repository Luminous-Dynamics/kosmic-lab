# 🧪 Track B Analyses - Corridor Navigation with SAC Controller

**Note**: This directory was previously named `paper2_analyses/` but has been renamed to `track_b_analyses/` for clarity. Track B results are part of **Paper 1** (combined with Track C), not Paper 2.

## Purpose

This directory contains analysis results from **Track B: SAC Controller experiments** testing coherence-guided navigation through bioelectric corridors.

## Data Contents

### Results (in `results/` subdirectory)

1. **`C_functional_results.csv`** (77 rows, 12 KB)
   - Episode-by-episode results from Track B experiments
   - Columns: episode_id, mode, avg_k, std_k, neg_free_energy, synergy, broadcast, meta_calibration, blanket_tightness, n_internal_cells, n_external_cells, n_timesteps, track
   - 76 data rows covering different training and testing modes

2. **`correlations_with_k_index.csv`** (6 rows, 398 bytes)
   - Correlation analysis between K-Index and other metrics
   - Pearson and Spearman correlations with p-values
   - Key finding: blanket_tightness shows strong negative correlation (-0.965)
   - neg_free_energy shows positive correlation (0.599)

3. **`summary_statistics.csv`** (9 rows, 1.1 KB)
   - Descriptive statistics across all Track B episodes
   - Mean K-Index: 0.803 ± 0.131
   - Mean negative free energy: -6163.78
   - Count: 76 complete episodes

## Track B in Paper 1

These results are incorporated into **Paper 1: Coherence-Guided Control** (`papers/paper1/`) which combines:
- **Track B**: SAC controller with K-index feedback (these results)
- **Track C**: Bioelectric rescue mechanisms

### Key Paper 1 Results from Track B
- 63% improvement in corridor navigation with K-index feedback
- Successful coherence-guided control demonstration
- Foundation for understanding bioelectric navigation

## Related Files

- **Track B runner**: `fre/track_b_runner.py`
- **Track B config**: `fre/configs/track_b_control.yaml`
- **Track B documentation**: `docs/track_b_*.md`
- **Paper 1 manuscript**: `papers/paper1/manuscript.tex`

## Analysis Notes

The C_functional blanket implementation showed:
- Effective separation of internal/external states
- Strong correlation between blanket tightness and K-Index
- Negative free energy as a key coherence indicator

---

*Directory renamed: November 12, 2025*
*Previously: paper2_analyses/*
*Data from: Track B experiments (October-November 2025)*
