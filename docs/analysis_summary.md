# Analysis Summary (as of current run)

## FRE Corridor Metrics
- Baseline Track A: corridor rate 0.473, mean K ≈ 0.97.
- 5D robustness: corridor rate 0.501, centroid shift 0.024 vs baseline.
- 6D generalization: corridor rate 0.653, Jaccard 0.57 vs 5D corridor.
- All summaries in `logs/fre_phase1_summary*.json` and `docs/fre_phase1_report.md`.

## Scaling β (mean-field prototype)
- Preliminary β ≈ 3.5 × 10⁻⁵ (flat). Indicates need for TE-based coupling.
- Logs: `logs/fre_scaling_summary.csv`, `logs/fre_scaling_beta.txt`.

## Historical K(t) Modern Era
- Post-1940 mean K ≈ 0.31 vs pre-1940 ≈ 0.16.
- Strong correlations: reciprocity ↔ K (0.98), play entropy ↔ K (0.89).
- Details in `docs/historical_modern_note.md`.

## Next Analyses
- Compare future controller runs to baseline via `fre/te_audit.py`.
- Track IoU improvements as rescue dynamics are implemented.
- Update staging docs once real β computed.
