# FRE Phase 2 – Scaling β (Preliminary)

**Config:** `fre/configs/experiment_scaling.yaml`  
**Script:** `poetry run python fre/run_scaling.py --config fre/configs/experiment_scaling.yaml`

| Universe Count | Mean K | Σ K |
|----------------|--------|-------|
| 3 | 1.068 | 3.204 |
| 5 | 1.058 | 5.290 |
| 7 | 1.043 | 7.299 |
| 10 | 1.057 | 10.569 |

Log-log regression (mean K vs. universe count) with the TE-bridge coupling currently yields **β ≈ -0.012** (slightly negative), indicating further tuning is needed. Outputs saved to `logs/fre_scaling_summary.csv` and `logs/fre_scaling_beta.txt`.

**Next steps:** adjust TE parameters (window, symmetry mixing, coupling strength) and explore richer driver/driven signals to seek the expected superlinear β band.
