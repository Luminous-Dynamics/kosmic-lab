# Track A Lab Log – Stage 1 Corridor Sweep

**Date:** 2025-10-29  
**Config:** `fre/configs/k_config.yaml` (5D grid, seeds_per_point=3)  
**Runs generated:** 486 passports (`logs/fre_phase1/*.json`) after isolating current sweep.

## Gate Metrics (auto-generated via `poetry run python fre/track_a_gate.py`)

| Samples | Corridor Rate | Jaccard vs Full | Mean K |
|---------|---------------|-----------------|--------|
| 50      | 0.40          | 0.82            | 0.953  |
| 150     | 0.43          | 0.91            | 0.951  |
| 300     | 0.48          | 1.00            | 0.970  |

**Interpretation:**
- Gate 1 (50 samples) passes threshold (`corridor_rate ≥ 0.35`, `Jaccard ≥ 0.5`).
- Gate 2 (150 samples) passes (`corridor_rate ≥ 0.40`, `Jaccard ≥ 0.65`).
- Gate 3 (300 samples) passes (`corridor_rate ≥ 0.45`, `Jaccard ≥ 0.70`).
- Proceed to compute full Track A corridor (300 samples sufficient; additional runs optional).

## Notes
- Logs archived: baseline `*.json` moved to `logs/fre_phase1` (current sweep), previous 6D runs stored under `logs/fre_phase1_6d/`.
- Corridor export for this sweep: `logs/fre_phase1/corridor_points_baseline.csv` (recomputed after gating).
- TE estimator: KSG (k=5, lag=1) – no anomalies observed.
- Failures: none (no population collapse, no negative K).

## Next Actions
1. Lock SAP/OSF preregistration (pending OSF upload availability).
2. Initiate Stage 2 (Tracks B/C, FRE Phase 2) once Track A prereg recorded.
3. Retain gate CSV at `logs/fre_phase1_trackA_gates.csv` for reproducibility.
