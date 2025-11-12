# Track B – Controller Validation Plan

## Objective
Demonstrate that the SAC controller maintains `K > 1.0` with improved TAT and Recovery relative to open-loop baselines (Stage 2 goal).

## Configuration
- Config file: `fre/configs/track_b_control.yaml`
- Runner (to be implemented): `poetry run python fre/run_track_b.py --config fre/configs/track_b_control.yaml`
- Metrics to record:
  - Corridor rate
  - Mean K, TAT, Recovery
  - Controller stability (variance of K)

## Gate Criteria
- Gate 1 (50 samples): controller corridor rate ≥ 0.45
- Gate 2 (150 samples): controller corridor rate ≥ 0.55, Jaccard ≥ 0.60 vs open-loop baseline
- Gate 3 (300 samples): controller corridor rate ≥ 0.65, Recovery ≤ 0.5× open-loop

## Next Steps
1. Implement `fre/run_track_b.py` to alternate between open-loop and SAC runs.
2. Reuse `fre/track_a_gate.py` for gating (with controller-specific thresholds).
3. Log results to `logs/fre_track_b_summary.json` and update Stage 2 notebook.
