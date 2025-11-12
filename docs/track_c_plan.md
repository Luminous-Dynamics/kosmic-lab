# Track C – Bioelectric Rescue Plan

## Objective
Show that targeted bioelectric interventions restore high K (IoU ≥ 0.85) relative to control conditions.

## Configuration
- Config file: `fre/configs/track_c_bioelectric.yaml`
- Runner (to be implemented): `poetry run python fre/run_track_c.py --config fre/configs/track_c_bioelectric.yaml`
- Metrics: K, survival rate, rescue IoU (placeholder until full bioelectric module implemented).

## Gate Strategy
- Gate 1: Stimulus intensity sweep (0.05–0.10) achieves IoU ≥ 0.80 in 50 samples.
- Gate 2: With knob tuning, IoU ≥ 0.85, corridor rate ≥ 0.40 in 150 samples.
- Gate 3: Rescue stable with IoU ≥ 0.85 across 300 samples; restore K to ≥ 90% of baseline within 20 timesteps.

## TODO
1. Implement rescue dynamics in simulator (currently placeholder).
2. Define IoU metric consistent with bioelectric model once module available.
3. Extend TE audit to include rescue-induced symmetry shifts.
4. Record results in `logs/fre_track_c_summary.json` and Stage 2 lab log.
