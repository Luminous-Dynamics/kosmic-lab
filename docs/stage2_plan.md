# Stage 2 Preparation Plan

## Entry Criteria
- Track A gates passed (see `docs/track_a_lab_log.md`).
- OSF preregistration pending upload (bundle ready).
- Historical Modern Era note drafted (`docs/historical_modern_note.md`).

## Stage 2 Tracks

### v0.5 Tracks B & C
- **Goal:** Demonstrate controller performance (Track B) and bioelectric rescue (Track C).
- **Next Actions:**
  - Finalize prereg text for Tracks B/C once Track A registered.
  - Implement runners (`fre/run_track_b.py`, `fre/run_track_c.py`) using configs `fre/configs/track_b_control.yaml` and `fre/configs/track_c_bioelectric.yaml`.
  - Use `fre/track_a_gate.py` for gating with Track B/C thresholds (see `docs/track_b_plan.md`, `docs/track_c_plan.md`).
  - Validate TE drift via `fre/te_audit.py` after each sweep.

### FRE Phase 2 – Scaling Law β
- **Goal:** Estimate β in `K_∞ ~ N^β` with N ∈ {3,5,7,10}.
- **Plan:**
  - Use `fre/run_scaling.py --config fre/configs/experiment_scaling.yaml` (current placeholder) and inspect `logs/fre_scaling_analysis.csv`.
  - Replace placeholder with full multi-universe coupling before interpreting β (target 0.15–0.25).

### Dependencies
- Historical pipeline updates (new proxies per `docs/historical_data_plan.md`) to continue in parallel.
- Holochain prototype to start capturing distributed telemetry once FRE Phase 2 results stable.

## Risk Mitigation
- If Track A fails to replicate (future reruns), pause Stage 2 and retune TE estimator or parameter ranges.
- Maintain baseline corridor exports (3D, 5D) for all future Jaccard comparisons.
- Log every Phase 2 run in the lab notebook (template in `Stage 1 checklist`).
