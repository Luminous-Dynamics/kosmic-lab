# Stage 2 Implementation Tasks

This document breaks down the remaining Stage 2 work so each item can be tackled independently without blocking others.

## 1. FRE – Bioelectric Rescue (Track C)
- Implement rescue dynamics per `docs/track_c_rescue_spec.md`.
  - `fep_to_bioelectric(agent, timestep)` trigger.
  - `bioelectric_to_autopoiesis(agent, target_morphology)`.
  - `compute_iou(current_morphology, target_morphology)`.
- Add unit tests covering the trigger, regeneration, and IoU edge cases.
- Integrate rescue logic into the simulator loop and connect to Track C runner.

## 2. FRE – SAC Controller (Track B)
- Implement SAC module (`fre/controller_sac.py`) following `docs/track_b_sac_spec.md`.
- Build Track B runner to alternate open-loop vs. SAC-controlled episodes.
- Extend `fre/track_a_gate.py` thresholds for controller evaluation.
- Validate rewards by logging K, ΔK, and loss curves.

## 3. FRE Phase 2 – TE-Based Coupling
- Replace mean-field coupling in `fre/multi_universe.py` with TE-driven bridges.
- Recompute β using `fre/run_scaling.py` and update `docs/stage2_scaling_note.md`.
- Add regression output (β, r²) and diagnostic plots for manuscript.

## 4. Historical K(t)
- Follow `docs/historical_data_plan.md` to ingest trade flows, aid/extraction, patents, etc.
- After each dataset:
  - Update `historical_k/k_config.yaml`.
  - Rerun `historical_k/compute_k.py`.
  - Log impact in `docs/historical_modern_note.md`.
- Add regression tests (reciprocity → recovery) with controlled shock catalogue.

## 5. Holochain Prototype
- Flesh out Rust/WASM zomes in `holochain/zomes/*`:
  - `state_zome`: CRUD for agent state.
  - `metrics_zome`: write/query harmonies.
  - `control_zome`: knob updates/signals.
  - `bridge_zome`: TE bridge negotiation.
- Create integration tests to ensure entries validate correctly.
- Use `holochain/scripts/monitor.py` to verify corridor rate on network.

## 6. Pre-registration & Documentation
- Upload `osf_prereg_bundle_fre_phase1_v1.zip` once OSF access returns.
- Maintain Stage 2 lab log (template to add).
- Ensure all new scripts have docstrings, type hints, and TODO markers.
