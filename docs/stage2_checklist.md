# Stage 2 Execution Checklist

## Before Starting
- [ ] OSF preregistration uploaded (pending service availability).
- [x] Stage 1 gates logged (`docs/track_a_lab_log.md`).
- [x] Baseline reports collected (`docs/fre_phase1_report.md`, `docs/historical_modern_note.md`).

## Track C – Bioelectric Rescue
- [ ] Implement functions in `fre/rescue.py` (see `docs/track_c_rescue_spec.md`).
- [ ] Add simulator integration (trigger and regeneration calls).
- [ ] Populate Track C runner and gating thresholds.
- [ ] Unit/integration tests green.

## Track B – SAC Controller
- [ ] Flesh out `fre/controller_sac.py` with actual actor/critic networks.
- [ ] Implement Track B runner and logging.
- [ ] Extend gating script for controller metrics.
- [ ] Verify reward curves and baseline comparison.

## Scaling β
- [ ] Replace mean-field coupling with TE-based bridges in `fre/multi_universe.py`.
- [ ] Rerun `fre/run_scaling.py` and update `docs/stage2_scaling_note.md` with real β.
- [ ] Generate diagnostic plots (log-log fit, residuals).

## Historical Proxies
- [ ] Ingest trade flows dataset; rerun pipeline.
- [ ] Ingest aid/extraction dataset; rerun pipeline.
- [ ] Ingest patents dataset; rerun pipeline.
- [ ] Update `docs/historical_modern_note.md` after each addition.

## Holochain Prototype
- [ ] Implement zome logic (state, metrics, control, bridge).
- [ ] Deploy to local network, run monitor script.
- [ ] Log corridor statistics from live network.

## Documentation & Reporting
- [ ] Maintain Stage 2 lab log (create new file under `docs/`).
- [ ] Update `docs/paper_draft.md` as new results arrive.
- [ ] Prepare interim progress summary for collaborators.
