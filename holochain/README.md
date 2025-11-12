# Kosmic Lab – Holochain Prototype (Phase 2)

## Structure
- `agents/` – Rust/WASM components for agent state & harmonics computation.
- `zomes/` – Holochain zome modules (`state_zome`, `metrics_zome`, `control_zome`, `bridge_zome`).
- `scripts/` – CLI utilities (simulation runner, knob broadcaster, monitoring dashboard).

## MVP Checklist
1. Scaffold DNA with `state_zome` (store/retrieve `AgentState`) and `metrics_zome` (publish `HarmonyMetrics`).
2. Implement control channel (`control_zome`) to update knobs (`communication_cost`, `plasticity_rate`, `stimulus_intensity`).
3. Prototype bridge handshake in `bridge_zome` for TE computation (stub until TE engine ready).
4. Define schema mirroring K-passport fields for auditing.
5. Build monitoring script to aggregate `K > 1` counts and compare against baseline centroid.

Refer to `docs/holochain_design.md` for detailed architecture and message flow.
