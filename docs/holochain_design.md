# Holochain Prototype Design – FRE Phase 2

## Objectives
- Run a 100-node Holochain network mirroring the FRE Phase 1 dynamics.
- Broadcast tunable parameters (`communication_cost`, `plasticity_rate`, `stimulus_intensity`).
- Compute distributed K in real time using local harmonies + gossip TE bridges.

## Agent DNA (per node)
- **Entries:**
  - `AgentState` (serialized analogue of `core.Agent` snapshot).
  - `HarmonyMetrics` (H1–H7, K) with timestamp.
  - `BridgeTelemetry` (pairwise TE estimates with neighbors).
- **Zome Functions:**
  - `update_state(params)` – apply knob updates and recompute harmonies locally.
  - `publish_metrics()` – emit HarmonyMetrics, store to DHT.
  - `request_bridge(peer)` – handshake to compute TE across boundary.
- **Validation Rules:**
  - Only accept entries signed by agent key, minimal schema checks, optional K-Codex hash for auditing.

## Networking
- **Topology:** small-world overlay; each agent maintains ~6 neighbors with periodic rewiring.
- **Bridges:** TE estimation performed every `bridge_interval` steps; results cached  for DHT readers.
- **Control knobs (per agent):**
  - `communication_cost` (0.1–0.5)
  - `plasticity_rate` (0.05–0.25)
  - `stimulus_intensity` (0.0–0.2)
  - Additional optional: `noise_level`, `bridge_bandwidth`.
- **Message flow:**
  1. Coordinator publishes global knob update to DHT.
  2. Agents pull update, run local simulation step (e.g., 10 iterations) using harmonics module compiled to WASM.
  3. Agents push `HarmonyMetrics` entry; neighboring peers fetch latest metrics to compute TE and respond with `BridgeTelemetry`.
  4. Optional: aggregator zome computes rolling statistics (mean K, corridor rate, Jaccard vs baseline) and broadcasts alerts when thresholds breached.

## Distributed K Aggregation
- Each agent maintains sliding window of harmonies.
- A coordinator zome computes global K as mean of latest metrics (or decentralised aggregator via gossip).
- Corridor detection: count proportion of agents with `K > 1.0`, share on DHT.
- Baseline snapshot stored as `CorridorBaseline` entry to enable on-chain Jaccard comparisons for auditability.

## Implementation Plan
1. Scaffold DNA with zomes for state, metrics, control.
2. Implement control channel (CLI or UI) to adjust knobs network-wide.
3. Integrate `core/harmonics` Python logic via WASM-friendly port or via service bridge.
4. Log K-Codex equivalents per node for reproducibility (eternal records).
5. Extend to TE computation using shared mini-history windows (publish last N action vectors hashed for reference).

## Open Questions
- Should TE computation be local (agents share raw history) or via aggregated bridges?
- How to synchronise seeds for reproducibility in distributed context?
- What failure thresholds trigger reconfiguration (e.g., corridor collapse alerts)?
