# Zome Overview

- `state_zome`: CRUD for `AgentState` entries.
- `metrics_zome`: Records `HarmonyMetrics` and exposes query endpoints.
- `control_zome`: Applies knob updates broadcast through the DHT.
- `bridge_zome`: Handles TE bridge coordination between peers.

Each zome will expose Holochain callbacks (`init`, `validate`, `receive`, custom zome functions) once the Rust scaffolding is in place.
