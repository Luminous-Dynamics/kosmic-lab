# Holochain Zome Testing Strategy

## Goals
- Provide Rust unit tests for helper functions (anchoring, bucket computation).
- Add validation tests using the mock HDK utilities.
- Outline integration plan using `hc sandbox` or conductor harness.

## Immediate Tasks
1. Extend `state_zome` tests to cover `list_agent_states` with mocked links.
2. Add validation tests for `control_zome` entries to ensure knob updates stay in bounds. *(done)*
3. Create a conductor integration harness (Rust) that spins up a test conductor, installs zomes, and exercises signal + entry flows. Skeleton lives in `holochain/tests/integration.rs`.
   - Step A: define DNA/zome bundle for tests under `holochain/tests/fixtures`.
   - Step B: use `hc_sandbox` helper to bootstrap conductor and agents.
   - Step C: drive end-to-end signal → entry flows and assert DHT propagation.
   - Step D: script entry point tracked in `holochain/tests/run_sandbox.sh` (requires `hc` binary).
   - Step E: minimal API smoke-test will require `holochain_conductor_api` once `hc` is available.
4. Document how to run the harness once `hc sandbox` workflow is scripted (e.g., via cargo xtask or shell script).

## Tooling
- `holochain` crate's `test_utils` for mock HDK.
- `hc sandbox` for runtime integration.
- Document yarn-based script for running conductor tests once available.

## Status
- Anchor helper tests implemented (`zomes/state_zome/tests/unit.rs`).
- Metrics bucket helper tests implemented.
- Validation logic added for control knob entries.
- Integration placeholder created at `holochain/tests/integration.rs`.
- Next: mock HDK tests for `list_metrics` and fully scripted conductor harness.
