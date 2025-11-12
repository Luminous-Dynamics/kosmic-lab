# FRE Phase 4 – Centralized Coordination Experiment Design

## Objective
Demonstrate positive β scaling by coordinating universes through a centralized learning signal instead of decentralized TE/RPC coupling.

## Architecture Overview
1. **Shared Experience Buffer**
   - Each universe logs tuples `(state_summary, policy_action, reward_proxy)` to a global replay buffer.
   - State summaries include SAC observation (`IoU`, ΔIoU, energy use) plus harmonies (`cohesion`, `playfulness`, `K`).

2. **Central Critic / Coordinator**
   - Train a centralized critic `V_global(state_summary, universe_id)` using the shared buffer.
   - Critic gradients backprop to universe-specific policy heads (actor cloning) or to update universe control knobs directly.

3. **Universe Policy Heads**
   - Each universe maintains a lightweight actor (e.g., two-layer MLP) that receives critic feedback.
   - Actors update via off-policy TD error or policy-gradient using critic advantage.

4. **Coordination Loop**
   1. For each timestep, universes act (either scripted or via their actor).
   2. Observations and rewards enter the shared buffer.
   3. Central critic trains for `n_critic_updates` per step.
   4. Critic distributes updates to actors; actors update with entropy regularization to maintain diversity.
   5. Periodically evaluate with scaling runs (β measurement) using actors only.

## Key Components
- `core/shared_buffer.py`: thread-safe replay with prioritized sampling.
- `core/central_critic.py`: critic network, optimizer, training loop.
- `fre/multi_universe.py`: integrate actors/critic, manage data flow.
- `configs/phase4_experiment.yaml`: hyperparameters (buffer size, batch size, learning rates).

## Metrics & Evaluation
- Scaling β (mean K vs. universe count) over 50–100 steps.
- Critic loss, actor divergence (KL), buffer utilization.
- Ablations: critic disabled, diffusion only, decentralized RPC baseline.

## Timeline
1. Implement shared buffer + critic (Week 1).
2. Integrate actors in simulator (Week 2).
3. Run pilot scaling tests (Week 3).
4. Full experiment + ablations (Week 4).

## Risks & Mitigations
- **Instability**: use target networks and gradient clipping.
- **Mode collapse**: entropy bonuses and replay diversity filters.
- **Compute cost**: batch critic updates (multi-GPU optional) and reuse SAC model for initialization.
