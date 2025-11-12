# Track B – SAC Controller Specification

## Objective
Train a Soft Actor-Critic (SAC) controller that maximizes the K-index (or ΔK) by adjusting simulation knobs (e.g., `energy_gradient`, `communication_cost`, `plasticity_rate`). This implements Stage 2 Track B of the roadmap.

## Interfaces
- **State observation**: vector of harmonies (H1–H7), TE symmetry/mutual, recent ΔK, and current knob values.
- **Action space**: continuous adjustments to knobs in bounded ranges (e.g., ±0.05 per step).
- **Reward**: primary reward `r_t = K_t`; optional bonus `β * (K_t - K_{t-1})`.
- **Controller entry point**: `sac_controller.step(observation)` returning action.
- **Training loop**: `run_track_b.py` alternating open-loop and controller episodes.

## Architecture
- Two policy networks (Gaussian actor) with shared MLP body (e.g., 2 hidden layers × 128 units, ReLU).
- Two Q-networks for double estimation (same architecture).
- Temperature α automatically tuned (per Soft Actor-Critic).
- Replay buffer storing `(state, action, reward, next_state, done)` tuples.
- Optimizers: Adam (lr = 3e-4).
- Batch size: 256, discount γ = 0.99, target smoothing τ = 0.005.

## Algorithm Sketch
1. Initialize actor, Q networks, target Q networks, replay buffer.
2. For each environment step:
   - Sample action from actor (add exploration noise during warm-up).
   - Apply action to simulator (update params), record reward and next state.
   - Store transition in replay buffer.
3. After each step, perform gradient updates:
   - Sample mini-batch.
   - Compute target Q using target networks.
   - Update Q networks with MSE loss.
   - Update actor via policy gradient (entropy-regularized).
   - Adjust temperature α (if automatic).
   - Soft update target networks.

## Evaluation Gates
Use `fre/track_a_gate.py` (extended) with controller-specific thresholds:
- Gate 1 (50 samples): corridor rate ≥ 0.45.
- Gate 2 (150 samples): corridor rate ≥ 0.55 and Jaccard ≥ 0.60 vs. open-loop baseline.
- Gate 3 (300 samples): corridor rate ≥ 0.65, recovery ≤ 0.5× open-loop.

## Testing
- Unit tests for replay buffer, action bounds, temperature tuning.
- Integration test comparing open-loop vs. controller on small parameter grid.
- Diagnostics: log reward curves, entropy, loss, and final corridor metrics.
