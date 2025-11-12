from __future__ import annotations

import numpy as np

from core.central_critic import CentralCritic, CriticConfig


def test_critic_td_update_reduces_loss() -> None:
    cfg = CriticConfig(state_dim=3, learning_rate=0.05, gamma=0.9)
    critic = CentralCritic(cfg)
    rng = np.random.default_rng(0)
    states = rng.normal(size=(128, 3)).astype(np.float32)
    rewards = rng.normal(loc=1.0, scale=0.2, size=128).astype(np.float32)
    next_states = states + 0.05
    dones = np.zeros(128, dtype=np.float32)

    loss_before = critic.update(states, rewards, next_states, dones)
    loss_after = critic.update(states, rewards, next_states, dones)
    assert loss_after <= loss_before + 1e-6


def test_predict_shapes() -> None:
    cfg = CriticConfig(state_dim=2)
    critic = CentralCritic(cfg)
    states = np.ones((4, 2), dtype=np.float32)
    values = critic.predict(states)
    assert values.shape == (4,)
