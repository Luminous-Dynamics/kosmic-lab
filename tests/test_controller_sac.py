from __future__ import annotations

import random

import numpy as np
import torch
import pytest

from fre.controller_sac import ReplayBuffer, SACController, Transition

pytestmark = pytest.mark.filterwarnings("ignore:CUDA initialization")


def test_replay_buffer_sampling() -> None:
    random.seed(0)
    buffer = ReplayBuffer(capacity=10)
    for idx in range(6):
        buffer.push(
            Transition(
                state=[float(idx), float(idx + 1)],
                action=[float(idx % 2)],
                reward=float(idx),
                next_state=[float(idx + 0.5), float(idx + 1.5)],
                done=False,
            )
        )

    batch = buffer.sample(4)
    assert len(batch) == 4
    unique_rewards = {transition.reward for transition in batch}
    assert len(unique_rewards) >= 2


def test_sac_select_action_shape_and_bounds() -> None:
    torch.manual_seed(0)
    controller = SACController(action_dim=2, state_dim=3, hidden_layers=(32, 32), buffer_capacity=256)
    state = [0.1, -0.2, 0.3]
    action = controller.select_action(state, evaluate=False)
    assert len(action) == 2
    assert all(-1.0 <= a <= 1.0 for a in action)

    eval_action = controller.select_action(state, evaluate=True)
    assert len(eval_action) == 2
    assert all(-1.0 <= a <= 1.0 for a in eval_action)


def test_sac_update_returns_metrics() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    controller = SACController(action_dim=1, state_dim=4, hidden_layers=(32, 32), buffer_capacity=512)

    for _ in range(64):
        noise = np.random.randn(4).astype(np.float32)
        next_noise = noise + 0.05 * np.random.randn(4).astype(np.float32)
        controller.store_transition(
            Transition(
                state=noise.tolist(),
                action=[float(np.tanh(noise[0]))],
                reward=float(np.clip(noise.sum(), -1.0, 1.0)),
                next_state=next_noise.tolist(),
                done=False,
            )
        )

    metrics = controller.update(batch_size=32)
    assert metrics, "update should return training metrics when enough samples exist"
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert metrics["alpha"] > 0.0
