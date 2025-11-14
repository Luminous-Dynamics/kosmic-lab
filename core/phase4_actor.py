"""Per-universe actor heads for Phase 4 centralized coordination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActorConfig:
    state_dim: int
    action_dim: int
    learning_rate: float = 1e-3


class Phase4Actor:
    """Simple linear actor with tanh squashing and advantage-weighted updates."""

    def __init__(self, config: ActorConfig, seed: int | None = None) -> None:
        self.cfg = config
        rng = np.random.default_rng(seed)
        limit = 1.0 / np.sqrt(config.state_dim)
        self.weights = rng.uniform(
            -limit, limit, size=(config.state_dim, config.action_dim)
        ).astype(np.float32)
        self.bias = np.zeros(config.action_dim, dtype=np.float32)

    def predict(self, state: np.ndarray) -> np.ndarray:
        z = state @ self.weights + self.bias
        return np.tanh(z)

    def update(self, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray) -> None:
        if len(states) == 0:
            return
        preds = self.predict(states)
        error = actions - preds
        coeff = advantages.reshape(-1, 1)
        grad_w = states.T @ (coeff * error)
        grad_b = np.sum(coeff * error, axis=0)
        self.weights += self.cfg.learning_rate * grad_w / states.shape[0]
        self.bias += self.cfg.learning_rate * grad_b / states.shape[0]
