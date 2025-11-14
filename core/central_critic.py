"""Centralized critic scaffolding for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CriticConfig:
    state_dim: int
    learning_rate: float = 3e-4
    gamma: float = 0.99
    weight_decay: float = 0.0


class CentralCritic:
    """Simple linear critic with TD(0) learning."""

    def __init__(self, config: CriticConfig) -> None:
        self.config = config
        self.weights = np.zeros((config.state_dim,), dtype=np.float32)
        self.bias = 0.0

    def predict(self, states: np.ndarray) -> np.ndarray:
        return states @ self.weights + self.bias

    def update(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        values = self.predict(states)
        next_values = self.predict(next_states)
        targets = rewards + self.config.gamma * (1.0 - dones) * next_values
        td_error = values - targets
        grad_w = states.T @ td_error / states.shape[0]
        grad_b = float(td_error.mean())

        if self.config.weight_decay:
            grad_w += self.config.weight_decay * self.weights

        self.weights -= self.config.learning_rate * grad_w
        self.bias -= self.config.learning_rate * grad_b
        loss = float((td_error**2).mean())
        return loss
