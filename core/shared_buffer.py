"""Shared experience replay buffer for Phase 4 centralized coordination."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Tuple

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float
    universe_id: int


class SharedReplayBuffer:
    """Minimal replay buffer with basic thread safety."""

    def __init__(self, capacity: int = 100_000, seed: int | None = None) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, 1), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.universe_ids = np.zeros(capacity, dtype=np.int32)
        self.position = 0
        self.full = False
        self.lock = Lock()
        self.rng = np.random.default_rng(seed)
        self.state_dim: int | None = None
        self.action_dim: int | None = None

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

    def push(self, exp: Experience) -> None:
        with self.lock:
            state = exp.state.astype(np.float32, copy=False)
            next_state = exp.next_state.astype(np.float32, copy=False)
            action = exp.action.astype(np.float32, copy=False)
            if self.state_dim is None:
                self.state_dim = state.size
                self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
                self.next_states = np.zeros_like(self.states)
            if self.action_dim is None:
                self.action_dim = action.size
                self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)

            idx = self.position
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = exp.reward
            self.next_states[idx] = next_state
            self.dones[idx] = float(exp.done)
            self.universe_ids[idx] = exp.universe_id

            self.position = (self.position + 1) % self.capacity
            if self.position == 0:
                self.full = True

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            current_size = len(self)
            if batch_size > current_size:
                raise ValueError("Not enough experiences to sample")
            indices = self.rng.choice(current_size, batch_size, replace=False)
            states = self.states[indices].copy()
            actions = self.actions[indices].copy()
            rewards = self.rewards[indices].copy()
            next_states = self.next_states[indices].copy()
            dones = self.dones[indices].copy()
            universe_ids = self.universe_ids[indices].copy()
        return (states, actions, rewards, next_states, dones, universe_ids)
