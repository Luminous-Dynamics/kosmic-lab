"""Phase 4 coordinator tying together shared buffer and critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from core.central_critic import CentralCritic, CriticConfig
from core.shared_buffer import Experience, SharedReplayBuffer


@dataclass
class BufferConfig:
    capacity: int = 200_000
    batch_size: int = 512
    seed: int | None = None


@dataclass
class TrainerConfig:
    warmup_steps: int = 1_000
    critic_updates_per_step: int = 4
    log_interval: int = 100


class Phase4Coordinator:
    """Manages shared replay buffer and centralized critic training."""

    def __init__(
        self,
        buffer_cfg: BufferConfig,
        critic_cfg: CriticConfig,
        trainer_cfg: TrainerConfig,
    ) -> None:
        self.buffer = SharedReplayBuffer(capacity=buffer_cfg.capacity, seed=buffer_cfg.seed)
        self.batch_size = buffer_cfg.batch_size
        self.critic = CentralCritic(critic_cfg)
        self.trainer_cfg = trainer_cfg
        self.step_count = 0
        self.last_loss: Optional[float] = None

    def record_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        universe_id: int,
    ) -> None:
        exp = Experience(
            state=state.astype(np.float32, copy=False),
            action=action.astype(np.float32, copy=False),
            reward=float(reward),
            next_state=next_state.astype(np.float32, copy=False),
            done=float(done),
            universe_id=universe_id,
        )
        self.buffer.push(exp)

    def maybe_train(
        self,
    ) -> Tuple[Optional[float], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        self.step_count += 1
        if len(self.buffer) < self.trainer_cfg.warmup_steps:
            return None, {}
        losses = []
        actor_batches: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for _ in range(self.trainer_cfg.critic_updates_per_step):
            batch = self.buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, ids = batch
            values = self.critic.predict(states)
            next_values = self.critic.predict(next_states)
            targets = rewards + self.critic.config.gamma * (1.0 - dones) * next_values
            advantages = targets - values
            loss = self.critic.update(states, rewards, next_states, dones)
            losses.append(loss)
            for uid in np.unique(ids):
                mask = ids == uid
                actor_batches[uid] = (
                    states[mask],
                    actions[mask],
                    advantages[mask],
                )
        if losses:
            self.last_loss = float(np.mean(losses))
        log_loss = None
        if self.trainer_cfg.log_interval and self.step_count % self.trainer_cfg.log_interval == 0:
            log_loss = self.last_loss
        return log_loss, actor_batches
