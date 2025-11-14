"""
Soft Actor-Critic controller scaffold for Track B.

This module provides class stubs (replay buffer, networks, controller)
that can be filled in with PyTorch or another deep learning framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.distributions import Normal


@dataclass
class Transition:
    state: List[float]
    action: List[float]
    reward: float
    next_state: List[float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough transitions to sample the requested batch.")
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class SACController:
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        *,
        hidden_layers: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        buffer_capacity: int = 1_000_000,
        init_temperature: float = 0.1,
        actor_clip_norm: float | None = 1.0,
        critic_clip_norm: float | None = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device or torch.device("cpu")
        if init_temperature <= 0.0:
            raise ValueError("init_temperature must be positive.")

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_entropy = -float(action_dim)

        self.actor = GaussianActor(state_dim, action_dim, hidden_layers).to(self.device)
        self.critic_1 = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.critic_2 = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_critic_1 = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_critic_2 = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.critic_opt = optim.Adam(critic_params, lr=critic_lr)

        self.log_alpha = torch.tensor(
            np.log(init_temperature), device=self.device, requires_grad=True
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.actor_clip_norm = float(actor_clip_norm) if actor_clip_norm is not None else None
        self.critic_clip_norm = float(critic_clip_norm) if critic_clip_norm is not None else None

    def select_action(self, state: List[float], evaluate: bool = False) -> List[float]:
        self.actor.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, deterministic = self.actor.sample(state_tensor)
        self.actor.train()
        chosen = deterministic if evaluate else action
        return chosen.squeeze(0).cpu().numpy().tolist()

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        states = torch.as_tensor([t.state for t in batch], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        next_states = torch.as_tensor(
            [t.next_state for t in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [float(t.done) for t in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_states)
            target_q1 = self.target_critic_1(next_states, next_action)
            target_q2 = self.target_critic_2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = rewards + (1.0 - dones) * self.gamma * target_q

        # critic update
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = nn.functional.mse_loss(current_q1, target_value) + nn.functional.mse_loss(
            current_q2, target_value
        )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.critic_clip_norm is not None and self.critic_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.critic_1.parameters(), max_norm=self.critic_clip_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.critic_2.parameters(), max_norm=self.critic_clip_norm
            )
        self.critic_opt.step()

        # actor update
        new_action, log_prob, _ = self.actor.sample(states)
        q1_new = self.critic_1(states, new_action)
        q2_new = self.critic_2(states, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_clip_norm is not None and self.actor_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_clip_norm)
        self.actor_opt.step()

        # temperature (entropy coefficient) update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.target_critic_1, self.critic_1)
        self._soft_update(self.target_critic_2, self.critic_2)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha),
        }

    def store_transition(self, transition: Transition) -> None:
        self.replay_buffer.push(transition)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = state_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        self.backbone = nn.Sequential(*layers)
        self.backbone_out_dim = prev_dim
        self.mean_head = nn.Linear(self.backbone_out_dim, action_dim)
        self.log_std_head = nn.Linear(self.backbone_out_dim, action_dim)
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.backbone(state) if len(self.backbone) > 0 else state
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        noise = dist.rsample()
        action = torch.tanh(noise)
        log_prob = dist.log_prob(noise) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mean)
        return action, log_prob, deterministic


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = state_dim + action_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
