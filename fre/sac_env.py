"""Gymnasium environment for Track C SAC controller experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.bioelectric import BioelectricGrid, compute_iou, mask_from_voltage


@dataclass
class EnvConfig:
    shape: tuple[int, int] = (32, 32)
    diffusion: float = 0.12
    leak: float = 0.08
    dt: float = 0.1
    stim_max: float = 1.0
    stim_radius_max: float = 5.0
    reward_weights: Dict[str, float] = None
    threshold: float = 0.1
    horizon: int = 100
    lesion_radius_scale: float = 1.0
    lesion_count: int = 1

    def __post_init__(self) -> None:
        if self.reward_weights is None:
            self.reward_weights = {"lambda_k": 1.0, "lambda_energy": 0.01}


class KosmicRescueEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.grid = BioelectricGrid(self.cfg.shape, D=self.cfg.diffusion, g=self.cfg.leak, dt=self.cfg.dt)
        self.target_mask = self._make_target_mask()
        self.state_mask = np.zeros(self.cfg.shape, dtype=bool)
        self.current_iou = 0.0
        self.prev_iou = 0.0
        self.steps = 0

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.grid = BioelectricGrid(self.cfg.shape, D=self.cfg.diffusion, g=self.cfg.leak, dt=self.cfg.dt)
        self.grid.V.fill(0.0)
        lesion = self._make_lesion_mask(rng)
        self.grid.V = lesion.astype(float) * 0.5
        for _ in range(20):
            self.grid.step()
        self.state_mask = mask_from_voltage(self.grid.V, self.cfg.threshold)
        self.prev_iou = compute_iou(self.target_mask, self.state_mask)
        self.current_iou = self.prev_iou
        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        delta_iou = self.current_iou - self.prev_iou
        energy_use = float(np.clip(np.abs(self.grid.V).mean(), 0.0, 1.0))
        obs = np.array([self.current_iou, delta_iou, energy_use], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        stim_amp = (action[0] + 1.0) / 2.0 * self.cfg.stim_max
        stim_radius = (action[1] + 1.0) / 2.0 * self.cfg.stim_radius_max

        perimeter = np.logical_and(self.target_mask, ~self.state_mask)
        I = self.grid.stimulate(perimeter, stim_amp, stim_radius)
        energy_cost = float((I ** 2).mean())
        self.grid.step(I_stim=I)
        self.state_mask = mask_from_voltage(self.grid.V, self.cfg.threshold)
        self.prev_iou = self.current_iou
        self.current_iou = compute_iou(self.target_mask, self.state_mask)
        delta_iou = self.current_iou - self.prev_iou

        rw = self.cfg.reward_weights
        reward = rw["lambda_k"] * delta_iou - rw["lambda_energy"] * energy_cost

        self.steps += 1
        terminated = self.current_iou >= 0.99
        truncated = self.steps >= self.cfg.horizon
        obs = self._get_obs()
        info = {"iou": self.current_iou}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def _make_target_mask(self) -> np.ndarray:
        yy, xx = np.indices(self.cfg.shape)
        center = np.array([self.cfg.shape[0] // 2, self.cfg.shape[1] // 2])
        mask = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= (self.cfg.shape[0] // 3) ** 2
        return mask

    def _make_lesion_mask(self, rng: np.random.Generator) -> np.ndarray:
        mask = self._make_target_mask().copy()
        yy, xx = np.indices(self.cfg.shape)
        shift = rng.integers(-5, 5, size=2)
        center = np.array([self.cfg.shape[0] // 2, self.cfg.shape[1] // 2]) + shift
        base_radius = max(2, int((self.cfg.shape[0] // 5) * self.cfg.lesion_radius_scale))
        for _ in range(max(1, self.cfg.lesion_count)):
            lesion = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= base_radius ** 2
            mask[lesion] = False
            # randomise next lesion centre slightly for multi-lesion scenarios
            center = center + rng.integers(-4, 4, size=2)
            center = np.clip(center, [0, 0], np.array(self.cfg.shape) - 1)
        return mask


def make_env(cfg_dict: dict, overrides: Dict[str, float] | None = None) -> gym.Env:
    env_cfg = EnvConfig(
        shape=tuple(cfg_dict.get("bioelectric", {}).get("shape", [32, 32])),
        diffusion=cfg_dict.get("bioelectric", {}).get("D", 0.12),
        leak=cfg_dict.get("bioelectric", {}).get("g", 0.08),
        dt=cfg_dict.get("bioelectric", {}).get("dt", 0.1),
        stim_max=cfg_dict.get("rescue", {}).get("stim_amplitude_max", 1.0),
        stim_radius_max=cfg_dict.get("rescue", {}).get("stim_radius_max", 5.0),
        reward_weights=cfg_dict.get("sac", {}).get("reward_weights", None),
        threshold=cfg_dict.get("rescue", {}).get("threshold", 0.1),
        horizon=cfg_dict.get("experiment", {}).get("horizon", 180),
        lesion_radius_scale=cfg_dict.get("rescue", {}).get("lesion_radius_scale", 1.0),
        lesion_count=cfg_dict.get("rescue", {}).get("lesion_count", 1),
    )
    if overrides:
        for key, value in overrides.items():
            if hasattr(env_cfg, key) and value is not None:
                setattr(env_cfg, key, value)
    return KosmicRescueEnv(env_cfg)
