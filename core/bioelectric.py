"""Bioelectric grid simulation utilities for Track C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def compute_iou(target_mask: np.ndarray, current_mask: np.ndarray) -> float:
    """Return intersection-over-union between two boolean masks."""
    intersection = np.logical_and(target_mask, current_mask).sum()
    union = np.logical_or(target_mask, current_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def mask_from_voltage(V: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Binary mask where |V| exceeds threshold."""
    return np.abs(V) >= threshold


@dataclass
class AnnealState:
    gain: float = 0.0
    steps_remaining: int = 0


class BioelectricGrid:
    """2D voltage grid with diffusion, leak, and stimulation operators."""

    def __init__(
        self,
        shape: Tuple[int, int],
        *,
        D: float = 0.12,
        g: float = 0.08,
        dt: float = 0.1,
        alpha: float = 0.0,
        beta: float = 1.0,
        bc: str = "neumann",
        leak_reversal: float = 0.0,
    ) -> None:
        self.shape = tuple(shape)
        self.V = np.zeros(self.shape, dtype=float)
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.bc = bc

        self.D_scalar = float(D)
        self.D_map = np.full(self.shape, float(D), dtype=float)
        self.g = float(g)
        self.leak_reversal = float(leak_reversal)  # V3: Attractor-based rescue
        self.anneal_state = AnnealState()

        yy, xx = np.indices(self.shape)
        self._grid_indices = (yy, xx)

    def step(self, I_stim: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance the grid by one time step, returning the updated voltage."""
        lap = self._laplacian(self.V)
        ion = self.alpha * np.tanh(self.beta * self.V)
        stim = I_stim if I_stim is not None else 0.0
        noise = np.random.normal(0.0, 1e-3, self.shape)

        g_eff = self.g
        if self.anneal_state.steps_remaining > 0:
            g_eff = self.g * (1.0 - self.anneal_state.gain)
            self.anneal_state.steps_remaining -= 1

        # V3 ATTRACTOR-BASED RESCUE: Leak pulls toward leak_reversal (default 0.0)
        # This creates a stable attractor that rescue can modify
        self.V += self.dt * (
            self.D_map * lap - g_eff * (self.V - self.leak_reversal) + ion + stim + noise
        )
        # CRITICAL FIX (2025-11-09): Use biological voltage scale (-100 to 0 mV)
        # Was: np.clip(self.V, -1.0, 1.0, out=self.V) - prevented hyperpolarization!
        np.clip(self.V, -100.0, 0.0, out=self.V)
        return self.V

    def stimulate(self, mask: np.ndarray, amplitude: float, radius: float) -> np.ndarray:
        """Return a stimulation current shaped around the True pixels of mask."""
        if not mask.any() or amplitude == 0.0:
            return np.zeros(self.shape, dtype=float)

        yy, xx = self._grid_indices
        coords = np.argwhere(mask)
        dist2 = np.full(self.shape, np.inf, dtype=float)
        for y, x in coords:
            d2 = (yy - y) ** 2 + (xx - x) ** 2
            dist2 = np.minimum(dist2, d2)

        if radius <= 0:
            stim = np.zeros_like(dist2)
            stim[mask] = amplitude
            return stim

        sigma2 = max(radius, 1.0) ** 2 / 2.0
        stim = amplitude * np.exp(-dist2 / sigma2)
        return stim

    def rewire(self, mask: np.ndarray, rho: float, radius: int = 1) -> None:
        """Increase diffusion constant locally to mimic gap-junction rewiring."""
        if rho == 0.0 or not mask.any():
            return
        yy, xx = self._grid_indices
        coords = np.argwhere(mask)
        for y, x in coords:
            y0, y1 = max(0, y - radius), min(self.shape[0], y + radius + 1)
            x0, x1 = max(0, x - radius), min(self.shape[1], x + radius + 1)
            self.D_map[y0:y1, x0:x1] *= 1.0 + rho
        np.clip(self.D_map, self.D_scalar * 0.5, self.D_scalar * 5.0, out=self.D_map)

    def anneal(self, gamma: float, steps: int) -> None:
        """Temporarily reduce leak conductance to allow plastic relaxation."""
        if gamma <= 0.0 or steps <= 0:
            return
        self.anneal_state = AnnealState(gain=float(gamma), steps_remaining=int(steps))

    def _laplacian(self, A: np.ndarray) -> np.ndarray:
        if self.bc == "periodic":
            return (
                -4.0 * A
                + np.roll(A, 1, axis=0)
                + np.roll(A, -1, axis=0)
                + np.roll(A, 1, axis=1)
                + np.roll(A, -1, axis=1)
            )
        # default: Neumann (zero-flux) via edge padding
        pad = np.pad(A, 1, mode="edge")
        lap = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4.0 * A
        return lap
