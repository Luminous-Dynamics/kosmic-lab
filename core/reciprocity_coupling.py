"""Reciprocal policy coupling utilities for Phase 2.5."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Tuple

import numpy as np

from core.te_bridge import knn_transfer_entropy


@dataclass
class PolicySnapshot:
    """Lightweight container describing a universe controller state."""

    params: Dict[str, np.ndarray]
    grads: Dict[str, np.ndarray]
    energy: float
    metadata: Dict[str, Any] | None = None


@dataclass
class CouplerConfig:
    """Configuration for the reciprocity coupler."""

    eta_min: float = 0.0
    eta_max: float = 0.05
    window: int = 32
    symmetry_lambda: float = 0.1
    projection: str = "identity"
    energy_cap: float = 1.0
    use_knn_te: bool = True
    k_neighbors: int = 5
    te_weight: float = 1.0
    gain_alpha: float = 5.0
    gain_tau: float = 0.0


class ReciprocityCoupler:
    """Compute cross-universe coupling updates using policy reciprocity."""

    def __init__(self, config: CouplerConfig | None = None) -> None:
        self.cfg = config or CouplerConfig()
        self.snapshots: Dict[str, PolicySnapshot] = {}
        self.grad_history: Dict[str, Deque[np.ndarray]] = {}
        self.group_order: List[str] | None = None
        self.trace: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_universe(self, uid: str, snapshot: PolicySnapshot) -> None:
        """Register a new universe with its initial snapshot."""

        if uid in self.snapshots:
            raise ValueError(f"Universe {uid} already registered")
        self.snapshots[uid] = snapshot
        self.grad_history[uid] = deque(maxlen=self.cfg.window)
        self._ensure_group_order(snapshot.grads.keys())
        self._append_history(uid, snapshot.grads)

    def update_snapshot(self, uid: str, snapshot: PolicySnapshot) -> None:
        """Update the stored snapshot and append gradient history."""

        if uid not in self.snapshots:
            raise KeyError(f"Universe {uid} is not registered")
        self.snapshots[uid] = snapshot
        self._ensure_group_order(snapshot.grads.keys())
        self._append_history(uid, snapshot.grads)

    def compute_coupling(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return parameter deltas to apply for each registered universe."""

        uids = list(self.snapshots.keys())
        deltas = {
            uid: {name: np.zeros_like(array) for name, array in snap.params.items()}
            for uid, snap in self.snapshots.items()
        }

        reciprocity = self._compute_reciprocity_matrix(uids)
        symmetry_lambda = float(self.cfg.symmetry_lambda)

        reciprocity_sym = reciprocity.copy()
        for i, uid_i in enumerate(uids):
            for j, uid_j in enumerate(uids):
                if i == j:
                    continue
                reciprocity_sym[i, j] = (1 - symmetry_lambda) * reciprocity[i, j] + symmetry_lambda * reciprocity[j, i]

        for j, uid_j in enumerate(uids):
            snap_j = self.snapshots[uid_j]
            energy_val = snap_j.energy
            energy_gate = self._energy_gate(energy_val)
            if energy_gate <= 0.0:
                continue

            for i, uid_i in enumerate(uids):
                if i == j:
                    continue

                eta = self._map_to_gain(reciprocity_sym[i, j]) * energy_gate
                if eta <= 0.0:
                    continue

                snap_i = self.snapshots[uid_i]
                for group in self._parameter_groups(snap_i, snap_j):
                    diff = self._projected_diff(group, snap_i.grads[group], snap_j.grads[group])
                    deltas[uid_j][group] += eta * diff
                    self.trace.append(
                        {
                            "source": uid_i,
                            "target": uid_j,
                            "group": group,
                            "reciprocity": reciprocity_sym[i, j],
                            "eta": eta,
                            "delta_norm": float(np.linalg.norm(diff) * eta),
                            "energy_gate": energy_gate,
                            "energy": energy_val,
                            "energy_cap": self.cfg.energy_cap,
                        }
                    )

        return deltas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_group_order(self, groups: Iterable[str]) -> None:
        if self.group_order is None:
            self.group_order = sorted(groups)

    def _append_history(self, uid: str, grads: Dict[str, np.ndarray]) -> None:
        vector = self._flatten_grads(grads)
        self.grad_history[uid].append(vector)

    def _flatten_grads(self, grads: Dict[str, np.ndarray]) -> np.ndarray:
        if self.group_order is None:
            raise RuntimeError("Group order not initialised")
        flat_arrays = []
        for name in self.group_order:
            if name not in grads:
                raise KeyError(f"Gradient group '{name}' missing in snapshot")
            flat_arrays.append(grads[name].ravel())
        return np.concatenate(flat_arrays)

    def _compute_reciprocity_matrix(self, uids: List[str]) -> np.ndarray:
        n = len(uids)
        matrix = np.zeros((n, n), dtype=float)
        for i, uid_i in enumerate(uids):
            for j, uid_j in enumerate(uids):
                if i == j:
                    continue
                matrix[i, j] = self._estimate_reciprocity(uid_i, uid_j)
        return matrix

    def _estimate_reciprocity(self, uid_i: str, uid_j: str) -> float:
        window = self.cfg.window
        hist_i = self.grad_history[uid_i]
        hist_j = self.grad_history[uid_j]
        if len(hist_i) < window or len(hist_j) < window:
            return 0.0

        grad_i = np.asarray(list(hist_i)[-window:])
        grad_j = np.asarray(list(hist_j)[-window:])

        te_weight = float(self.cfg.te_weight)
        reciprocity = 0.0

        if self.cfg.use_knn_te and te_weight > 0.0:
            try:
                te_val = knn_transfer_entropy(
                    grad_i[:-1],
                    grad_j[:-1],
                    grad_j[1:],
                    k=self.cfg.k_neighbors,
                )
            except ValueError:
                te_val = 0.0
            reciprocity += te_weight * te_val

        if te_weight < 1.0:
            remaining = 1.0 - te_weight
            reciprocity += remaining * self._cosine_similarity(grad_i[-1], grad_j[-1])

        return float(max(0.0, reciprocity))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
        return num / den

    def _parameter_groups(self, snap_i: PolicySnapshot, snap_j: PolicySnapshot) -> List[str]:
        groups = set(snap_i.grads.keys()) & set(snap_j.grads.keys())
        if self.cfg.projection == "actor_only":
            groups = {g for g in groups if "actor" in g}
        return sorted(groups)

    def _projected_diff(self, group: str, grad_i: np.ndarray, grad_j: np.ndarray) -> np.ndarray:
        if grad_i.shape != grad_j.shape:
            raise ValueError(f"Gradient shapes differ for group '{group}' ({grad_i.shape} vs {grad_j.shape})")
        # Future extensions (attention, low rank) hook here.
        return grad_i - grad_j

    def _map_to_gain(self, reciprocity: float) -> float:
        eta_min = float(self.cfg.eta_min)
        eta_max = float(self.cfg.eta_max)
        if eta_max <= eta_min:
            return 0.0
        alpha = float(self.cfg.gain_alpha)
        tau = float(self.cfg.gain_tau)
        sigma = 1.0 / (1.0 + np.exp(-alpha * (reciprocity - tau)))
        return float(eta_min + (eta_max - eta_min) * sigma)

    def _energy_gate(self, energy: float) -> float:
        cap = max(self.cfg.energy_cap, 1e-6)
        gate = 1.0 - max(0.0, energy) / cap
        return float(max(0.0, gate))
