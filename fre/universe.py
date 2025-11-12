from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


def _gaussian_closeness(value: float, center: float, width: float) -> float:
    return float(np.exp(-((value - center) ** 2) / (2 * width ** 2)))


@dataclass
class UniverseSimulator:
    """Lightweight physics-inspired simulator for FRE Phase 1."""

    centers: Dict[str, float] = field(
        default_factory=lambda: {
            "energy_gradient": 0.5,
            "communication_cost": 0.3,
            "plasticity_rate": 0.15,
            "noise_spectrum_alpha": 0.0,
            "stimulus_jitter": 0.05,
        }
    )
    widths: Dict[str, float] = field(
        default_factory=lambda: {
            "energy_gradient": 0.12,
            "communication_cost": 0.08,
            "plasticity_rate": 0.05,
            "noise_spectrum_alpha": 0.4,
            "stimulus_jitter": 0.06,
        }
    )
    noise_sigma: float = 0.03

    def run(self, params: Dict[str, float], seed: int) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        closeness = {
            key: _gaussian_closeness(
                float(params.get(key, center)), center, self.widths[key]
            )
            for key, center in self.centers.items()
        }

        cohesion = np.mean(
            [closeness["energy_gradient"], closeness["communication_cost"], closeness["plasticity_rate"]]
        )
        reciprocity = closeness["noise_spectrum_alpha"]
        playfulness = closeness["stimulus_jitter"]

        phi = float(np.clip(2.5 * cohesion + rng.normal(0, self.noise_sigma), 0.0, 3.0))
        te_mutual = float(np.clip(1.2 * cohesion + rng.normal(0, self.noise_sigma), 0.0, 2.0))
        te_symmetry = float(np.clip((cohesion + reciprocity) / 2.0, 0.0, 1.0))

        # Compose K from harmonies (simplified weights)
        k = float(
            np.clip(
                0.35 * cohesion
                + 0.2 * te_symmetry
                + 0.15 * (1.0 - abs(params.get("noise_spectrum_alpha", 0.0)))
                + 0.15 * (1.0 - params.get("stimulus_jitter", 0.05))
                + 0.15 * te_mutual / 2.0
                + rng.normal(0, self.noise_sigma),
                0.0,
                2.5,
            )
        )

        tat = float(np.clip(k / 2.0, 0.0, 1.0))
        recovery = float(np.maximum(0.5, 2.2 - k))

        return {
            "phi": phi,
            "te_mutual": te_mutual,
            "te_symmetry": te_symmetry,
            "cohesion": cohesion,
            "reciprocity": reciprocity,
            "playfulness": playfulness,
            "K": k,
            "TAT": tat,
            "Recovery": recovery,
        }
