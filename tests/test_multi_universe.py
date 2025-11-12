from __future__ import annotations

import numpy as np

from fre.multi_universe import MultiUniverseSimulator


def test_rpc_diffusion_updates_all_groups() -> None:
    simulator = MultiUniverseSimulator(
        universe_count=2,
        base_params={"energy_gradient": 0.5, "plasticity_rate": 0.15},
        param_bounds={
            "energy_gradient": (0.0, 1.0),
            "plasticity_rate": (0.0, 1.0),
        },
        seeds=[10, 20],
        coupling_strength=0.0,
        coupling_mode="rpc",
        rpc_config={"diffusion_strength": 0.5, "ema_alpha": 0.2},
    )
    simulator.universes[0].params["energy_gradient"] = 0.0
    simulator.universes[0].params["plasticity_rate"] = 0.0
    simulator.universes[1].params["energy_gradient"] = 0.6
    simulator.universes[1].params["plasticity_rate"] = 0.2

    weights = np.ones(7, dtype=float) / 7.0
    before_energy = simulator.universes[0].params["energy_gradient"]
    before_plasticity = simulator.universes[0].params["plasticity_rate"]

    simulator.step(0, weights)

    assert simulator.universes[0].params["energy_gradient"] > before_energy
    assert simulator.universes[0].params["plasticity_rate"] > before_plasticity
