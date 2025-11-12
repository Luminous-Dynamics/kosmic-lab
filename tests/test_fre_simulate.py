from __future__ import annotations

import pytest

from fre.simulate import (
    DEFAULT_SEED_BASE,
    _weights_from_config,
    compute_metrics,
    expand_parameter_grid,
    simulate_phase1,
)
from fre.universe import UniverseSimulator


def test_expand_parameter_grid_scalar_and_list() -> None:
    params = {
        "energy": 0.5,
        "noise": [0.0, 0.1],
    }
    grid = expand_parameter_grid(params)
    assert len(grid) == 2
    assert grid[0]["energy"] == 0.5
    assert grid[1]["noise"] == 0.1


def test_simulate_phase1_returns_runs() -> None:
    config = {
        "parameters": {"energy_gradient": [0.4, 0.6], "noise_spectrum_alpha": 0.1},
        "corridor_threshold": 1.0,
        "seed_base": 100,
        "seeds_per_point": 2,
        "k_weights": {"resonant_coherence": 0.4, "pan_sentient_flourishing": 0.3},
    }
    runs = simulate_phase1(config)
    assert len(runs) == 4
    assert runs[0]["seed"] == 100
    assert runs[0]["metrics"]["K"] != runs[1]["metrics"]["K"]


def test_compute_metrics_reproducible() -> None:
    params = {"energy_gradient": 0.5, "communication_cost": 0.3}
    simulator = UniverseSimulator()
    metrics1 = compute_metrics(params, DEFAULT_SEED_BASE, simulator)
    metrics2 = compute_metrics(params, DEFAULT_SEED_BASE, simulator)
    assert metrics1 == metrics2
    assert 0.0 <= metrics1["K"] <= 2.5
    assert "H1_Coherence" in metrics1


def test_weights_from_config_rejects_zero_sum() -> None:
    zero_weights = {
        "resonant_coherence": 0.0,
        "pan_sentient_flourishing": 0.0,
        "integral_wisdom": 0.0,
        "infinite_play": 0.0,
        "universal_interconnectedness": 0.0,
        "sacred_reciprocity": 0.0,
        "evolutionary_progression": 0.0,
    }
    with pytest.raises(ValueError):
        _weights_from_config(zero_weights)
