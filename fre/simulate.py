"""Structured FRE Phase 1 simulation hooks."""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from fre.universe import UniverseSimulator
from harmonics_module import HarmonyCalculator


CALCULATOR = HarmonyCalculator()
GLOBAL_TIMESTEP = 0


DEFAULT_SEED_BASE = 44


def simulate_phase1(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute a Phase 1 sweep for all parameter combinations defined in config.

    Args:
        config: Parsed YAML configuration payload.

    Returns:
        List of per-run dictionaries containing parameters, seed, and metrics.
    """
    param_grid = expand_parameter_grid(config.get("parameters", {}))
    seed_base = int(config.get("seed_base", DEFAULT_SEED_BASE))
    threshold = float(config.get("corridor_threshold", 1.0))
    simulator = UniverseSimulator()
    weights = _weights_from_config(config.get("k_weights", {}))

    seeds_per_point = int(config.get("seeds_per_point", 1))
    results: List[Dict[str, Any]] = []
    for idx, params in enumerate(param_grid):
        for offset in range(seeds_per_point):
            seed = seed_base + idx * max(1, seeds_per_point) + offset
            metrics = compute_metrics(params, seed, simulator, weights)
            metrics["in_corridor"] = metrics["K"] >= threshold
            results.append(
                {
                    "seed": seed,
                    "parameters": params,
                    "metrics": metrics,
                }
            )
    return results


def expand_parameter_grid(parameters: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    Expand configuration dictionary into a Cartesian product of parameter sets.

    Each value can be a scalar/int/float or an iterable; iterables are treated as
    a sweep across the provided values.
    """
    if not parameters:
        return [{}]

    keys: List[str] = []
    value_lists: List[List[float]] = []
    for key, value in parameters.items():
        keys.append(key)
        if isinstance(value, (list, tuple)):
            value_lists.append([float(v) for v in value])
        else:
            value_lists.append([float(value)])

    combinations: Iterable[Tuple[float, ...]] = product(*value_lists)
    return [dict(zip(keys, combo)) for combo in combinations]


def compute_metrics(
    params: Dict[str, float],
    seed: int,
    simulator: UniverseSimulator | None = None,
    weights: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute harmonies and derived metrics for a parameter set."""
    global GLOBAL_TIMESTEP

    simulator = simulator or UniverseSimulator()
    sim_metrics = simulator.run(params, seed)
    GLOBAL_TIMESTEP += 1
    harmony_inputs = _synthetic_harmony_inputs(sim_metrics, params, seed, GLOBAL_TIMESTEP)
    scores = CALCULATOR.compute_all(**harmony_inputs)
    weights = weights if weights is not None else np.ones(7) / 7.0
    k_value = float(np.clip(scores.kosmic_signature(weights=weights) * 1.2, 0.0, 2.5))

    metrics = {
        "K": k_value,
        "TAT": float(np.clip(k_value / 2.0, 0.0, 1.0)),
        "Recovery": float(np.maximum(0.5, 2.2 - k_value)),
        "AUC_K": float(k_value * 80.0),
        "TE_macro_micro": sim_metrics["te_mutual"],
        "TE_symmetry": sim_metrics["te_symmetry"],
        "phi": sim_metrics["phi"],
        "cohesion": sim_metrics["cohesion"],
        "reciprocity": sim_metrics["reciprocity"],
        "playfulness": sim_metrics["playfulness"],
    }
    metrics.update(scores.to_dict())
    return metrics


def _synthetic_harmony_inputs(
    sim_metrics: Dict[str, float], params: Dict[str, float], seed: int, timestep: int
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    agent_count = 50
    cohesion = np.clip(sim_metrics["cohesion"], 0.0, 1.0)
    reciprocity = np.clip(sim_metrics["reciprocity"], 0.0, 1.0)
    playfulness = np.clip(sim_metrics["playfulness"], 0.0, 1.0)
    alive_agents = max(5, int(agent_count * (0.4 + 0.5 * cohesion)))
    dead_agents = agent_count - alive_agents

    agent_types = ["explorer", "cooperator", "mediator"]
    type_probs = np.array([playfulness + 0.1, cohesion + 0.1, reciprocity + 0.1])
    type_probs = type_probs / type_probs.sum()
    agent_states: List[Dict[str, Any]] = []
    for _ in range(alive_agents):
        agent_states.append({"alive": True, "type": rng.choice(agent_types, p=type_probs)})
    for _ in range(dead_agents):
        agent_states.append({"alive": False, "type": rng.choice(agent_types)})

    te_matrix = rng.normal(loc=sim_metrics["te_mutual"], scale=0.05, size=(agent_count, agent_count))
    te_matrix = np.clip(te_matrix, 0.0, None)
    np.fill_diagonal(te_matrix, 0.0)

    prediction_errors = {
        "sensory": float(np.clip(1 - cohesion, 0.0, 1.0)),
        "motor": float(np.clip(1 - reciprocity, 0.0, 1.0)),
        "social": float(np.clip(1 - playfulness, 0.0, 1.0)),
    }

    history_length = 15
    action_dim = 3
    behavioral_history = [
        rng.normal(loc=0.0, scale=0.5 + playfulness, size=(agent_count, action_dim))
        for _ in range(history_length)
    ]

    return {
        "phi": sim_metrics["phi"],
        "agent_states": agent_states,
        "te_matrix": te_matrix,
        "prediction_errors": prediction_errors,
        "behavioral_history": behavioral_history,
        "timestep": timestep,
    }


def _weights_from_config(weights_cfg: Dict[str, float]) -> np.ndarray:
    if not weights_cfg:
        return np.ones(7) / 7.0
    order = [
        "resonant_coherence",
        "pan_sentient_flourishing",
        "integral_wisdom",
        "infinite_play",
        "universal_interconnectedness",
        "sacred_reciprocity",
        "evolutionary_progression",
    ]
    arr = np.array([float(weights_cfg.get(key, 1.0)) for key in order], dtype=float)
    total = arr.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        raise ValueError("K-weight configuration must have a non-zero finite sum.")
    arr = arr / total
    return arr
