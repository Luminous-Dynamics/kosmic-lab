"""Training harness for Phase 4 centralized coordination."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from core.config import load_yaml_config
from fre.multi_universe import MultiUniverseSimulator
from fre.simulate import _weights_from_config
from harmonics_module import HarmonyCalculator


def build_base_params(
    raw_params: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    base_params: Dict[str, float] = {}
    bounds: Dict[str, Tuple[float, float]] = {}
    for key, value in raw_params.items():
        if isinstance(value, (list, tuple)) and value:
            base_params[key] = float(np.mean(value))
            bounds[key] = (float(min(value)), float(max(value)))
        else:
            base_params[key] = float(value)
            span = abs(base_params[key]) if base_params[key] != 0 else 0.5
            bounds[key] = (base_params[key] - span, base_params[key] + span)
    return base_params, bounds


def train(config_path: Path) -> None:
    cfg_bundle = load_yaml_config(config_path)
    cfg = cfg_bundle.payload

    raw_params = cfg.get("parameters", {})
    base_params, bounds = build_base_params(raw_params)
    weights = _weights_from_config(cfg.get("k_weights", {}))

    coupling_cfg = cfg.get("coupling", {})
    phase4_cfg = cfg.get("phase4", {})
    training_cfg = cfg.get("training", {})

    universe_count = int(training_cfg.get("universe_count", 5))
    total_steps = int(training_cfg.get("total_steps", 5000))
    eval_interval = int(training_cfg.get("eval_interval", 500))
    seed_base = int(training_cfg.get("seed_base", 44))

    seeds = [seed_base + i for i in range(universe_count)]
    coupling_strength = float(coupling_cfg.get("strength", 0.1))
    te_cfg = coupling_cfg.get("te_bridge", {})
    rpc_cfg = coupling_cfg.get("rpc", {})

    simulator = MultiUniverseSimulator(
        universe_count=universe_count,
        base_params=base_params,
        param_bounds=bounds,
        seeds=seeds,
        coupling_strength=coupling_strength,
        harmonics_calculator=HarmonyCalculator(),
        coupling_mode=coupling_cfg.get("mode", "rpc"),
        te_config=te_cfg,
        rpc_config=rpc_cfg,
        phase4_config=phase4_cfg,
    )

    records = []
    for t in range(total_steps):
        harmonies = simulator.step(t, weights)
        mean_k = float(
            np.mean([scores.kosmic_signature(weights=weights) for scores in harmonies.values()])
        )
        loss = simulator.phase4_last_loss
        records.append(
            {"step": t, "mean_K": mean_k, "critic_loss": loss if loss is not None else np.nan}
        )

        if eval_interval and (t + 1) % eval_interval == 0:
            print(
                f"Step {t+1}: mean K {mean_k:.4f}, critic loss {loss if loss is not None else float('nan'):.4f}"
            )

    df = pd.DataFrame(records)
    log_path = cfg.get("output", {}).get("training_log")
    if log_path:
        out_path = Path(log_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved training log to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase4 centralized coordination")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_experiment.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
