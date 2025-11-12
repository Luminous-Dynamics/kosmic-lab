"""Run scaling-law experiment for FRE Phase 2 using coupled universes."""

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


def run_scaling(config_path: Path) -> pd.DataFrame:
    config_bundle = load_yaml_config(config_path)
    config = config_bundle.payload
    scaling_cfg = config.get("scaling", {})
    universe_counts = scaling_cfg.get("universe_counts", [3, 5, 7, 10])
    raw_params: Dict[str, float] = config.get("parameters", {})
    base_params: Dict[str, float] = {}
    bounds: Dict[str, Tuple[float, float]] = {}
    for key, value in raw_params.items():
        if isinstance(value, (list, tuple)):
            base_params[key] = float(np.mean(value))
            bounds[key] = (float(min(value)), float(max(value)))
        else:
            base_params[key] = float(value)
            span = abs(base_params[key]) if base_params[key] != 0 else 0.5
            bounds[key] = (base_params[key] - span, base_params[key] + span)
    weights = _weights_from_config(config.get("k_weights", {}))

    coupling_cfg = config.get("coupling", {})
    coupling_mode = coupling_cfg.get("mode", config.get("coupling_mode", "rpc")).lower()
    coupling_strength = float(coupling_cfg.get("strength", config.get("coupling_strength", 0.1)))
    te_cfg = coupling_cfg.get("te_bridge", config.get("te_bridge", {}))
    rpc_cfg = coupling_cfg.get("rpc", {})
    phase4_cfg = config.get("phase4", {})

    rows = []
    for count in universe_counts:
        seeds = [int(config.get("seed_base", 44)) + 100 * count + i for i in range(count)]
        simulator = MultiUniverseSimulator(
            universe_count=count,
            base_params=base_params,
            param_bounds=bounds,
            seeds=seeds,
            coupling_strength=coupling_strength,
            harmonics_calculator=HarmonyCalculator(),
            coupling_mode=coupling_mode,
            te_config=te_cfg,
            rpc_config=rpc_cfg,
            phase4_config=phase4_cfg,
        )
        histories = simulator.run(weights, steps=50)
        if simulator.phase4_loss_log:
            avg_loss = float(np.mean([loss for _, loss in simulator.phase4_loss_log]))
            last_step, last_loss = simulator.phase4_loss_log[-1]
            print(f"Phase4 critic loss (avg): {avg_loss:.4f}, last step {last_step} loss {last_loss:.4f}")
        mean_k = []
        for hist in histories.values():
            k_vals = [scores.kosmic_signature(weights=weights) for scores in hist]
            mean_k.append(float(np.mean(k_vals)))
        rows.append({
            "universe_count": count,
            "mean_K": float(np.mean(mean_k)),
            "sum_K": float(np.sum(mean_k)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="FRE scaling experiment")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    df = run_scaling(args.config)
    print(df.to_string(index=False))
    output = load_yaml_config(args.config).payload.get("output", {}).get("summary_csv")
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved scaling summary to {out_path}")


if __name__ == "__main__":
    main()
