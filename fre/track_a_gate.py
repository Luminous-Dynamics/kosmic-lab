"""Stage-1 Track A gating analysis for FRE corridor sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


LOG_DIR = Path("logs/fre_phase1")
GATE_STEPS = [50, 150, 300]


def load_passports(logdir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(logdir.glob("*.json")):
        data = json.loads(path.read_text())
        metrics = data.get("metrics", {})
        params = data.get("params", {})
        rows.append({"file": path.name, **params, **metrics})
    if not rows:
        raise FileNotFoundError(f"No JSON passports found in {logdir}")
    df = pd.DataFrame(rows)
    df = df.sort_values("file").reset_index(drop=True)
    return df


def jaccard(a: set[Tuple], b: set[Tuple]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compute_gate_metrics(df: pd.DataFrame, gate_sizes: List[int]) -> pd.DataFrame:
    param_cols = [c for c in df.columns if c.startswith("param_")]
    if not param_cols:
        param_cols = [
            c
            for c in df.columns
            if c in {"energy_gradient", "communication_cost", "plasticity_rate"}
        ]

    df["param_tuple"] = df[param_cols].apply(lambda r: tuple(np.round(r.values, 6)), axis=1)
    full_corridor = set(df.loc[df["in_corridor"] > 0.0, "param_tuple"].tolist())

    seen: List[Tuple] = []
    gate_results = []

    for idx, row in df.iterrows():
        seen.append(row["param_tuple"])
        if idx + 1 in gate_sizes:
            seen_df = df.iloc[: idx + 1]
            corridor = set(seen_df.loc[seen_df["in_corridor"] > 0.0, "param_tuple"].tolist())
            hit_rate = bool(len(seen_df)) and float(seen_df["in_corridor"].mean())
            gate_results.append(
                {
                    "samples": idx + 1,
                    "corridor_rate": hit_rate,
                    "jaccard_vs_full": jaccard(corridor, full_corridor),
                    "mean_K": float(seen_df["K"].mean()),
                }
            )

    return pd.DataFrame(gate_results)


def main() -> None:
    df = load_passports(LOG_DIR)
    summary = compute_gate_metrics(df, GATE_STEPS)
    print("Stage-1 Track A Gate Summary")
    print(summary.to_string(index=False))
    out_path = Path("logs/fre_phase1_trackA_gates.csv")
    summary.to_csv(out_path, index=False)
    print(f"Saved gate metrics to {out_path}")


if __name__ == "__main__":
    main()
