from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import load_yaml_config
from historical_k.etl import build_harmony_frame, compute_k_series


def _compute_for_strategy(payload: Dict, years: List[int], strategy: str) -> pd.DataFrame:
    win = payload.get("windows", {})
    overrides = win.get("normalization_overrides", {})
    frame = build_harmony_frame(payload.get("proxies", {}), years, strategy, overrides)
    k = compute_k_series(frame)
    df = frame.copy().reset_index(names="year")
    df[f"K_{strategy}"] = k.to_numpy()
    return df[["year", f"K_{strategy}"]]


def _slope_1950_1990(df: pd.DataFrame, col: str) -> float:
    seg = df[(df["year"] >= 1950) & (df["year"] <= 1990)]
    if len(seg) < 3:
        return float("nan")
    x = seg["year"].to_numpy()
    y = seg[col].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def main() -> None:
    cfg = load_yaml_config(Path("historical_k/k_config.yaml"))
    payload = cfg.payload

    # Use prereg years to set span; fallback to 1800–2020
    events = payload.get("preregistered_events", {})
    cand = []
    for v in events.values():
        cand.extend(v)
    if cand:
        start = int(min(cand) // 10 * 10)
        end = int(max(cand) // 10 * 10)
    else:
        start, end = 1800, 2020
    years = list(range(start, end + 10, 10))

    strategies = [
        "minmax_by_century",
        "zscore_by_century",
        "zscore_rolling_3",
    ]

    # Compute per-strategy series
    series_list = []
    for s in strategies:
        df = _compute_for_strategy(payload, years, s)
        series_list.append(df)

    merged = series_list[0]
    for df in series_list[1:]:
        merged = merged.merge(df, on="year", how="inner")

    outdir = Path("logs/historical_k")
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outdir / "compare_series.csv", index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    for s in strategies:
        plt.plot(merged["year"], merged[f"K_{s}"], marker="o", label=s)
    plt.xlabel("Year")
    plt.ylabel("K-index (normalized)")
    plt.title("Historical K(t) – Normalization Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "k_compare.png")
    plt.close()

    # Summaries
    summary = {}
    for s in strategies:
        col = f"K_{s}"
        summary[s] = {
            "mean": float(merged[col].mean()),
            "std": float(merged[col].std(ddof=0)),
            "slope_1950_1990": _slope_1950_1990(merged, col),
        }

    (outdir / "compare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

