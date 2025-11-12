"""Quick monitor for FRE corridor metrics in Holochain prototype."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_metrics(path: Path) -> pd.DataFrame:
    records = []
    for file in path.glob("*.json"):
        try:
            data = json.loads(file.read_text())
        except json.JSONDecodeError:
            continue
        metrics = data.get("metrics", {})
        params = data.get("params", {})
        records.append({"file": file.name, **metrics, **params})
    return pd.DataFrame(records)


def main() -> None:
    log_dir = Path("logs/fre_phase1")
    df = load_metrics(log_dir)
    if df.empty:
        print("No metrics found in logs/fre_phase1")
        return
    corridor_rate = df.get("in_corridor", 0).mean()
    mean_k = df.get("K", pd.Series()).mean()
    print(f"Corridor rate: {corridor_rate:.3f}")
    print(f"Mean K: {mean_k:.3f}")


if __name__ == "__main__":
    main()
