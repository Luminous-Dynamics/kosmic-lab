from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fre.corridor import (
    CorridorSummary,
    compare_to_baseline,
    compute_corridor_metrics,
    save_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate FRE Phase 1 runs from K-passport logs."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("logs/fre_phase1"),
        help="Directory containing passport JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/fre_phase1_summary.json"),
        help="Destination for aggregated summary JSON.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline corridor CSV for comparison.",
    )
    parser.add_argument(
        "--export-corridor",
        type=Path,
        help="Optional path to export current corridor points (CSV).",
    )
    return parser.parse_args()


def load_passports(logdir: Path) -> pd.DataFrame:
    records: List[Dict] = []
    for path in sorted(logdir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        record = {
            "run_id": data.get("run_id"),
            "seed": data.get("seed"),
            "K": data.get("metrics", {}).get("K"),
            "TAT": data.get("metrics", {}).get("TAT"),
            "Recovery": data.get("metrics", {}).get("Recovery"),
            "in_corridor": data.get("metrics", {}).get("in_corridor", False),
        }
        params = data.get("params", {})
        for key, value in params.items():
            record[f"param_{key}"] = value
        records.append(record)

    if not records:
        raise FileNotFoundError(f"No passport JSON files found in {logdir}")
    return pd.DataFrame(records)


def compute_summary(df: pd.DataFrame, param_columns: List[str], threshold: float) -> CorridorSummary:
    return compute_corridor_metrics(df, threshold=threshold, param_columns=param_columns)


def create_plots(df: pd.DataFrame, param_columns: List[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.histplot(df["K"], kde=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Corridor threshold")
    plt.title("Distribution of K-index")
    plt.xlabel("K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "k_distribution.png")
    plt.close()

    if param_columns:
        scatter_cols = param_columns[:2]
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df,
            x=scatter_cols[0],
            y=scatter_cols[1],
            hue=df["in_corridor"],
            palette={True: "green", False: "gray"},
        )
        plt.title("Parameter scatter (first two dimensions)")
        plt.tight_layout()
        plt.savefig(outdir / "parameter_scatter.png")
        plt.close()


def main() -> None:
    args = parse_args()
    df = load_passports(args.logdir)
    param_columns = [col for col in df.columns if col.startswith("param_")]
    stripped_cols = [col.replace("param_", "") for col in param_columns]
    df.rename(columns=dict(zip(param_columns, stripped_cols)), inplace=True)
    summary = compute_summary(df, stripped_cols, threshold=1.0)

    extras: Dict[str, object] = {}
    if args.baseline:
        extras["baseline_comparison"] = compare_to_baseline(df, args.baseline, stripped_cols)

    save_summary(summary, args.output, extras if extras else None)

    if args.export_corridor:
        hits = df[df["in_corridor"]]
        args.export_corridor.parent.mkdir(parents=True, exist_ok=True)
        hits.to_csv(args.export_corridor, index=False)
    plots_dir = args.output.parent / "fre_phase1_plots"
    create_plots(df, stripped_cols, plots_dir)
    print(f"[FRE] Aggregated {summary.total_runs} runs. Summary -> {args.output}")


if __name__ == "__main__":
    main()
