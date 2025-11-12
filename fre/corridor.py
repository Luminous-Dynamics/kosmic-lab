from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CorridorSummary:
    total_runs: int
    corridor_hits: int
    corridor_rate: float
    mean_K: float
    std_K: float
    mean_TAT: float
    mean_Recovery: float
    centroid: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_runs": self.total_runs,
            "corridor_hits": self.corridor_hits,
            "corridor_rate": self.corridor_rate,
            "mean_K": self.mean_K,
            "std_K": self.std_K,
            "mean_TAT": self.mean_TAT,
            "mean_Recovery": self.mean_Recovery,
            "centroid": self.centroid,
            "bounds": {k: list(v) for k, v in self.bounds.items()},
        }


def compute_corridor_metrics(df: pd.DataFrame, threshold: float, param_columns: List[str]) -> CorridorSummary:
    total = len(df)
    corridor_df = df[df["in_corridor"]]
    hits = len(corridor_df)
    rate = hits / total if total else 0.0
    centroid = {col: float(corridor_df[col].mean()) if hits else float(df[col].mean()) for col in param_columns}
    bounds = {col: (float(df[col].min()), float(df[col].max())) for col in param_columns}
    return CorridorSummary(
        total_runs=total,
        corridor_hits=hits,
        corridor_rate=rate,
        mean_K=float(df["K"].mean()),
        std_K=float(df["K"].std(ddof=0)),
        mean_TAT=float(df["TAT"].mean()),
        mean_Recovery=float(df["Recovery"].mean()),
        centroid=centroid,
        bounds=bounds,
    )


def discretize_corridor(df: pd.DataFrame, param_columns: List[str], bins: int = 12) -> List[Tuple[int, ...]]:
    if df.empty:
        return []
    grids = []
    for col in param_columns:
        values = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
        grids.append(values.to_numpy())
    stacked = np.vstack(grids).T
    return [tuple(row) for row in stacked]


def compare_to_baseline(
    current_df: pd.DataFrame,
    baseline_path: Path,
    param_columns: List[str],
    bins: int = 12,
) -> Dict[str, float]:
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    baseline_df = pd.read_csv(baseline_path)
    if "in_corridor" not in baseline_df.columns:
        baseline_df["in_corridor"] = True
    else:
        if baseline_df["in_corridor"].dtype != bool:
            baseline_df["in_corridor"] = baseline_df["in_corridor"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    current_hits = current_df[current_df["in_corridor"]]
    baseline_hits = baseline_df[baseline_df["in_corridor"]]
    if current_hits.empty or baseline_hits.empty:
        return {
            "baseline_path": str(baseline_path),
            "baseline_hits": int(len(baseline_hits)),
            "jaccard": 0.0,
            "centroid_shift": float("nan"),
        }

    available_cols = [col for col in param_columns if col in current_hits.columns and col in baseline_hits.columns]
    if not available_cols:
        return {
            "baseline_path": str(baseline_path),
            "baseline_hits": int(len(baseline_hits)),
            "jaccard": 0.0,
            "centroid_shift": float("nan"),
        }

    combined = pd.concat([current_hits[available_cols], baseline_hits[available_cols]])
    bin_edges = {
        col: np.linspace(combined[col].min(), combined[col].max(), bins + 1)
        if combined[col].min() != combined[col].max()
        else np.array([combined[col].min(), combined[col].min() + 1e-6])
        for col in available_cols
    }

    def bin_set(df: pd.DataFrame) -> set[Tuple[int, ...]]:
        tuples: List[Tuple[int, ...]] = []
        for _, row in df[available_cols].iterrows():
            idxs = []
            for col in available_cols:
                edges = bin_edges[col]
                idx = np.digitize(row[col], edges[1:-1], right=False)
                idxs.append(int(idx))
            tuples.append(tuple(idxs))
        return set(tuples)

    set_current = bin_set(current_hits)
    set_baseline = bin_set(baseline_hits)
    union = set_current | set_baseline
    intersection = set_current & set_baseline
    jaccard = len(intersection) / len(union) if union else 0.0

    current_centroid = np.array([current_hits[col].mean() for col in available_cols])
    baseline_centroid = np.array([baseline_hits[col].mean() for col in available_cols])
    centroid_shift = float(np.linalg.norm(current_centroid - baseline_centroid))

    return {
        "baseline_path": str(baseline_path),
        "baseline_hits": int(len(baseline_hits)),
        "jaccard": jaccard,
        "centroid_shift": centroid_shift,
    }


def save_summary(summary: CorridorSummary, path: Path, extra: Dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = summary.to_dict()
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
