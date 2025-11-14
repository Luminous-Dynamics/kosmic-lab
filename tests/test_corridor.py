from __future__ import annotations

from pathlib import Path

import pandas as pd

from fre.corridor import compare_to_baseline, compute_corridor_metrics, save_summary


def test_compare_to_baseline(tmp_path: Path) -> None:
    current = pd.DataFrame(
        {
            "K": [1.2, 1.1],
            "TAT": [0.6, 0.55],
            "Recovery": [0.9, 1.0],
            "param_energy": [0.5, 0.55],
            "param_comm": [0.3, 0.32],
            "in_corridor": [True, True],
        }
    )
    baseline = current.copy()
    baseline_path = tmp_path / "baseline.csv"
    baseline.to_csv(baseline_path, index=False)

    comparison = compare_to_baseline(current, baseline_path, ["param_energy", "param_comm"], bins=4)
    assert comparison["jaccard"] == 1.0
    assert comparison["centroid_shift"] == 0.0


def test_save_summary_extra(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "K": [0.9, 1.1],
            "TAT": [0.4, 0.6],
            "Recovery": [1.1, 0.9],
            "param_energy": [0.4, 0.6],
            "in_corridor": [False, True],
        }
    )
    summary = compute_corridor_metrics(df, threshold=1.0, param_columns=["param_energy"])
    out = tmp_path / "summary.json"
    save_summary(summary, out, extra={"note": "test"})
    text = out.read_text()
    assert '"note": "test"' in text
