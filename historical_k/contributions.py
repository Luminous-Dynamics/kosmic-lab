from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("logs/historical_k/k_t_series.csv")


def rolling_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, fill_value=np.nan, dtype=float)
    n = len(x)
    half = w // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        xs = x[a:b]
        ys = y[a:b]
        if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
            r = np.corrcoef(xs, ys)[0, 1]
        else:
            r = np.nan
        out[i] = r
    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    years = df["year"].to_numpy()
    k = df["K"].to_numpy()

    harmony_cols = [
        "resonant_coherence",
        "interconnection",
        "reciprocity",
        "play_entropy",
        "wisdom_accuracy",
        "flourishing",
    ]

    w = 5  # window of decades
    out = {"year": years}
    for col in harmony_cols:
        if col in df.columns:
            out[col] = rolling_corr(df[col].to_numpy(), k, w)

    out_df = pd.DataFrame(out)
    outdir = Path("logs/historical_k")
    outdir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outdir / "contrib_rolling.csv", index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    for col in harmony_cols:
        if col in out_df.columns:
            plt.plot(years, out_df[col], label=col)
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    plt.xlabel("Year")
    plt.ylabel(f"Rolling r (window={w} decades)")
    plt.title("Harmony ↔ K Rolling Correlations")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "contrib_rolling.png")
    plt.close()


if __name__ == "__main__":
    main()

