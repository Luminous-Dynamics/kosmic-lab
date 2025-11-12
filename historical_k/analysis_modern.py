"""Quick analysis for modern-era historical K(t) hypotheses."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path("logs/historical_k/k_t_series.csv")


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df[df["year"] >= 1890].reset_index(drop=True)


def main() -> None:
    df = load()
    corr_rec_k = df["reciprocity"].corr(df["K"])
    corr_play_k = df["play_entropy"].corr(df["K"])
    corr_wis_k = df["wisdom_accuracy"].corr(df["K"])

    pre = df[df["year"] < 1940]["K"].mean()
    post = df[df["year"] >= 1940]["K"].mean()

    crises = df[df["year"].isin([1910, 1930, 1940])][["year", "K"]]

    print("Reciprocity ↔ K correlation:", round(corr_rec_k, 3))
    print("Play entropy ↔ K correlation:", round(corr_play_k, 3))
    print("Wisdom accuracy ↔ K correlation:", round(corr_wis_k, 3))
    print(f"Pre-1940 mean K: {pre:.3f} | Post-1940 mean K: {post:.3f}")
    print("Crises (K values):")
    print(crises.to_string(index=False))


if __name__ == "__main__":
    main()
