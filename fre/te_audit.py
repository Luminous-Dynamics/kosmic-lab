"""Audit utility for comparing corridor sets against a reference."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_corridor(path: Path) -> set[tuple[float, ...]]:
    df = pd.read_csv(path)
    param_cols = [c for c in df.columns if c not in {"run_id", "seed", "K", "TAT", "Recovery"}]
    tuples = [tuple(df[col].iloc[i] for col in param_cols) for i in range(len(df))]
    return set(tuples)


def jaccard(a: set[tuple], b: set[tuple]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare corridor exports")
    parser.add_argument("reference", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    ref = load_corridor(args.reference)
    tgt = load_corridor(args.target)
    score = jaccard(ref, tgt)
    print(f"Reference size: {len(ref)} | Target size: {len(tgt)}")
    print(f"Jaccard similarity: {score:.3f}")


if __name__ == "__main__":
    main()
