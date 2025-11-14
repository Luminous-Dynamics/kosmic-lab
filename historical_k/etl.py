"""
Historical data ingestion utilities.

Provides lightweight ETL helpers that look for proxy CSV files under
historical_k/data/. When files are absent, deterministic fallback series are
generated so the pipeline remains reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_feature_series(feature: str, years: Iterable[int], data_dir: Path = DATA_DIR) -> pd.Series:
    """
    Load a single proxy series.

    Expected CSV format: columns ["year", "value"].
    Missing files trigger a deterministic zero baseline.
    """
    years = list(years)
    index = pd.Index(years, name="year")
    path = data_dir / f"{feature}.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "year" not in df.columns or "value" not in df.columns:
            raise ValueError(f"{path} missing required columns 'year' and 'value'.")
        series = df.set_index("year")["value"].astype(float)
        series = series.reindex(index).interpolate().ffill().bfill()
    else:
        # Deterministic fallback (zeros)
        series = pd.Series([0.0 for _ in years], index=index, name=feature)
    return series


def build_harmony_frame(proxies: Dict[str, List[str]], years: Iterable[int]) -> pd.DataFrame:
    """
    Construct a DataFrame with one column per harmony derived from feature averages.
    """
    years = list(years)
    index = pd.Index(years, name="year")
    frame = pd.DataFrame(index=index)

    for harmony, features in proxies.items():
        if not features:
            raise ValueError(f"Harmony '{harmony}' has no associated features.")
        series_list = [load_feature_series(feature, years) for feature in features]
        stacked = pd.concat(series_list, axis=1)
        harmony_series = stacked.mean(axis=1)
        frame[harmony] = harmony_series

    return frame


def compute_k_series(harmony_frame: pd.DataFrame) -> pd.Series:
    """
    Compute K-index as an equal-weighted average of harmony columns.
    """
    if harmony_frame.empty:
        raise ValueError("Harmony frame is empty.")
    return harmony_frame.mean(axis=1)
