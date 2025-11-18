"""
Historical data ingestion utilities.

Provides lightweight ETL helpers that look for proxy CSV files under
historical_k/data/. When files are absent, deterministic fallback series are
generated so the pipeline remains reproducible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

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

def _century_label(year: int) -> int:
    """Map a year to its century anchor (e.g., 1800..1899 -> 1800)."""
    return (int(year) // 100) * 100


def _zscore_grouped(series: pd.Series, groups: pd.Series) -> pd.Series:
    def _z(g: pd.Series) -> pd.Series:
        mu = g.mean()
        sigma = g.std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            sigma = 1.0
        return (g - mu) / sigma

    return series.groupby(groups).transform(_z)


def _minmax_grouped(series: pd.Series, groups: pd.Series) -> pd.Series:
    def _mm(g: pd.Series) -> pd.Series:
        lo = g.min()
        hi = g.max()
        if hi == lo:
            return g * 0.0
        return (g - lo) / (hi - lo)

    return series.groupby(groups).transform(_mm)


def normalize_series(series: pd.Series, years: Iterable[int], strategy: str = "none") -> pd.Series:
    """Normalize a feature series according to the requested strategy.

    Supported strategies:
      - none: no normalization
      - zscore_by_century: z-score within each century bucket
      - minmax_by_century: min-max scale within each century bucket
      - zscore_global: z-score across entire series
      - minmax_global: min-max across entire series
    """
    strategy = (strategy or "none").lower()
    years = list(years)
    s = series.copy()
    s.index = pd.Index(years, name="year")

    if strategy == "none":
        return s

    if strategy == "zscore_global":
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            sigma = 1.0
        return (s - mu) / sigma

    if strategy == "minmax_global":
        lo, hi = s.min(), s.max()
        if hi == lo:
            return s * 0.0
        return (s - lo) / (hi - lo)

    # Grouped strategies
    if strategy in {"zscore_by_century", "minmax_by_century"}:
        groups = pd.Series([_century_label(y) for y in years], index=s.index)
        if strategy == "zscore_by_century":
            return _zscore_grouped(s, groups)
        else:
            return _minmax_grouped(s, groups)

    # Rolling strategies (window = 3 decades)
    if strategy in {"zscore_rolling_3", "minmax_rolling_3"}:
        window = 3
        if strategy == "zscore_rolling_3":
            mu = s.rolling(window=window, center=True, min_periods=1).mean()
            sigma = s.rolling(window=window, center=True, min_periods=1).std(ddof=0)
            sigma = sigma.replace(0, 1.0).fillna(1.0)
            return (s - mu) / sigma
        else:
            lo = s.rolling(window=window, center=True, min_periods=1).min()
            hi = s.rolling(window=window, center=True, min_periods=1).max()
            scale = (hi - lo).replace(0, 1.0).fillna(1.0)
            return (s - lo) / scale

    # Unknown -> no-op
    return s


def build_harmony_frame(
    proxies: Dict[str, List[str]],
    years: Iterable[int],
    normalization: str = "none",
    normalization_overrides: Optional[Dict[str, str]] = None,
    feature_aggregation: str = "mean",
    feature_aggregation_overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Construct a DataFrame with one column per harmony derived from normalized
    feature averages according to the selected strategy.
    """
    years = list(years)
    index = pd.Index(years, name="year")
    frame = pd.DataFrame(index=index)

    overrides = normalization_overrides or {}
    agg_over = feature_aggregation_overrides or {}
    for harmony, features in proxies.items():
        if not features:
            raise ValueError(f"Harmony '{harmony}' has no associated features.")
        strat = overrides.get(harmony, normalization)
        series_list = [normalize_series(load_feature_series(feature, years), years, strat) for feature in features]
        stacked = pd.concat(series_list, axis=1)
        agg = (agg_over.get(harmony, feature_aggregation) or "mean").lower()
        if agg == "median":
            harmony_series = stacked.median(axis=1)
        else:
            harmony_series = stacked.mean(axis=1)
        frame[harmony] = harmony_series

    return frame


def compute_k_series(harmony_frame: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Compute K-index from harmony frame.

    - If weights provided: use weighted average across harmony columns.
    - Else: simple mean across harmonies.
    """
    if harmony_frame.empty:
        raise ValueError("Harmony frame is empty.")
    if not weights:
        return harmony_frame.mean(axis=1)
    # Align and normalize weights
    cols = list(harmony_frame.columns)
    w = {k: float(v) for k, v in (weights or {}).items() if k in cols}
    if not w:
        return harmony_frame.mean(axis=1)
    w_sum = sum(w.values()) or 1.0
    w_norm = {k: v / w_sum for k, v in w.items()}
    # Build matrix multiply
    # Ensure all columns present; missing weights -> 0
    weight_vec = [w_norm.get(c, 0.0) for c in cols]
    arr = harmony_frame.to_numpy()
    import numpy as _np  # local import to avoid hard dep if unused
    return pd.Series(arr @ _np.array(weight_vec), index=harmony_frame.index, name="K")
