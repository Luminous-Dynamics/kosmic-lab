from __future__ import annotations

import pandas as pd

from historical_k.etl import normalize_series, build_harmony_frame, compute_k_series


def test_normalize_zscore_by_century_means_zero():
    years = [1800, 1810, 1820, 1830, 1840, 1900, 1910, 1920]
    # Two centuries with simple ramps
    vals = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0])
    s = normalize_series(vals, years, "zscore_by_century")
    # Means per century approximately zero
    s.index = pd.Index(years, name="year")
    mean_1800s = float(s.loc[[1800, 1810, 1820, 1830, 1840]].mean())
    mean_1900s = float(s.loc[[1900, 1910, 1920]].mean())
    assert abs(mean_1800s) < 1e-9
    assert abs(mean_1900s) < 1e-9


def test_normalize_minmax_by_century_bounds():
    years = [1800, 1810, 1820, 1830]
    vals = pd.Series([2.0, 4.0, 6.0, 8.0])
    s = normalize_series(vals, years, "minmax_by_century")
    assert float(s.min()) == 0.0
    assert float(s.max()) == 1.0


def test_normalize_rolling_strategies_no_nans():
    years = [1800, 1810, 1820, 1830, 1840]
    vals = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    z = normalize_series(vals, years, "zscore_rolling_3")
    m = normalize_series(vals, years, "minmax_rolling_3")
    assert len(z) == len(vals)
    assert len(m) == len(vals)
    assert not z.isna().any()
    assert not m.isna().any()


def test_build_harmony_frame_overrides():
    years = [1800, 1810, 1820]
    proxies = {
        "H1": ["network_modularity_inverse"],
        "H2": ["trade_network_degree"],
    }
    # Use different strategies per harmony
    frame = build_harmony_frame(
        proxies, years, normalization="zscore_global", normalization_overrides={"H2": "minmax_global"}
    )
    assert set(frame.columns) == {"H1", "H2"}
    assert len(frame) == len(years)


def test_feature_aggregation_and_weights():
    years = [1800, 1810, 1820]
    # Two features for H1 where one is an outlier; median should reduce its effect
    proxies = {
        "H1": ["network_modularity_inverse", "trade_network_degree"],
        "H2": ["trade_network_degree"],
    }
    # Build with median aggregation for H1
    frame = build_harmony_frame(
        proxies,
        years,
        normalization="minmax_global",
        normalization_overrides={},
        feature_aggregation="mean",
        feature_aggregation_overrides={"H1": "median"},
    )
    # Weighted K: weight H1 twice as much as H2
    k_w = compute_k_series(frame, weights={"H1": 2.0, "H2": 1.0})
    k_u = compute_k_series(frame, weights=None)
    assert (k_w != k_u).any()
