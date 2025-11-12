from __future__ import annotations

import pandas as pd
import pytest

from historical_k.compute_k import _bootstrap_mean_ci, _years_from_config, validate_config
from historical_k.etl import build_harmony_frame, compute_k_series


def test_validate_config_ok() -> None:
    payload = {
        "windows": {"size": "decade"},
        "proxies": {"resonant_coherence": ["network_modularity_inverse"]},
    }
    validate_config(payload)


def test_validate_config_missing() -> None:
    with pytest.raises(ValueError):
        validate_config({"proxies": {}})


def test_years_from_config_uses_events() -> None:
    payload = {
        "windows": {"size": "decade"},
        "preregistered_events": {"peaks": [1890, 1995, 2010]},
    }
    years = _years_from_config(payload)
    assert years[0] == 1890
    assert years[-1] == 2010


def test_harmony_frame_fallback(tmp_path, monkeypatch) -> None:
    years = [1800, 1810]
    proxies = {"H1": ["nonexistent_feature"]}
    frame = build_harmony_frame(proxies, years)
    assert list(frame.index) == years
    assert frame.loc[1800, "H1"] == 0.0
    series = compute_k_series(frame)
    assert isinstance(series, pd.Series)


def test_bootstrap_mean_ci() -> None:
    series = pd.Series([1.0, 1.2, 0.9, 1.1])
    low, high = _bootstrap_mean_ci(series, {"bootstrap_samples": 500, "ci": 0.9, "seed": 1})
    assert low <= high
