"""Comprehensive Integration Tests for FRE Pipeline"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from core.config import load_yaml_config
from core.kpass import KPassportWriter
from fre.universe import UniverseSimulator


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create temporary configuration file."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("universes: 3\\nsamples: 10\\nseed: 42\\n")
    return config_path


class TestFREIntegration:
    """Integration tests for FRE pipeline."""

    def test_universe_deterministic(self):
        """Universe simulator is deterministic with same seed."""
        sim = UniverseSimulator()
        params = {"energy_gradient": 0.5}

        result1 = sim.run(params, seed=42)
        result2 = sim.run(params, seed=42)

        assert result1['K'] == result2['K']

    def test_k_index_bounds(self):
        """K-index always within [0, 2.5]."""
        sim = UniverseSimulator()
        rng = np.random.default_rng(42)

        for _ in range(100):
            params = {"energy_gradient": rng.uniform(0, 1)}
            result = sim.run(params, seed=rng.integers(0, 10000))
            assert 0 <= result['K'] <= 2.5
