"""Property-Based Tests using Hypothesis"""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, strategies as st, settings
    from hypothesis import assume

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators if hypothesis not available
    given = lambda *args, **kwargs: lambda f: f
    settings = lambda *args, **kwargs: lambda f: f

    class DummySt:
        @staticmethod
        def floats(*args, **kwargs):
            return None

        @staticmethod
        def integers(*args, **kwargs):
            return None

    st = DummySt()

from fre.universe import UniverseSimulator


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestUniverseProperties:
    """Property-based tests for universe simulator."""

    @given(energy=st.floats(0, 1), comm_cost=st.floats(0, 1), seed=st.integers(0, 100000))
    @settings(max_examples=200)
    def test_k_always_bounded(self, energy, comm_cost, seed):
        """K-index always in valid range for any parameters."""
        sim = UniverseSimulator()
        params = {"energy_gradient": energy, "communication_cost": comm_cost}

        result = sim.run(params, seed)
        assert 0 <= result["K"] <= 2.5

    @given(seed=st.integers(0, 10000))
    @settings(max_examples=100)
    def test_determinism(self, seed):
        """Same params and seed always produce same result."""
        sim = UniverseSimulator()
        params = {"energy_gradient": 0.5}

        result1 = sim.run(params, seed)
        result2 = sim.run(params, seed)

        assert result1["K"] == result2["K"]
