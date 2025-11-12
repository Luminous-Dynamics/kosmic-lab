"""Integration tests for Holochain bridge."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.holochain_bridge import HolochainBridge, HolochainConfig


@pytest.fixture
def mock_bridge():
    """Create bridge in mock mode (no conductor required)."""
    config = HolochainConfig(conductor_url="mock://localhost")
    return HolochainBridge(config)


@pytest.fixture
def sample_passport(tmp_path: Path) -> Path:
    """Create sample K-passport for testing."""
    passport = {
        "run_id": "test-holochain-001",
        "commit": "abc123def456",
        "config_hash": "789ghi012jkl",
        "seed": 42,
        "experiment": "holochain_integration_test",
        "params": {
            "energy_gradient": 0.5,
            "communication_cost": 0.3
        },
        "estimators": {
            "phi": "empirical",
            "te": {
                "estimator": "kraskov",
                "k": 3,
                "lag": 1
            }
        },
        "metrics": {
            "K": 1.23,
            "TAT": 0.615,
            "Recovery": 0.99,
            "phi": 2.5,
            "te_mutual": 1.2,
            "TE_symmetry": 0.85
        },
        "timestamp": "2025-11-09T12:34:56Z"
    }

    passport_file = tmp_path / "test_passport.json"
    with passport_file.open('w') as f:
        json.dump(passport, f, indent=2)

    return passport_file


class TestHolochainBridge:
    """Test suite for Holochain integration."""

    def test_publish_passport(self, mock_bridge, sample_passport):
        """Test K-passport publishing to DHT."""
        header_hash = mock_bridge.publish_passport(sample_passport)

        assert header_hash is not None
        assert "test-holochain-001" in header_hash  # Mock hash contains run_id

    def test_publish_directory(self, mock_bridge, tmp_path):
        """Test batch publishing of K-passports."""
        # Create multiple test passports
        for i in range(5):
            passport = {
                "run_id": f"batch-test-{i:03d}",
                "commit": "abc123",
                "config_hash": "def456",
                "seed": i,
                "experiment": "batch_test",
                "params": {},
                "estimators": {"phi": "star", "te": {"estimator": "kraskov", "k": 3, "lag": 1}},
                "metrics": {"K": 1.0 + i * 0.1, "TAT": 0.5, "Recovery": 1.0},
                "timestamp": "2025-11-09T12:00:00Z"
            }

            passport_file = tmp_path / f"passport_{i}.json"
            with passport_file.open('w') as f:
                json.dump(passport, f)

        # Publish directory
        published = mock_bridge.publish_directory(tmp_path)

        assert len(published) == 5
        assert all("passport_" in name for name in published.keys())

    def test_passport_transform(self, mock_bridge):
        """Test K-passport transformation to Holochain format."""
        kosmic_passport = {
            "run_id": "transform-test",
            "commit": "xyz789",
            "config_hash": "abc123",
            "seed": 99,
            "experiment": "test_experiment",
            "params": {"energy_gradient": 0.7},
            "estimators": {"phi": "star", "te": {"estimator": "kraskov", "k": 3, "lag": 1}},
            "metrics": {
                "K": 1.5,
                "TAT": 0.75,
                "Recovery": 0.8
            },
            "timestamp": "2025-11-09T14:00:00Z"
        }

        holochain_passport = mock_bridge._transform_passport(kosmic_passport)

        # Verify structure
        assert holochain_passport['run_id'] == "transform-test"
        assert holochain_passport['metrics']['k_index'] == 1.5
        assert holochain_passport['metrics']['tat'] == 0.75
        assert 'researcher_agent' in holochain_passport

    def test_missing_required_fields(self, mock_bridge, tmp_path):
        """Test validation of incomplete K-passports."""
        incomplete_passport = {
            "run_id": "incomplete-test",
            # Missing: commit, config_hash, seed, experiment, metrics
        }

        passport_file = tmp_path / "incomplete.json"
        with passport_file.open('w') as f:
            json.dump(incomplete_passport, f)

        with pytest.raises(ValueError, match="Missing required fields"):
            mock_bridge.publish_passport(passport_file)


@pytest.mark.skipif(
    not Path("/usr/bin/hc").exists(),
    reason="Holochain not installed"
)
class TestHolochainLiveIntegration:
    """Live integration tests (requires running Holochain conductor)."""

    @pytest.fixture
    def live_bridge(self):
        """Create bridge connected to live conductor."""
        return HolochainBridge()

    def test_live_publish_and_query(self, live_bridge, sample_passport):
        """
        Test publishing to live Holochain and querying corridor.

        Requires:
        - Holochain conductor running: hc sandbox run
        - passport_zome installed
        """
        # Publish
        header_hash = live_bridge.publish_passport(sample_passport)
        assert header_hash is not None
        assert len(header_hash) > 10  # Real hash is longer

        # Query corridor
        passports = live_bridge.query_corridor(min_k=1.0, max_k=1.5)

        # Should include our published passport
        assert any(
            p['run_id'] == 'test-holochain-001'
            for p in passports
        )

    def test_live_verify_passport(self, live_bridge):
        """Test passport integrity verification."""
        # Assume passport already published
        header_hash = "QmXXXXXXXXXXXXXXXX"  # Replace with actual hash

        verification = live_bridge.verify_passport(header_hash)

        assert verification.valid is True
        assert verification.commit_verified is True
        assert verification.config_verified is True
