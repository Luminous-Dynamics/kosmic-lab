from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest

from fre import track_g_runner
from fre.track_h_runner import (  # type: ignore[attr-defined]
    _apply_track_h_warm_start_overrides,
    _validate_track_h_config,
)


def test_to_serializable_flattens_numpy_types() -> None:
    sample = {
        "foo": np.array([1.0, 2.0]),
        "bar": np.bool_(True),
        "nested": {"vals": (np.float32(3.5), np.int64(7))},
    }

    converted = track_g_runner._to_serializable(sample)  # type: ignore[attr-defined]
    encoded = json.dumps(converted)

    assert '"bar": true' in encoded
    assert '"vals": [3.5, 7]' in encoded


def test_simple_agent_checkpoint_roundtrip() -> None:
    agent = track_g_runner.SimpleAgent(obs_dim=4, action_dim=2, learning_rate=0.01)
    agent.W[:] = np.array([[0.1, 0.2, 0.3, 0.4], [-0.4, -0.3, -0.2, -0.1]])
    agent.b[:] = np.array([0.5, -0.5])

    state = agent.state_dict()

    restored = track_g_runner.SimpleAgent(obs_dim=4, action_dim=2, learning_rate=0.5)
    restored.load_state_dict(state)

    np.testing.assert_allclose(restored.W, agent.W)
    np.testing.assert_allclose(restored.b, agent.b)
    assert restored.lr == agent.lr


def test_apply_warm_start_overrides_updates_config() -> None:
    config = {
        "phase_g2": {
            "warm_start": {
                "load_path": None,
                "save_path": "old_save.json",
            }
        }
    }
    track_g_runner._apply_warm_start_overrides(config, "g2", "new_load.json", "new_save.json")  # type: ignore[attr-defined]
    warm_cfg = config["phase_g2"]["warm_start"]
    assert warm_cfg["load_path"] == "new_load.json"
    assert warm_cfg["save_path"] == "new_save.json"


def test_track_h_warm_start_override() -> None:
    config = {
        "track_h": {
            "warm_start": {"load_path": "orig.json", "save_path": None},
        }
    }
    _apply_track_h_warm_start_overrides(config, "override.json", "new.json")
    warm_cfg = config["track_h"]["warm_start"]
    assert warm_cfg["load_path"] == "override.json"
    assert warm_cfg["save_path"] == "new.json"


def test_validate_phase_config_checks_files(tmp_path) -> None:
    temp = tmp_path / "checkpoint.json"
    temp.write_text("{}", encoding="utf-8")
    cfg = {
        "phase_g2": {
            "warm_start": {"load_path": str(temp), "save_path": str(tmp_path / "out.json")},
        }
    }
    track_g_runner._validate_phase_config(cfg, "g2")  # type: ignore[attr-defined]


def test_validate_phase_config_missing_file_raises(tmp_path) -> None:
    cfg = {
        "phase_g2": {
            "warm_start": {"load_path": str(tmp_path / "missing.json")},
        }
    }
    try:
        track_g_runner._validate_phase_config(cfg, "g2")  # type: ignore[attr-defined]
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing checkpoint")


def test_validate_track_h_config_checks_paths(tmp_path) -> None:
    ckpt = tmp_path / "h_load.json"
    ckpt.write_text("{}", encoding="utf-8")
    cfg = {
        "track_h": {
            "warm_start": {"load_path": str(ckpt), "save_path": None},
            "memory_architectures": [
                {"type": "lstm", "warm_start": {"save_path": str(tmp_path / "lstm.json")}}
            ],
        }
    }
    _validate_track_h_config(cfg)


def test_validate_track_h_config_missing_file(tmp_path) -> None:
    cfg = {
        "track_h": {
            "warm_start": {"load_path": str(tmp_path / "missing.json")},
            "memory_architectures": [],
        }
    }
    try:
        _validate_track_h_config(cfg)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing Track H checkpoint")


def test_episode_logger_writes_json(tmp_path) -> None:
    log_path = tmp_path / "episodes.jsonl"
    logger = track_g_runner.EpisodeLogger(log_path, "g2")  # type: ignore[attr-defined]
    logger.log({"episode_index": 1, "k_index": 0.5})
    logger.close()
    content = log_path.read_text(encoding="utf-8").strip()
    assert '"k_index": 0.5' in content


def test_checkpoint_metadata_includes_config_hash(tmp_path, monkeypatch) -> None:
    agent = track_g_runner.SimpleAgent(obs_dim=2, action_dim=1)
    warm_cfg = {"save_path": str(tmp_path / "ckpt.json")}
    old_config = track_g_runner._CURRENT_CONFIG_METADATA  # type: ignore[attr-defined]
    old_commit = track_g_runner._CURRENT_GIT_COMMIT  # type: ignore[attr-defined]
    track_g_runner._CURRENT_CONFIG_METADATA = {"path": "cfg.yaml", "sha256": "abc123"}  # type: ignore[attr-defined]
    track_g_runner._CURRENT_GIT_COMMIT = "deadbeef"  # type: ignore[attr-defined]
    try:
        track_g_runner._maybe_save_agent(agent, warm_cfg, "G2", extra_metadata={"episodes_completed": 1})  # type: ignore[attr-defined]
    finally:
        track_g_runner._CURRENT_CONFIG_METADATA = old_config  # type: ignore[attr-defined]
        track_g_runner._CURRENT_GIT_COMMIT = old_commit  # type: ignore[attr-defined]
    data = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    meta = data["metadata"]
    assert meta["config"]["path"] == "cfg.yaml"
    assert meta["config"]["sha256"] == "abc123"
    assert meta["git_commit"] == "deadbeef"


def _write_checkpoint(tmp_path, config_hash: str) -> Path:
    agent = track_g_runner.SimpleAgent(obs_dim=2, action_dim=1)
    path = tmp_path / "ckpt.json"
    payload = {
        "state": agent.state_dict(),
        "metadata": {
            "config": {"sha256": config_hash, "path": "old.yaml"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_hash_mismatch_raises(tmp_path) -> None:
    ckpt = _write_checkpoint(tmp_path, "abc123")
    warm_cfg = {"load_path": str(ckpt)}
    agent = track_g_runner.SimpleAgent(obs_dim=2, action_dim=1)
    old_meta = track_g_runner._CURRENT_CONFIG_METADATA  # type: ignore[attr-defined]
    track_g_runner._CURRENT_CONFIG_METADATA = {"sha256": "zzz999"}  # type: ignore[attr-defined]
    try:
        with pytest.raises(RuntimeError):
            track_g_runner._maybe_load_agent(agent, warm_cfg, "G2")  # type: ignore[attr-defined]
    finally:
        track_g_runner._CURRENT_CONFIG_METADATA = old_meta  # type: ignore[attr-defined]


def test_config_hash_override_allows(tmp_path) -> None:
    ckpt = _write_checkpoint(tmp_path, "abc123")
    warm_cfg = {"load_path": str(ckpt), "allow_mismatch": True}
    agent = track_g_runner.SimpleAgent(obs_dim=2, action_dim=1)
    old_meta = track_g_runner._CURRENT_CONFIG_METADATA  # type: ignore[attr-defined]
    track_g_runner._CURRENT_CONFIG_METADATA = {"sha256": "zzz999"}  # type: ignore[attr-defined]
    try:
        track_g_runner._maybe_load_agent(agent, warm_cfg, "G2")  # type: ignore[attr-defined]
    finally:
        track_g_runner._CURRENT_CONFIG_METADATA = old_meta  # type: ignore[attr-defined]


def test_checkpoint_contains_config_snapshot(tmp_path) -> None:
    agent = track_g_runner.SimpleAgent(obs_dim=2, action_dim=1)
    warm_cfg = {"save_path": str(tmp_path / "ckpt.json")}
    old_meta = track_g_runner._CURRENT_CONFIG_METADATA  # type: ignore[attr-defined]
    old_text = track_g_runner._CURRENT_CONFIG_TEXT  # type: ignore[attr-defined]
    track_g_runner._CURRENT_CONFIG_METADATA = {"sha256": "abc", "path": "cfg.yaml"}  # type: ignore[attr-defined]
    track_g_runner._CURRENT_CONFIG_TEXT = "alpha: 1\n"  # type: ignore[attr-defined]
    try:
        track_g_runner._maybe_save_agent(agent, warm_cfg, "G2")  # type: ignore[attr-defined]
    finally:
        track_g_runner._CURRENT_CONFIG_METADATA = old_meta  # type: ignore[attr-defined]
        track_g_runner._CURRENT_CONFIG_TEXT = old_text  # type: ignore[attr-defined]
    data = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    assert data["metadata"]["config_snapshot"] == "alpha: 1\n"
