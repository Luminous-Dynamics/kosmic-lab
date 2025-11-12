from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fre.track_b_runner import TrackBRunner


@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_track_b_runner_generates_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "track_b.yaml"
    summary_path = tmp_path / "summary.json"
    diagnostics_path = tmp_path / "diagnostics.csv"
    training_path = tmp_path / "training.csv"

    config_payload = {
        "experiment": {
            "name": "track_b_test",
            "description": "Smoke test for Track B runner",
            "base_config": "fre/configs/k_config.yaml",
            "open_loop_episodes": 1,
        },
        "controllers": {
            "sac": {
                "action_interval": 2,
                "horizon": 12,
                "gamma": 0.95,
                "tau": 0.01,
                "batch_size": 4,
                "warmup_steps": 0,
                "train_episodes": 1,
                "eval_episodes": 1,
                "action_scale": 0.05,
                "hidden_layers": [32, 32],
                "buffer_capacity": 512,
                "reward_beta": 0.05,
                "actor_lr": 3e-4,
                "critic_lr": 3e-4,
                "alpha_lr": 3e-4,
                "init_temperature": 0.1,
            }
        },
        "parameters": {
            "energy_gradient": [0.5],
            "communication_cost": [0.3],
            "plasticity_rate": [0.15],
        },
        "output": {
            "summary_json": str(summary_path),
            "diagnostics_csv": str(diagnostics_path),
            "training_csv": str(training_path),
        },
    }

    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    runner = TrackBRunner(config_path)
    summary = runner.run()
    runner.write_outputs(summary)

    assert summary_path.exists()
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_data["episodes"], "Expected episode summaries in output"
    modes = {episode["mode"] for episode in summary_data["episodes"]}
    # Expect open-loop, train, and eval episodes present
    assert any(mode.startswith("open_loop") for mode in modes)
    assert any(mode.startswith("controller_train") for mode in modes)
    assert any(mode.startswith("controller_eval") for mode in modes)

    assert diagnostics_path.exists()
    diagnostics_rows = diagnostics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(diagnostics_rows) > 1  # header + data

    assert training_path.exists()
    training_rows = training_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(training_rows) > 1, "training metrics should contain at least one update"

    # Controller should have collected data in replay buffer
    assert len(runner.controller.replay_buffer) > 0
