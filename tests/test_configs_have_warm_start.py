from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_EXPECTATIONS = [
    ("fre/configs/track_g_phase_g2.yaml", ("phase_g2", "warm_start")),
    ("fre/configs/track_g_phase_g3.yaml", ("phase_g3", "warm_start")),
    ("fre/configs/track_g2plus.yaml", ("phase_g2plus", "warm_start")),
    ("fre/configs/track_h_memory.yaml", ("track_h", "warm_start")),
]


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_configs_include_warm_start_blocks() -> None:
    for config_path, key_path in CONFIG_EXPECTATIONS:
        payload = _load_config(config_path)
        node = payload
        for key in key_path:
            assert key in node, f"{config_path} missing section {'.'.join(key_path)}"
            node = node[key]
        assert "save_path" in node, f"{config_path} warm_start.save_path missing"
        # load_path can be null but key should exist
        assert "load_path" in node, f"{config_path} warm_start.load_path missing"
