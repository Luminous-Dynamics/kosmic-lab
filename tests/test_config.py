from __future__ import annotations

from pathlib import Path

import pytest

from core.config import ConfigBundle, ConfigurationError, load_yaml_config


def test_load_yaml_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("alpha: 1\nbeta: {gamma: 2}\n", encoding="utf-8")

    bundle = load_yaml_config(config_file)
    assert isinstance(bundle, ConfigBundle)
    assert bundle.payload["alpha"] == 1
    assert bundle.sha256


def test_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_yaml_config(tmp_path / "missing.yaml")
