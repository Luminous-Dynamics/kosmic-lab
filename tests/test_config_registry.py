from __future__ import annotations

from pathlib import Path

import json

import pytest

from scripts import config_registry
from scripts import validate_registry


def test_register_and_lookup(tmp_path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.json"
    monkeypatch.setattr(config_registry, "REGISTRY_PATH", registry_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("alpha: 1", encoding="utf-8")

    entry = config_registry.register_config(config_file, "Test Config", "notes")
    assert entry["label"] == "Test Config"

    found = config_registry.lookup(hash_value=entry["hash"])
    assert found["path"] == str(config_file.resolve())

    found2 = config_registry.lookup(config_path=config_file)
    assert found2["hash"] == entry["hash"]


def test_diff_configs(tmp_path) -> None:
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("alpha: 1\nbeta: 2\n", encoding="utf-8")
    cfg_b.write_text("alpha: 1\nbeta: 3\n", encoding="utf-8")
    diff = config_registry.diff_configs(cfg_a, cfg_b)
    assert "-beta: 2" in diff
    assert "+beta: 3" in diff


def test_validate_registry_script(tmp_path) -> None:
    reg = tmp_path / "registry.json"
    data = {"hash": {"hash": "abc", "path": "foo"}}
    reg.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_registry.validate_registry(reg)

    reg.write_text('{"hash":{"hash":"abc","path":"foo"}}', encoding="utf-8")
    with pytest.raises(SystemExit):
        validate_registry.validate_registry(reg)
