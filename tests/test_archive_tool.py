from __future__ import annotations

from pathlib import Path

from scripts import archive_tool


def test_create_and_verify_archive(tmp_path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    checkpoint.write_text('{"state": {}, "metadata": {"config_snapshot": "alpha: 1\\n"}}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("alpha: 1", encoding="utf-8")
    log = tmp_path / "log.jsonl"
    log.write_text('{"phase":"G2","k_index":0.5}\n', encoding="utf-8")

    output = tmp_path / "bundle.tar.gz"
    archive_tool.create_archive(checkpoint, config, log, output)
    archive_tool.verify_archive(output)
    metadata = archive_tool.load_metadata(output)
    assert "files" in metadata
    assert "config_snapshot" in metadata["files"]
    assert "checkpoint_config_snapshot" in metadata["files"]
