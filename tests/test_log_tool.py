from __future__ import annotations

from scripts import log_tool


def test_validate_file_pass(tmp_path) -> None:
    log_path = tmp_path / "log.jsonl"
    log_path.write_text('{"phase": "G2", "episode_index": 1, "k_index": 0.5}\n', encoding="utf-8")
    log_tool.validate_file(log_path)


def test_validate_file_rejects_non_numeric_k(tmp_path) -> None:
    log_path = tmp_path / "bad.jsonl"
    log_path.write_text('{"phase": "G2", "k_index": "oops"}\n', encoding="utf-8")
    try:
        log_tool.validate_file(log_path)
    except SystemExit as exc:
        assert "k_index must be numeric" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for invalid log")
