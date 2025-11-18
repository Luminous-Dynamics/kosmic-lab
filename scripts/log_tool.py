#!/usr/bin/env python3
"""
Episode log utility for Track G JSONL streams.

Usage examples:

    # Show the last 20 records
    python scripts/log_tool.py tail --path logs/track_g/episodes/phase_g2.jsonl

    # Follow a live log
    python scripts/log_tool.py tail --path logs/track_g/episodes/phase_g2.jsonl --follow

    # Validate a log file
    python scripts/log_tool.py validate --path logs/track_g/episodes/phase_g2.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def tail_file(path: Path, lines: int, follow: bool) -> None:
    if not path.exists():
        raise SystemExit(f"❌ Log file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        buffer = fh.readlines()
        for line in buffer[-lines:]:
            print(line.rstrip())
        if follow:
            while True:
                where = fh.tell()
                line = fh.readline()
                if not line:
                    time.sleep(0.5)
                    fh.seek(where)
                else:
                    print(line.rstrip())


def validate_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"❌ Log file not found: {path}")
    episode_index = None
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"❌ Invalid JSON at line {lineno}: {exc}")
            if "phase" not in payload:
                raise SystemExit(f"❌ Missing 'phase' field at line {lineno}")
            if "k_index" not in payload:
                raise SystemExit(f"❌ Missing 'k_index' field at line {lineno}")
            k_val = payload["k_index"]
            if not isinstance(k_val, (int, float)):
                raise SystemExit(f"❌ k_index must be numeric (line {lineno})")
            epi = payload.get("episode_index") or payload.get("episode_global_index")
            if epi is not None:
                if episode_index is not None and epi < episode_index:
                    raise SystemExit(f"❌ Episode index decreased at line {lineno} ({episode_index} -> {epi})")
                episode_index = epi
    print(f"✅ Log validation passed for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode log utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tail_parser = subparsers.add_parser("tail", help="Display last lines of log")
    tail_parser.add_argument("--path", type=Path, required=True)
    tail_parser.add_argument("--lines", type=int, default=20)
    tail_parser.add_argument("--follow", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate JSONL structure")
    validate_parser.add_argument("--path", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "tail":
        tail_file(args.path, args.lines, args.follow)
    elif args.command == "validate":
        validate_file(args.path)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
