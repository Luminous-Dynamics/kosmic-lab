#!/usr/bin/env python3
"""
Checkpoint helper utilities for Track G/H warm-start workflow.

Usage examples:
    # Show metadata for a checkpoint
    python scripts/checkpoint_tool.py info --path logs/track_g/checkpoints/phase_g2_latest.json

    # List all checkpoints in a directory
    python scripts/checkpoint_tool.py list --dir logs/track_g/checkpoints

    # Copy a checkpoint to a new file (e.g., best run archive)
    python scripts/checkpoint_tool.py copy --src logs/track_g/checkpoints/phase_g2_latest.json \
                                           --dest logs/track_g/checkpoints/phase_g2_best.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    from scripts.config_registry import lookup as registry_lookup  # type: ignore
except Exception:  # pragma: no cover
    registry_lookup = None


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"❌ Checkpoint not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"❌ Invalid checkpoint JSON ({path}): {exc}") from exc


def cmd_info(path: Path) -> None:
    payload = load_checkpoint(path)
    metadata = payload.get("metadata", {})
    saved_at = payload.get("saved_at", "unknown")
    agent_cls = payload.get("agent_class", "unknown")
    context = payload.get("phase") or payload.get("context", "n/a")
    print("📄 Checkpoint Info")
    print("=================")
    print(f"Path:        {path}")
    print(f"Agent:       {agent_cls}")
    print(f"Phase/Task:  {context}")
    print(f"Saved At:    {saved_at}")
    if "max_k" in metadata:
        print(f"Max K:       {metadata['max_k']}")
    if "episodes_completed" in metadata:
        print(f"Episodes:    {metadata['episodes_completed']}")
    if metadata:
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
        config_meta = metadata.get("config")
        if config_meta and registry_lookup:
            entry = registry_lookup(hash_value=config_meta.get("sha256"))
            if entry:
                print(f"\n🗂  Config Registry: {entry.get('label')} ({entry.get('notes')})")


def cmd_list(directory: Path) -> None:
    if not directory.exists():
        raise SystemExit(f"❌ Directory not found: {directory}")
    checkpoints = sorted(directory.glob("*.json"))
    if not checkpoints:
        print(f"(empty) No checkpoints found under {directory}")
        return
    print(f"📂 Checkpoints under {directory}:")
    for path in checkpoints:
        payload = load_checkpoint(path)
        saved_at = payload.get("saved_at", "unknown")
        context = payload.get("phase") or payload.get("context", "n/a")
        print(f" - {path.name} | {context} | {saved_at}")


def cmd_copy(src: Path, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        raise SystemExit(f"❌ Destination already exists (use --overwrite): {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"✅ Copied {src} → {dest} at {datetime.now().isoformat()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Checkpoint management helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="Show metadata for a checkpoint")
    info_parser.add_argument("--path", type=Path, required=True)

    list_parser = subparsers.add_parser("list", help="List checkpoints in a directory")
    list_parser.add_argument("--dir", type=Path, required=True)

    copy_parser = subparsers.add_parser("copy", help="Copy a checkpoint to a new file")
    copy_parser.add_argument("--src", type=Path, required=True)
    copy_parser.add_argument("--dest", type=Path, required=True)
    copy_parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing destination")

    args = parser.parse_args(argv)
    if args.command == "info":
        cmd_info(args.path)
    elif args.command == "list":
        cmd_list(args.dir)
    elif args.command == "copy":
        cmd_copy(args.src, args.dest, args.overwrite)
    elif args.command == "extract-config":
        payload = load_checkpoint(args.path)
        snapshot = payload.get("metadata", {}).get("config_snapshot")
        if not snapshot:
            raise SystemExit("❌ No config snapshot stored in this checkpoint.")
        args.output.write_text(snapshot, encoding="utf-8")
        print(f"✅ Config snapshot written to {args.output}")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
    extract_parser = subparsers.add_parser("extract-config", help="Write embedded config snapshot")
    extract_parser.add_argument("--path", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
