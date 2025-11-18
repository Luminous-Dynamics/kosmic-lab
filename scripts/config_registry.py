#!/usr/bin/env python3
"""
Config registry helper for Kosmic Lab.

Tracks SHA256 hashes of YAML configs with human-readable labels.

Usage:
    # Register a config
    python scripts/config_registry.py register --config fre/configs/track_g_phase_g2.yaml --label "Track G Phase G2" --notes "Baseline extended training"

    # Lookup by hash
    python scripts/config_registry.py lookup --hash 1234abcd...

    # Lookup by config path
    python scripts/config_registry.py lookup --config fre/configs/track_g_phase_g2.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import difflib

REGISTRY_PATH = Path("configs/config_registry.json")


def compute_hash(config_path: Path) -> str:
    data = config_path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_registry() -> Dict[str, Dict[str, str]]:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(data: Dict[str, Dict[str, str]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def register_config(config_path: Path, label: str, notes: Optional[str]) -> Dict[str, str]:
    config_path = config_path.resolve()
    digest = compute_hash(config_path)
    registry = load_registry()
    registry[digest] = {
        "hash": digest,
        "path": str(config_path),
        "label": label,
        "notes": notes or "",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_registry(registry)
    return registry[digest]


def lookup(hash_value: Optional[str] = None, config_path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    registry = load_registry()
    if hash_value:
        return registry.get(hash_value)
    if config_path:
        digest = compute_hash(config_path.resolve())
        return registry.get(digest)
    return None

def diff_configs(config_a: Path, config_b: Path) -> str:
    text_a = config_a.read_text(encoding="utf-8").splitlines()
    text_b = config_b.read_text(encoding="utf-8").splitlines()
    diff = difflib.unified_diff(
        text_a,
        text_b,
        fromfile=str(config_a),
        tofile=str(config_b),
        lineterm="",
    )
    return "\n".join(diff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config registry helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reg_parser = subparsers.add_parser("register", help="Register/update a config entry")
    reg_parser.add_argument("--config", type=Path, required=True)
    reg_parser.add_argument("--label", type=str, required=True)
    reg_parser.add_argument("--notes", type=str, default=None)

    lookup_parser = subparsers.add_parser("lookup", help="Lookup config metadata")
    lookup_parser.add_argument("--hash", type=str, default=None)
    lookup_parser.add_argument("--config", type=Path, default=None)

    diff_parser = subparsers.add_parser("diff", help="Diff two configs")
    diff_parser.add_argument("--config-a", type=Path, required=True)
    diff_parser.add_argument("--config-b", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "register":
        entry = register_config(args.config, args.label, args.notes)
        print("✅ Registered config:")
        print(json.dumps(entry, indent=2))
    elif args.command == "lookup":
        entry = lookup(args.hash, args.config)
        if entry:
            print(json.dumps(entry, indent=2))
        else:
            print("⚠️  No entry found.")
    elif args.command == "diff":
        print(diff_configs(args.config_a, args.config_b))


if __name__ == "__main__":
    main()
