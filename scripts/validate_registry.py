#!/usr/bin/env python3
"""
Validate configs/config_registry.json formatting.
Ensures file is valid JSON and canonical (sorted keys, indent=2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_registry(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text or "{}")
    canonical = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if text != canonical:
        raise SystemExit(
            f"❌ Registry {path} not canonical. "
            "Run `poetry run python -m json.tool configs/config_registry.json > tmp && mv tmp ...`."
        )
    print(f"✅ Registry {path} valid.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("configs/config_registry.json"))
    args = parser.parse_args()
    validate_registry(args.path)


if __name__ == "__main__":
    main()
