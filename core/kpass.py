"""
Utilities for writing K-passport telemetry compliant with the JSON schema.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from jsonschema import Draft7Validator

from core.config import ConfigBundle


class KPassportError(RuntimeError):
    """Raised when a passport record fails validation."""


class KPassportWriter:
    """Helper to emit run metadata adhering to schemas/k_passport.json."""

    def __init__(self, schema_path: Path) -> None:
        if not schema_path.exists():
            raise FileNotFoundError(f"K-passport schema missing: {schema_path}")
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self._validator = Draft7Validator(schema)
        self.schema_path = schema_path

    def build_record(
        self,
        experiment: str,
        params: Dict[str, Any],
        estimators: Dict[str, Any],
        metrics: Dict[str, Any],
        config: Optional[ConfigBundle] = None,
        seed: Optional[int] = None,
        universe: Optional[str] = None,
        environment: Optional[Dict[str, Any]] = None,
        ci: Optional[Dict[str, float]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct a full passport record."""
        record: Dict[str, Any] = {
            "run_id": run_id or str(uuid4()),
            "commit": self._infer_git_sha(),
            "config_hash": config.sha256 if config else "",
            "seed": seed if seed is not None else 0,
            "experiment": experiment,
            "params": params,
            "estimators": estimators,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if universe:
            record["universe"] = universe
        if environment:
            record["environment"] = environment
        if ci:
            record["ci"] = ci
        self._validate(record)
        return record

    def write(self, record: Dict[str, Any], output_dir: Path) -> Path:
        """Persist record to JSON file inside output_dir."""
        self._validate(record)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{record['run_id']}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
        return path

    def _validate(self, record: Dict[str, Any]) -> None:
        errors = sorted(self._validator.iter_errors(record), key=lambda e: e.path)
        if errors:
            msg = "; ".join(err.message for err in errors)
            raise KPassportError(f"Passport validation failed: {msg}")

    @staticmethod
    def _infer_git_sha() -> str:
        """Best-effort HEAD commit hash; returns 'UNKNOWN' if unavailable."""
        head = Path(".git/HEAD")
        if not head.exists():
            return "UNKNOWN"

        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = Path(".git") / ref.split(" ", 1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        elif len(ref) == 40:
            return ref
        return "UNKNOWN"
