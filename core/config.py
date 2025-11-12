"""
Centralised configuration utilities.

YAML loader that preserves ordering and hashes configs for K-passport tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigurationError(RuntimeError):
    """Raised when configuration files are missing or malformed."""


@dataclass(frozen=True)
class ConfigBundle:
    """Container for raw config and deterministic hash."""

    payload: Dict[str, Any]
    sha256: str


def load_yaml_config(path: Path) -> ConfigBundle:
    """
    Load YAML configuration and compute a stable SHA256 digest.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        ConfigBundle with parsed payload and hex digest.
    """
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}") from exc

    digest = sha256(text.encode("utf-8")).hexdigest()
    return ConfigBundle(payload=payload, sha256=digest)
