"""
K-Codex: Eternal experimental records for reproducible consciousness research.

This module provides utilities for creating K-Codices (formerly K-Passports) -
standardized, immutable metadata records that ensure complete experimental
reproducibility.

Terminology Evolution:
    - K-Passport: Local JSON file (pre-publication stage)
    - K-Codex: Immutable DHT entry in Mycelix network (published, eternal)

Etymology:
    "Codex" from Latin for "bound book" or "tree trunk" - symbolizing permanent
    knowledge compiled into the mycelial network, where each experiment becomes
    an eternal page in the collective wisdom library.

Backwards Compatibility:
    - KPassport class alias maintained for existing code
    - KPassportWriter alias maintained for existing code
    - All old function names continue to work

See K_CODEX_MIGRATION.md for complete migration guide.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from jsonschema import Draft7Validator

from core.config import ConfigBundle
from core.utils import infer_git_sha


class KCodexError(RuntimeError):
    """Raised when a K-Codex record fails validation."""


class KCodexWriter:
    """
    Helper to emit K-Codex records adhering to schemas/k_codex.json.

    A K-Codex is an eternal, immutable record of an experiment that ensures
    bit-for-bit reproducibility years or decades later. Once published to
    Mycelix DHT, it becomes part of the permanent collective wisdom library.

    Local Stage (K-Passport):
        JSON file on filesystem during development/testing

    Eternal Stage (K-Codex):
        Published to Holochain DHT with cryptographic verification

    Example:
        >>> writer = KCodexWriter(Path("schemas/k_codex.json"))
        >>> codex = writer.build_record(
        ...     experiment="coherence_test",
        ...     params={"energy_gradient": 0.5},
        ...     estimators={"phi": "star", "te": {"estimator": "kraskov", "k": 3}},
        ...     metrics={"K": 1.23, "TAT": 0.615},
        ...     seed=42
        ... )
        >>> path = writer.write(codex, Path("logs/codices"))
        >>> print(f"K-Codex (local stage) written to {path}")

    See Also:
        - K_CODEX_EXPLAINED.md for conceptual overview
        - K_CODEX_MIGRATION.md for migration from K-Passport
    """

    def __init__(self, schema_path: Path) -> None:
        """
        Initialize K-Codex writer with JSON schema validation.

        Args:
            schema_path: Path to k_codex.json schema file
                        (or k_passport.json for backwards compatibility)

        Raises:
            FileNotFoundError: If schema file doesn't exist
        """
        if not schema_path.exists():
            raise FileNotFoundError(f"K-Codex schema missing: {schema_path}")
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
        researcher_agent: Optional[str] = None,
        # Epistemic Charter v2.0 fields (with smart defaults)
        epistemic_tier_e: str = "E4",
        epistemic_tier_n: str = "N1",
        epistemic_tier_m: str = "M3",
        verifiability_method: str = "PublicCode",
        verifiability_status: str = "Verified",
        related_claims: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Construct a complete K-Codex v2.0 record with Epistemic Charter compliance.

        This creates a local K-Codex (K-Passport stage) - a JSON record capturing
        all information needed to reproduce the experiment perfectly, plus epistemic
        classification following the Mycelix Epistemic Charter v2.0.

        Required Fields:
            experiment: Experiment type identifier
            params: All hyperparameters used
            estimators: Exact algorithms (Φ variant, TE parameters, etc.)
            metrics: Output measurements (K, TAT, Recovery, etc.)

        Auto-Generated:
            run_id: Unique UUID (or provided)
            commit: Git SHA of exact code version
            config_hash: SHA256 digest of configuration
            timestamp: ISO 8601 datetime (UTC)

        Optional:
            seed: Random seed for reproducibility
            universe: Universe identifier for multi-universe sims
            environment: System environment details
            ci: Confidence intervals for metrics
            researcher_agent: AgentPubKey for Mycelix (added at publication)

        Epistemic Charter v2.0 Fields:
            epistemic_tier_e: E-Axis (E0-E4) - Empirical verifiability
                             Default: E4 (Publicly Reproducible - highest standard)
            epistemic_tier_n: N-Axis (N0-N3) - Normative authority
                             Default: N1 (Communal Consensus - research community)
            epistemic_tier_m: M-Axis (M0-M3) - Materiality/permanence
                             Default: M3 (Foundational - eternal preservation)
            verifiability_method: How to verify the claim
                                 Default: PublicCode (open source + data + checksums)
            verifiability_status: Current verification status
                                 Default: Verified
            related_claims: Knowledge graph relationships to other K-Codices
                           Enables tracking scientific evolution (v1 → v2 → v3)

        Kosmic Lab Defaults:
            All experiments default to (E4, N1, M3) - the gold standard for
            reproducible computational research:
            - E4: Anyone can reproduce with open code/data
            - N1: Accepted by research community
            - M3: Preserved eternally on DHT

        Returns:
            Validated K-Codex v2.0 record ready for writing or publishing

        Raises:
            KCodexError: If record fails schema validation

        Example:
            >>> # Basic usage (uses smart defaults)
            >>> codex = writer.build_record(
            ...     experiment="track_c_rescue_v3",
            ...     params={"leak_reversal": -70.0},
            ...     estimators={"phi": "star", "te": {"estimator": "kraskov", "k": 3}},
            ...     metrics={"K": 1.45, "TAT": 0.72},
            ...     seed=42
            ... )
            >>> # Automatically gets E4, N1, M3 classification!

            >>> # With knowledge graph relationships
            >>> codex_v3 = writer.build_record(
            ...     experiment="track_c_rescue_v3",
            ...     params={...},
            ...     metrics={...},
            ...     related_claims=[
            ...         {
            ...             "relationship_type": "SUPERCEDES",
            ...             "related_claim_id": track_c_v2_run_id,
            ...             "context": "Attractor-based replaces perturbation approach"
            ...         }
            ...     ]
            ... )
        """
        record: Dict[str, Any] = {
            "run_id": run_id or str(uuid4()),
            "commit": infer_git_sha(),
            "config_hash": config.sha256 if config else "",
            "seed": seed if seed is not None else 0,
            "experiment": experiment,
            "params": params,
            "estimators": estimators,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Epistemic Charter v2.0 fields
            "epistemic_tier_e": epistemic_tier_e,
            "epistemic_tier_n": epistemic_tier_n,
            "epistemic_tier_m": epistemic_tier_m,
            "verifiability": {
                "method": verifiability_method,
                "status": verifiability_status,
            },
        }
        if universe:
            record["universe"] = universe
        if environment:
            record["environment"] = environment
        if ci:
            record["ci"] = ci
        if researcher_agent:
            record["researcher_agent"] = researcher_agent
        if related_claims:
            record["related_claims"] = related_claims
        self._validate(record)
        return record

    def write(self, record: Dict[str, Any], output_dir: Path) -> Path:
        """
        Persist K-Codex record to JSON file (local K-Passport stage).

        Writes the record to a JSON file named with the run_id. This is the
        "K-Passport" stage - before publication to Mycelix DHT.

        To publish to DHT (eternal K-Codex stage):
            >>> from scripts.holochain_bridge import HolochainBridge
            >>> bridge = HolochainBridge()
            >>> header_hash = bridge.publish_codex(record)

        Args:
            record: K-Codex record from build_record()
            output_dir: Directory to write JSON file

        Returns:
            Path to written JSON file

        Raises:
            KCodexError: If record fails validation
        """
        self._validate(record)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{record['run_id']}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
        return path

    def _validate(self, record: Dict[str, Any]) -> None:
        """Validate record against K-Codex JSON schema."""
        errors = sorted(self._validator.iter_errors(record), key=lambda e: e.path)
        if errors:
            msg = "; ".join(err.message for err in errors)
            raise KCodexError(f"K-Codex validation failed: {msg}")


# =============================================================================
# Backwards Compatibility Aliases (Phase 1-2 Migration)
# =============================================================================
#
# To support existing code during migration, we provide aliases:
#   - KPassport → KCodex (terminology evolution)
#   - KPassportWriter → KCodexWriter (class alias)
#   - KPassportError → KCodexError (exception alias)
#
# These will be removed in Phase 3 (Month 2) after full migration.
# See K_CODEX_MIGRATION.md for timeline and migration guide.
#
# Existing code continues to work:
#   >>> from core.kcodex import KPassportWriter  # Still works!
#   >>> writer = KPassportWriter(schema_path)
#   >>> passport = writer.build_record(...)      # Still works!
#
# New code should use K-Codex terminology:
#   >>> from core.kcodex import KCodexWriter     # Preferred!
#   >>> writer = KCodexWriter(schema_path)
#   >>> codex = writer.build_record(...)
# =============================================================================

# Type alias for backwards compatibility
KPassport = Dict[str, Any]  # Local JSON record (pre-publication)
KCodex = Dict[str, Any]  # Same structure, but conceptually eternal once published

# Class aliases for backwards compatibility
KPassportWriter = KCodexWriter
KPassportError = KCodexError

# Legacy imports for complete backwards compatibility
__all__ = [
    # Primary (new) names
    "KCodex",
    "KCodexWriter",
    "KCodexError",
    # Backwards compatibility aliases
    "KPassport",
    "KPassportWriter",
    "KPassportError",
]
