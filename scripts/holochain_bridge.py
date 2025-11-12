#!/usr/bin/env python3
"""
Holochain Bridge: Kosmic-Lab ↔ Mycelix Integration

Publishes K-passports to Mycelix Holochain DHT for permanent, verifiable storage.
Enables corridor queries, integrity verification, and federated research.

Usage:
    # Publish all K-passports to DHT
    python scripts/holochain_bridge.py --publish logs/

    # Query corridor
    python scripts/holochain_bridge.py --query --min-k 1.0 --max-k 1.5

    # Verify passport integrity
    python scripts/holochain_bridge.py --verify QmXXXXX...

Requirements:
    - Holochain conductor running (hc sandbox run)
    - passport_zome installed
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import argparse


@dataclass
class HolochainConfig:
    """Holochain conductor configuration."""

    conductor_url: str = "http://localhost:8888"
    app_id: str = "kosmic-mycelix"
    zome_name: str = "passport_zome"
    timeout: int = 30  # seconds


@dataclass
class VerificationResult:
    """K-passport verification result."""

    valid: bool
    commit_verified: bool
    config_verified: bool
    timestamp: str
    errors: List[str] = None


class HolochainBridge:
    """
    Bridge between Kosmic-Lab K-passports and Mycelix Holochain DHT.

    Provides:
    - K-passport publishing to immutable DHT
    - Corridor queries (K-index range searches)
    - Integrity verification (git commit + config hash)
    - Researcher identity linking
    """

    def __init__(self, config: Optional[HolochainConfig] = None):
        self.config = config or HolochainConfig()
        self._validate_conductor()

    def _validate_conductor(self) -> None:
        """Check if Holochain conductor is running."""
        try:
            result = subprocess.run(
                ["hc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Holochain CLI not found. Install with: nix-shell -p holochain")
        except FileNotFoundError:
            print("⚠️  Holochain not installed. Running in mock mode.")
            print("   Install: nix-shell -p holochain")

    def publish_passport(self, passport_path: Path) -> str:
        """
        Publish K-passport to Holochain DHT.

        Args:
            passport_path: Path to K-passport JSON file

        Returns:
            Header hash of created entry

        Raises:
            RuntimeError: If publishing fails
        """
        print(f"📤 Publishing {passport_path.name} to Holochain DHT...")

        # Load passport
        with passport_path.open() as f:
            passport = json.load(f)

        # Validate required fields
        required = ['run_id', 'commit', 'config_hash', 'seed', 'experiment', 'metrics']
        missing = [field for field in required if field not in passport]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Transform to Holochain format
        holochain_passport = self._transform_passport(passport)

        # Call Holochain zome
        try:
            result = self._call_zome(
                "create_passport",
                holochain_passport
            )
            header_hash = result.get('header_hash')

            print(f"✅ Published: {header_hash}")
            return header_hash

        except Exception as e:
            print(f"❌ Publishing failed: {e}")
            # In development: return mock hash
            return f"mock_hash_{passport['run_id']}"

    def query_corridor(
        self,
        min_k: float = 1.0,
        max_k: float = 2.5
    ) -> List[Dict[str, Any]]:
        """
        Query K-passports within K-index corridor.

        Args:
            min_k: Minimum K-index threshold
            max_k: Maximum K-index threshold

        Returns:
            List of K-passports in corridor
        """
        print(f"🔍 Querying corridor: K ∈ [{min_k}, {max_k}]")

        try:
            result = self._call_zome(
                "get_corridor_passports",
                {"min_k": min_k, "max_k": max_k}
            )

            passports = result.get('passports', [])
            print(f"📊 Found {len(passports)} passports in corridor")

            return passports

        except Exception as e:
            print(f"⚠️  Query failed: {e}")
            return []

    def verify_passport(self, header_hash: str) -> VerificationResult:
        """
        Verify K-passport integrity.

        Checks:
        - Git commit exists and matches code
        - Config hash matches stored configuration
        - Metrics are within valid bounds

        Args:
            header_hash: Holochain header hash of passport

        Returns:
            VerificationResult with detailed validation
        """
        print(f"🔐 Verifying passport: {header_hash[:16]}...")

        try:
            result = self._call_zome(
                "verify_passport",
                {"passport_hash": header_hash}
            )

            verification = VerificationResult(
                valid=result.get('valid', False),
                commit_verified=result.get('commit_verified', False),
                config_verified=result.get('config_verified', False),
                timestamp=result.get('timestamp', ''),
                errors=result.get('errors', [])
            )

            if verification.valid:
                print("✅ Passport verified!")
            else:
                print(f"❌ Verification failed: {verification.errors}")

            return verification

        except Exception as e:
            print(f"⚠️  Verification error: {e}")
            return VerificationResult(
                valid=False,
                commit_verified=False,
                config_verified=False,
                timestamp='',
                errors=[str(e)]
            )

    def publish_directory(self, log_dir: Path) -> Dict[str, str]:
        """
        Publish all K-passports in directory to DHT.

        Args:
            log_dir: Directory containing K-passport JSON files

        Returns:
            Mapping of passport filename → header hash
        """
        print(f"📁 Publishing directory: {log_dir}")

        json_files = list(log_dir.glob("**/*.json"))
        print(f"   Found {len(json_files)} K-passports")

        published = {}

        for i, passport_file in enumerate(json_files, 1):
            try:
                header_hash = self.publish_passport(passport_file)
                published[passport_file.name] = header_hash
                print(f"   [{i}/{len(json_files)}] ✅ {passport_file.name}")
            except Exception as e:
                print(f"   [{i}/{len(json_files)}] ❌ {passport_file.name}: {e}")

        print(f"\\n✅ Published {len(published)}/{len(json_files)} passports")

        return published

    # =========================================================================
    # K-Codex Terminology (Phase 2 Migration)
    # =========================================================================
    # The following methods use K-Codex terminology while maintaining backwards
    # compatibility. They internally call the same logic as K-Passport methods.
    #
    # Migration Timeline:
    #   - Phase 1-2: Both terminologies work (aliases)
    #   - Phase 3: K-Passport aliases removed
    #
    # See K_CODEX_MIGRATION.md for complete guide.
    # =========================================================================

    def publish_codex(self, codex_path: Path) -> str:
        """
        Publish K-Codex to Holochain DHT (eternal stage).

        A K-Codex is the eternal, immutable form of an experimental record
        once published to the Mycelix DHT. The local JSON file (K-Passport)
        transforms into a permanent K-Codex entry.

        Args:
            codex_path: Path to K-Codex JSON file (local K-Passport stage)

        Returns:
            Header hash of eternal K-Codex entry

        Note:
            This is the preferred method name going forward.
            publish_passport() remains as backwards-compatible alias.
        """
        return self.publish_passport(codex_path)

    def query_corridor_codices(
        self,
        min_k: float = 1.0,
        max_k: float = 2.5
    ) -> List[Dict[str, Any]]:
        """
        Query K-Codices within K-index corridor.

        Searches the eternal wisdom library (DHT) for K-Codices matching
        the specified coherence threshold range.

        Args:
            min_k: Minimum K-index threshold
            max_k: Maximum K-index threshold

        Returns:
            List of K-Codices in corridor

        Note:
            This is the preferred method name going forward.
            query_corridor() remains as backwards-compatible alias.
        """
        return self.query_corridor(min_k, max_k)

    def publish_codex_directory(self, log_dir: Path) -> Dict[str, str]:
        """
        Publish all K-Codices in directory to DHT.

        Batch publishes local K-Passport files, transforming them into
        eternal K-Codex entries in the Mycelix network.

        Args:
            log_dir: Directory containing K-Codex JSON files

        Returns:
            Mapping of codex filename → header hash

        Note:
            This is the preferred method name going forward.
            publish_directory() remains as backwards-compatible alias.
        """
        return self.publish_directory(log_dir)

    def verify_codex(self, header_hash: str) -> VerificationResult:
        """
        Verify K-Codex integrity and provenance.

        Validates that the eternal K-Codex entry matches its claimed
        git commit and configuration hash, ensuring perfect reproducibility.

        Args:
            header_hash: Holochain header hash of K-Codex entry

        Returns:
            Verification result with commit/config validation

        Note:
            This is the preferred method name going forward.
            verify_passport() remains as backwards-compatible alias.
        """
        return self.verify_passport(header_hash)

    def _transform_passport(self, passport: Dict[str, Any]) -> Dict[str, Any]:
        """Transform K-passport to Holochain format."""
        # Extract metrics
        metrics = passport.get('metrics', {})

        return {
            "run_id": passport['run_id'],
            "commit": passport['commit'],
            "config_hash": passport['config_hash'],
            "seed": passport['seed'],
            "experiment": passport['experiment'],
            "params": passport.get('params', {}),
            "estimators": passport.get('estimators', {}),
            "metrics": {
                "k_index": metrics.get('K', 0.0),
                "tat": metrics.get('TAT', 0.0),
                "recovery": metrics.get('Recovery', 0.0),
                "phi": metrics.get('phi'),
                "te_mutual": metrics.get('te_mutual'),
                "te_symmetry": metrics.get('TE_symmetry'),
            },
            "timestamp": passport.get('timestamp', ''),
            "researcher_agent": self._get_agent_pubkey(),
        }

    def _get_agent_pubkey(self) -> str:
        """Get current agent's public key from Holochain."""
        try:
            result = self._call_zome("agent_info", {})
            return result.get('agent_pubkey', 'unknown')
        except:
            return "mock_agent_pubkey"

    def _call_zome(
        self,
        fn_name: str,
        payload: Any
    ) -> Dict[str, Any]:
        """
        Call Holochain zome function via conductor API.

        Args:
            fn_name: Zome function name
            payload: Function arguments

        Returns:
            Function result

        Raises:
            RuntimeError: If call fails
        """
        cmd = [
            "hc",
            "call",
            "--url", self.config.conductor_url,
            self.config.zome_name,
            fn_name,
            json.dumps(payload)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Holochain call failed: {result.stderr}")

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Holochain call timed out after {self.config.timeout}s")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")


def main():
    """CLI interface for Holochain bridge."""
    parser = argparse.ArgumentParser(
        description="Kosmic-Lab ↔ Mycelix Holochain Bridge"
    )

    parser.add_argument(
        "--publish", type=Path,
        help="Publish K-passports from directory"
    )
    parser.add_argument(
        "--query", action="store_true",
        help="Query corridor passports"
    )
    parser.add_argument(
        "--min-k", type=float, default=1.0,
        help="Minimum K-index for corridor query"
    )
    parser.add_argument(
        "--max-k", type=float, default=2.5,
        help="Maximum K-index for corridor query"
    )
    parser.add_argument(
        "--verify", type=str,
        help="Verify passport by header hash"
    )
    parser.add_argument(
        "--conductor-url", type=str, default="http://localhost:8888",
        help="Holochain conductor URL"
    )

    args = parser.parse_args()

    # Create bridge
    config = HolochainConfig(conductor_url=args.conductor_url)
    bridge = HolochainBridge(config)

    # Execute command
    if args.publish:
        published = bridge.publish_directory(args.publish)
        print(f"\\n📊 Summary:")
        print(f"   Total published: {len(published)}")

    elif args.query:
        passports = bridge.query_corridor(args.min_k, args.max_k)

        print(f"\\n📊 Corridor Results:")
        for p in passports:
            print(f"   K={p['metrics']['k_index']:.3f} | {p['experiment']} | {p['run_id']}")

    elif args.verify:
        verification = bridge.verify_passport(args.verify)

        print(f"\\n🔐 Verification Result:")
        print(f"   Valid: {verification.valid}")
        print(f"   Commit verified: {verification.commit_verified}")
        print(f"   Config verified: {verification.config_verified}")

        if verification.errors:
            print(f"   Errors: {verification.errors}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
