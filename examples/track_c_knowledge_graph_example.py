"""
Track C Knowledge Graph Example: Using K-Codex v2.0 related_claims

This example demonstrates how to use K-Codex v2.0's related_claims feature
to track scientific evolution across experimental iterations (Track C v1 → v2 → v3).

The K-Codex system with Epistemic Charter v2.0 integration enables:
- Tracking experimental lineage (which experiments build on which)
- Documenting mechanistic improvements
- Creating queryable knowledge graphs
- Supporting meta-analyses across labs

Author: Kosmic Lab Team
Date: November 10, 2025
License: MIT
"""

from pathlib import Path
from core.kcodex import KCodexWriter

# Initialize K-Codex writer with v2.0 schema
schema_path = Path("schemas/k_codex.json")
writer = KCodexWriter(schema_path)

print("=" * 80)
print("Track C Knowledge Graph Example: Scientific Evolution Tracking")
print("=" * 80)
print()

# =============================================================================
# Step 1: Track C v1 - Initial Rescue Attempt (Bug Discovery)
# =============================================================================
print("Creating K-Codex for Track C v1 (bug discovery)...")

track_c_v1 = writer.build_record(
    experiment="track_c_rescue_v1",
    params={
        "mask_threshold": 0.021,  # 21 mV
        "correction_strength": 0.15,
        "grid_voltage_range": [-1.0, 1.0],  # ❌ Bug! Should be [-100, 0]
    },
    estimators={
        "phi": "empirical",
        "te": {"estimator": "kraskov", "k": 3, "lag": 1}
    },
    metrics={
        "K": 0.765,
        "IoU": 0.0,  # Clipped to -1 mV, all dynamics broken
        "rescue_triggers": 200,
    },
    seed=2000,
    # Epistemic classification (defaults to E4, N1, M3)
    epistemic_tier_e="E4",  # Publicly reproducible
    epistemic_tier_n="N1",  # Research community consensus
    epistemic_tier_m="M3",  # Eternal preservation
    # No related_claims - this is the first attempt
)

track_c_v1_id = track_c_v1["run_id"]
print(f"✓ Track C v1 K-Codex created: {track_c_v1_id}")
print(f"  Classification: (E4, N1, M3) - gold standard reproducibility")
print(f"  Key finding: Grid clipping bug discovered (-1 mV instead of -100 mV)")
print()

# =============================================================================
# Step 2: Track C v2 - Bug Fixed, But Interference Problem
# =============================================================================
print("Creating K-Codex for Track C v2 (interference problem)...")

track_c_v2 = writer.build_record(
    experiment="track_c_rescue_v2",
    params={
        "mask_threshold": 0.021,
        "correction_strength": 0.5,  # Increased from 0.15
        "grid_voltage_range": [-100.0, 0.0],  # ✓ Bug fixed!
        "momentum_decay": 0.9,
        "momentum_new": 0.1,
    },
    estimators={
        "phi": "empirical",
        "te": {"estimator": "kraskov", "k": 3, "lag": 1}
    },
    metrics={
        "K": 0.706,
        "IoU": 0.706,  # Rescue worse than 77.6% baseline!
        "rescue_triggers": 3.5,
    },
    seed=2004,
    # Link to v1 (builds on it)
    related_claims=[
        {
            "relationship_type": "REFERENCES",
            "related_claim_id": track_c_v1_id,
            "context": "Fixed grid clipping bug discovered in v1 (-1 mV → -100 mV)"
        }
    ]
)

track_c_v2_id = track_c_v2["run_id"]
print(f"✓ Track C v2 K-Codex created: {track_c_v2_id}")
print(f"  Relationship: REFERENCES v1 (builds on bug fix)")
print(f"  Key finding: Rescue mechanism interferes with natural dynamics")
print()

# =============================================================================
# Step 3: Track C v3 - Attractor-Based Breakthrough
# =============================================================================
print("Creating K-Codex for Track C v3 (attractor breakthrough)...")

track_c_v3 = writer.build_record(
    experiment="track_c_rescue_v3",
    params={
        "mask_threshold": 0.021,
        "leak_reversal_target": -70.0,  # ✓ Novel attractor mechanism
        "conductance_acceleration": 1.5,
        "attractor_based": True,
    },
    estimators={
        "phi": "empirical",
        "te": {"estimator": "kraskov", "k": 3, "lag": 1}
    },
    metrics={
        "K": 0.520,
        "IoU": 0.788,  # 20% rescue success rate
        "rescue_triggers": 1.2,
    },
    seed=3000,
    # Multiple relationships - shows scientific evolution
    related_claims=[
        {
            "relationship_type": "SUPERCEDES",
            "related_claim_id": track_c_v2_id,
            "context": "Attractor-based mechanism (modifying physics) replaces "
                      "perturbation-based approach (forcing states). Eliminates "
                      "interference with natural dynamics."
        },
        {
            "relationship_type": "REFERENCES",
            "related_claim_id": track_c_v1_id,
            "context": "Maintains v1 bug fix (proper voltage range)"
        }
    ]
)

track_c_v3_id = track_c_v3["run_id"]
print(f"✓ Track C v3 K-Codex created: {track_c_v3_id}")
print(f"  Relationships:")
print(f"    - SUPERCEDES v2 (mechanistic improvement)")
print(f"    - REFERENCES v1 (maintains bug fix)")
print(f"  Key finding: Creating attractors beats forcing states")
print()

# =============================================================================
# Write K-Codices and complete
# =============================================================================
print("=" * 80)
print("✓ Track C Knowledge Graph Example Complete!")
print("=" * 80)
