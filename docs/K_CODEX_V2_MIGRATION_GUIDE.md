# K-Codex v2.0 Migration Guide

**From**: K-Passport/K-Codex v1.0  
**To**: K-Codex v2.0 (Epistemic Charter Integration)  
**Date**: November 10, 2025  
**Breaking Changes**: None (fully backwards compatible)

---

## Executive Summary

K-Codex v2.0 adds **Epistemic Charter v2.0 classification** while maintaining **100% backwards compatibility**. All existing code continues to work unchanged.

### What's New in v2.0

✅ **E-Axis Classification**: Empirical verifiability (E0-E4)  
✅ **N-Axis Classification**: Normative authority (N0-N3)  
✅ **M-Axis Classification**: Materiality/permanence (M0-M3)  
✅ **Verifiability Metadata**: Method and status tracking  
✅ **Knowledge Graph**: related_claims field for scientific evolution  
✅ **Smart Defaults**: (E4, N1, M3) for all experiments  
✅ **Mycelix DHT Ready**: Prepared for decentralized publication

### Migration Effort

- **Existing Code**: ✅ Works unchanged (0 minutes)
- **To Use New Features**: 🎯 Add optional parameters (2-5 minutes)
- **Update Documentation**: 📝 Add epistemic context (10-15 minutes)

---

## Table of Contents

1. [Backwards Compatibility](#backwards-compatibility)
2. [Schema Changes](#schema-changes)
3. [Code Migration (Optional)](#code-migration-optional)
4. [Best Practices](#best-practices)
5. [Common Migration Patterns](#common-migration-patterns)
6. [Validation and Testing](#validation-and-testing)

---

## Backwards Compatibility

### Key Point: No Action Required

**All existing code works without modification!**

```python
# This code from v1.0 works identically in v2.0:
from core.kcodex import KCodexWriter

writer = KCodexWriter(Path("schemas/k_passport.json"))  # Still works!
codex = writer.build_record(
    experiment="track_b",
    params={...},
    estimators={...},
    metrics={...}
)
# ✅ This creates a valid K-Codex v2.0 with smart defaults
```

### What Happens Automatically

When you use the v1.0 API, v2.0 automatically:
1. Adds epistemic_tier_e = "E4" (Publicly Reproducible)
2. Adds epistemic_tier_n = "N1" (Communal Consensus)
3. Adds epistemic_tier_m = "M3" (Foundational)
4. Adds verifiability = {"method": "PublicCode", "status": "Verified"}
5. Validates against the v2.0 schema

### Schema Files

Both schemas are supported:
- `schemas/k_passport.json` - v1.0 schema (still valid)
- `schemas/k_codex.json` - v2.0 schema (with epistemic fields)

**Recommendation**: Use `k_codex.json` for new code to get validation of epistemic fields.

---

## Schema Changes

### New Required Fields (Auto-Populated)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epistemic_tier_e` | string | "E4" | Empirical verifiability (E0-E4) |
| `epistemic_tier_n` | string | "N1" | Normative authority (N0-N3) |
| `epistemic_tier_m` | string | "M3" | Materiality/permanence (M0-M3) |
| `verifiability` | object | {...} | Verification metadata |
| `verifiability.method` | string | "PublicCode" | How to verify |
| `verifiability.status` | string | "Verified" | Current status |

### New Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `verifiability.proof_cid` | string | null | IPFS CID for proof data |
| `related_claims` | array | [] | Knowledge graph relationships |

### Existing Fields (Unchanged)

All v1.0 fields remain identical:
- `run_id`, `commit`, `config_hash`, `seed`
- `experiment`, `params`, `estimators`, `metrics`
- `timestamp`, `universe`, `environment`, `ci`
- `researcher_agent`

---

## Code Migration (Optional)

### Option 1: No Changes (Recommended for Most)

Keep your existing code - it works perfectly!

```python
# v1.0 code (still works in v2.0)
writer = KCodexWriter(Path("schemas/k_passport.json"))
codex = writer.build_record(
    experiment="track_b_sac",
    params={...},
    estimators={...},
    metrics={...}
)
# Gets smart defaults: (E4, N1, M3)
```

### Option 2: Update Schema Path Only

Point to new schema to get validation:

```python
# v2.0 code (minimal change)
writer = KCodexWriter(Path("schemas/k_codex.json"))  # Changed!
codex = writer.build_record(
    experiment="track_b_sac",
    params={...},
    estimators={...},
    metrics={...}
)
# Same behavior, now validates epistemic fields
```

### Option 3: Explicit Epistemic Classification

Override defaults for special cases:

```python
# v2.0 code (full features)
writer = KCodexWriter(Path("schemas/k_codex.json"))
codex = writer.build_record(
    experiment="pilot_study",
    params={...},
    estimators={...},
    metrics={...},
    # Optional: override defaults
    epistemic_tier_e="E1",  # Testimonial (pilot study)
    epistemic_tier_m="M2",  # Persistent (not eternal yet)
    verifiability_status="Pending",
)
```

### Option 4: Knowledge Graph Integration

Track experimental evolution:

```python
# v2.0 code (knowledge graph)
writer = KCodexWriter(Path("schemas/k_codex.json"))

# First experiment
v1 = writer.build_record(experiment="track_c_v1", ...)
v1_id = v1["run_id"]

# Follow-up experiment with relationship
v2 = writer.build_record(
    experiment="track_c_v2",
    params={...},
    metrics={...},
    # New feature: link experiments
    related_claims=[
        {
            "relationship_type": "REFERENCES",
            "related_claim_id": v1_id,
            "context": "Fixed bug discovered in v1"
        }
    ]
)

# Breakthrough experiment
v3 = writer.build_record(
    experiment="track_c_v3",
    params={...},
    metrics={...},
    related_claims=[
        {
            "relationship_type": "SUPERCEDES",
            "related_claim_id": v2["run_id"],
            "context": "Attractor-based mechanism"
        }
    ]
)
```

---

## Best Practices

### For Published Experiments

Use defaults - they're perfect!

```python
codex = writer.build_record(...)
# Automatically gets (E4, N1, M3)
# E4 = Publicly Reproducible
# N1 = Research Community Consensus
# M3 = Eternal Preservation
```

### For Pilot Studies

Override M-Axis:

```python
pilot = writer.build_record(
    ...,
    epistemic_tier_m="M2",  # Persistent (may prune after 1 year)
    verifiability_status="Pending"
)
```

### For Work-in-Progress

Override E-Axis:

```python
wip = writer.build_record(
    ...,
    epistemic_tier_e="E1",  # Testimonial (not fully documented)
    epistemic_tier_m="M2"   # Persistent (not eternal yet)
)
```

### For System Logs

Override all axes:

```python
log = writer.build_record(
    ...,
    epistemic_tier_e="E1",  # Testimonial
    epistemic_tier_n="N0",  # Personal only
    epistemic_tier_m="M1"   # Temporal (prune after use)
)
```

---

## Common Migration Patterns

### Pattern 1: Batch Processing (No Changes)

```python
# Existing v1.0 code works unchanged
for seed in [100, 200, 300]:
    codex = writer.build_record(
        experiment=f"track_b_seed_{seed}",
        params={"seed": seed},
        metrics=compute_metrics(seed)
    )
    writer.write(codex, output_dir)
# All get (E4, N1, M3) automatically
```

### Pattern 2: Experimental Series (Add Relationships)

```python
# Enhanced v2.0: track experiment evolution
codices = []
for i, config in enumerate(configs):
    codex = writer.build_record(
        experiment=f"optimization_step_{i}",
        params=config,
        metrics=evaluate(config),
        # Link to previous step
        related_claims=[
            {
                "relationship_type": "REFERENCES",
                "related_claim_id": codices[-1]["run_id"],
                "context": f"Iteration {i} building on {i-1}"
            }
        ] if codices else []
    )
    codices.append(codex)
    writer.write(codex, output_dir)
```

### Pattern 3: Multi-Track Experiments (Same As v1.0)

```python
# Track B, Track C - no changes needed
track_b = writer.build_record(experiment="track_b_sac", ...)
track_c = writer.build_record(experiment="track_c_rescue", ...)
# Both get appropriate defaults
```

---

## Validation and Testing

### Check Your Migration

Run this script to verify everything works:

```python
from pathlib import Path
from core.kcodex import KCodexWriter

# Test v1.0 compatibility
print("Testing v1.0 compatibility...")
writer_v1 = KCodexWriter(Path("schemas/k_passport.json"))
codex_v1 = writer_v1.build_record(
    experiment="test",
    params={},
    estimators={"phi": "empirical", "te": {"estimator": "kraskov", "k": 3, "lag": 1}},
    metrics={"K": 1.0}
)
print("✓ v1.0 API works")

# Test v2.0 schema
print("\nTesting v2.0 schema...")
writer_v2 = KCodexWriter(Path("schemas/k_codex.json"))
codex_v2 = writer_v2.build_record(
    experiment="test",
    params={},
    estimators={"phi": "empirical", "te": {"estimator": "kraskov", "k": 3, "lag": 1}},
    metrics={"K": 1.0}
)

# Verify epistemic fields
assert "epistemic_tier_e" in codex_v2
assert "epistemic_tier_n" in codex_v2
assert "epistemic_tier_m" in codex_v2
assert "verifiability" in codex_v2
print("✓ v2.0 schema validation works")

# Test knowledge graph
print("\nTesting knowledge graph...")
codex_v3 = writer_v2.build_record(
    experiment="test_related",
    params={},
    estimators={"phi": "empirical", "te": {"estimator": "kraskov", "k": 3, "lag": 1}},
    metrics={"K": 1.0},
    related_claims=[
        {
            "relationship_type": "REFERENCES",
            "related_claim_id": codex_v2["run_id"],
            "context": "Test relationship"
        }
    ]
)
assert "related_claims" in codex_v3
assert len(codex_v3["related_claims"]) == 1
print("✓ Knowledge graph works")

print("\n🎉 All migration tests passed!")
```

### Expected Output

```
Testing v1.0 compatibility...
✓ v1.0 API works

Testing v2.0 schema...
✓ v2.0 schema validation works

Testing knowledge graph...
✓ Knowledge graph works

🎉 All migration tests passed!
```

---

## Summary

### Migration Checklist

- [ ] **Nothing required!** - Code works unchanged
- [ ] (Optional) Update schema path: `k_passport.json` → `k_codex.json`
- [ ] (Optional) Add knowledge graph relationships for experimental series
- [ ] (Optional) Override epistemic defaults for special cases
- [ ] Run validation script to verify

### Key Takeaways

✅ **Zero Breaking Changes**: All v1.0 code works in v2.0  
✅ **Smart Defaults**: (E4, N1, M3) perfect for 99% of experiments  
✅ **Backwards Compatible**: Both schemas supported indefinitely  
✅ **Forward Compatible**: Ready for Mycelix DHT integration  
✅ **Knowledge Graph**: Optional but powerful for tracking evolution

### Next Steps

1. **No migration needed** - current code works perfectly
2. **To use new features** - see examples in `examples/track_c_knowledge_graph_example.py`
3. **For deep understanding** - read `docs/K_CODEX_EPISTEMIC_CLASSIFICATION.md`
4. **For manuscript integration** - see updated `manuscript_paper1.tex`

---

## Resources

- **Epistemic Charter**: https://github.com/Luminous-Dynamics/Mycelix-Core/blob/main/docs/architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md
- **Schema**: `schemas/k_codex.json` (v2.0) or `schemas/k_passport.json` (v1.0)
- **Implementation**: `core/kcodex.py`
- **Examples**: `examples/track_c_knowledge_graph_example.py`
- **Classification Guide**: `docs/K_CODEX_EPISTEMIC_CLASSIFICATION.md`

---

**Migration Status**: ✅ Complete  
**Backwards Compatibility**: ✅ Guaranteed  
**Last Updated**: November 10, 2025  
**Version**: 2.0.0 🚀
