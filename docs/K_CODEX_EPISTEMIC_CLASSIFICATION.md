# K-Codex Epistemic Classification Guide

**Version**: 2.0.0  
**Date**: November 10, 2025  
**Status**: Production Ready  
**Based on**: Mycelix Epistemic Charter v2.0

---

## Table of Contents

1. [Overview](#overview)
2. [The Three-Dimensional Epistemic Cube](#the-three-dimensional-epistemic-cube)
3. [Kosmic Lab Defaults](#kosmic-lab-defaults)
4. [Classification Examples](#classification-examples)
5. [Knowledge Graph Relationships](#knowledge-graph-relationships)
6. [Integration with Mycelix](#integration-with-mycelix)
7. [Practical Usage](#practical-usage)

---

## Overview

K-Codex v2.0 integrates the **Mycelix Epistemic Charter v2.0**, a revolutionary three-dimensional framework for classifying knowledge claims. This integration provides:

- **Enhanced Reproducibility**: E4 (Publicly Reproducible) certification
- **Clear Authority**: N-Axis tracks who accepts the claim
- **Smart Archival**: M-Axis controls permanence and pruning
- **Knowledge Graphs**: Track scientific evolution through related_claims
- **Dispute Resolution**: Built-in framework for challenges
- **Decentralized Science**: Ready for DHT integration

### Why This Matters

Traditional reproducibility focuses only on "can you recreate this?" The Epistemic Cube adds:
- **HOW verifiable** is it? (E-Axis: E0-E4)
- **WHO accepts** it? (N-Axis: N0-N3)
- **HOW LONG** should it be kept? (M-Axis: M0-M3)

This enables decentralized science where experiments from different labs can be:
- **Classified consistently** (same standards globally)
- **Queried efficiently** (find all E4 experiments)
- **Linked meaningfully** (track v1 → v2 → v3 evolution)
- **Preserved appropriately** (M3 forever, M1 pruned)

---

## The Three-Dimensional Epistemic Cube

Every K-Codex is classified along three independent axes:

### E-Axis: Empirical Verifiability

**Question**: How do we verify this claim is true?

| Tier | Name | Description | Verification | Kosmic Lab Usage |
|------|------|-------------|--------------|------------------|
| **E0** | Null | Unverifiable belief/opinion | None possible | Rarely used |
| **E1** | Testimonial | Personal attestation | DID signature | Pilot studies |
| **E2** | Privately Verifiable | Trusted 3rd party review | Audit guild | Proprietary data |
| **E3** | Cryptographically Proven | ZK proofs, signatures | Math/crypto | Future: ZK-STARK proofs |
| **E4** | Publicly Reproducible | Open code + open data | Anyone can verify | **DEFAULT** ✨ |

**Kosmic Lab Standard**: All experiments are **E4** because:
- ✅ Open-source code (GitHub)
- ✅ Open data (OSF)
- ✅ Cryptographic checksums (SHA256SUMS.txt)
- ✅ Complete provenance (Git SHA, config hash, seeds)

### N-Axis: Normative Authority

**Question**: Who agrees this claim is binding?

| Tier | Name | Description | Binding On | Kosmic Lab Usage |
|------|------|-------------|-----------|------------------|
| **N0** | Personal | Self only | Individual | Personal notes |
| **N1** | Communal Consensus | Local community | Research community | **DEFAULT** ✨ |
| **N2** | Network Consensus | Global MIP/DAO vote | All Mycelix participants | Future: after adoption |
| **N3** | Axiomatic | Constitutional/mathematical | Universal truth | Math proofs only |

**Kosmic Lab Standard**: All experiments are **N1** because:
- Accepted by the research community (via peer review)
- Not yet global Mycelix standard (N2)
- Not mathematical axioms (N3)

### M-Axis: Materiality/Permanence

**Question**: How long should this be preserved?

| Tier | Name | Description | Storage | Kosmic Lab Usage |
|------|------|-------------|---------|------------------|
| **M0** | Ephemeral | Transient signals | Discard immediately | System logs |
| **M1** | Temporal | Valid until state changes | Prune after completion | Pilot runs |
| **M2** | Persistent | Archival for auditing | Cold storage (1 year) | Intermediate results |
| **M3** | Foundational | Never prune | Permanent DHT | **DEFAULT** ✨ |

**Kosmic Lab Standard**: All published experiments are **M3** because:
- Part of scientific record (citable)
- Required for meta-analyses
- Support future reproducibility
- Contribute to collective wisdom

---

## Kosmic Lab Defaults

**All Kosmic Lab experiments default to (E4, N1, M3)** - the gold standard for reproducible computational research.

### Default Classification

```python
from core.kcodex import KCodexWriter

writer = KCodexWriter(Path("schemas/k_codex.json"))

# Basic usage - automatically gets (E4, N1, M3)
codex = writer.build_record(
    experiment="track_b_sac",
    params={"learning_rate": 3e-4},
    estimators={"phi": "empirical", "te": {"estimator": "kraskov", "k": 3}},
    metrics={"K": 1.45, "corridor_rate": 0.58}
)

# Default classification:
assert codex["epistemic_tier_e"] == "E4"  # Publicly reproducible
assert codex["epistemic_tier_n"] == "N1"  # Research community
assert codex["epistemic_tier_m"] == "M3"  # Eternal preservation
assert codex["verifiability"]["method"] == "PublicCode"
assert codex["verifiability"]["status"] == "Verified"
```

### When to Override Defaults

**Use E1 (Testimonial)** for:
- Pilot studies without complete documentation
- Work-in-progress experiments
- Personal observations

**Use N0 (Personal)** for:
- Private notes
- Unpublished experiments
- Personal research logs

**Use M2 (Persistent)** for:
- Intermediate analysis results
- Temporary visualizations
- Debug outputs

**Use M1 (Temporal)** for:
- System logs
- Temporary cache files
- Transient state

---

## Classification Examples

### Example 1: Published Track B Experiment

```python
track_b_published = writer.build_record(
    experiment="track_b_sac_seed_225",
    params={...},
    metrics={"K": 1.67, "corridor_rate": 0.58},
    # Classification: (E4, N1, M3) - defaults are perfect!
)
```

**Rationale**:
- **E4**: Open code, open data, SHA256 checksums → anyone can reproduce
- **N1**: Published in PLOS Comp Bio → research community consensus
- **M3**: Part of permanent scientific record → eternal preservation

### Example 2: Pilot Study (Work in Progress)

```python
pilot_study = writer.build_record(
    experiment="pilot_new_rescue_v4",
    params={...},
    metrics={"K": 0.89, "IoU": 0.65},
    epistemic_tier_e="E1",  # Override: testimonial (not fully documented yet)
    epistemic_tier_m="M2",  # Override: persistent (not eternal yet)
    verifiability_status="Pending"  # Not yet verified
)
```

**Rationale**:
- **E1**: Preliminary results, documentation incomplete
- **N1**: Still research community (default OK)
- **M2**: Archive for 1 year, may become M3 after publication

### Example 3: System Debug Log

```python
debug_log = writer.build_record(
    experiment="system_debug_voltage_spike",
    params={...},
    metrics={"spike_count": 47},
    epistemic_tier_e="E1",  # Just our observation
    epistemic_tier_n="N0",  # Personal only
    epistemic_tier_m="M1",  # Prune after bug fixed
)
```

**Rationale**:
- **E1**: Personal observation of bug
- **N0**: Not binding on anyone else
- **M1**: Temporary - delete after fix

---

## Knowledge Graph Relationships

The `related_claims` field enables tracking scientific evolution:

### Relationship Types

| Type | Meaning | Example |
|------|---------|---------|
| **SUPPORTS** | Provides evidence for | Replication study supporting original |
| **REFUTES** | Contradicts | Study finding different results |
| **CLARIFIES** | Explains or elaborates | Follow-up adding context |
| **CONTEXTUALIZES** | Provides background | Historical perspective |
| **SUPERCEDES** | Replaces with improvement | Track C v3 supersedes v2 |
| **REFERENCES** | Builds upon | Track C v2 references v1 bug fix |
| **AUTHENTICATES** | Verifies identity/origin | Signature validation |

### Track C Evolution Example

```python
# Track C v1 (bug discovery)
v1 = writer.build_record(
    experiment="track_c_rescue_v1",
    metrics={"IoU": 0.0},  # Bug!
    # No related_claims - first in series
)

# Track C v2 (bug fixed)
v2 = writer.build_record(
    experiment="track_c_rescue_v2",
    metrics={"IoU": 0.706},
    related_claims=[
        {
            "relationship_type": "REFERENCES",
            "related_claim_id": v1["run_id"],
            "context": "Fixed grid clipping bug from v1"
        }
    ]
)

# Track C v3 (breakthrough)
v3 = writer.build_record(
    experiment="track_c_rescue_v3",
    metrics={"IoU": 0.788},
    related_claims=[
        {
            "relationship_type": "SUPERCEDES",
            "related_claim_id": v2["run_id"],
            "context": "Attractor-based mechanism eliminates interference"
        },
        {
            "relationship_type": "REFERENCES",
            "related_claim_id": v1["run_id"],
            "context": "Maintains v1 bug fix"
        }
    ]
)
```

This creates a queryable knowledge graph:

```
v1 (bug discovery)
 ↓ REFERENCES
v2 (bug fixed, but worse rescue)
 ↓ SUPERCEDES
v3 (attractor breakthrough) ✨
```

---

## Integration with Mycelix

### Current State (Local K-Codex)

K-Codices are currently stored as local JSON files with epistemic classification ready for DHT publication.

### Future State (Mycelix DHT)

When published to Mycelix Decentralized Knowledge Graph:

1. **Immutable Storage**: K-Codex published to Holochain DHT
2. **AgentPubKey**: researcher_agent field populated
3. **Global Queries**: Find all E4 experiments across all labs
4. **Meta-Analyses**: Aggregate results without sharing raw data
5. **Dispute Resolution**: Built-in challenge mechanisms
6. **Federated Learning**: Privacy-preserving collaboration

### Example Future Query

```
Query: "Find all E4 experiments that SUPERCEDE others in bioelectric rescue"
Results:
  - Track C v3 (Kosmic Lab) - Attractor-based mechanism
  - [Other labs' breakthroughs]
  - Ranked by N-Axis authority and community validation
```

---

## Practical Usage

### Basic Usage (Recommended)

```python
from pathlib import Path
from core.kcodex import KCodexWriter

# Initialize with v2.0 schema
writer = KCodexWriter(Path("schemas/k_codex.json"))

# Create K-Codex with smart defaults
codex = writer.build_record(
    experiment="your_experiment",
    params={"param1": 0.5},
    estimators={"phi": "empirical", "te": {...}},
    metrics={"K": 1.23}
)

# Write to filesystem
path = writer.write(codex, Path("logs/codices"))
print(f"K-Codex created: {path}")
```

### Advanced Usage (Custom Classification)

```python
# Custom classification for special cases
codex = writer.build_record(
    experiment="pilot_study",
    params={...},
    metrics={...},
    # Override defaults
    epistemic_tier_e="E1",  # Testimonial (not fully reproducible yet)
    epistemic_tier_m="M2",  # Persistent (archive after 1 year)
    verifiability_status="Pending",
    # Add relationships
    related_claims=[
        {
            "relationship_type": "REFERENCES",
            "related_claim_id": previous_experiment_id,
            "context": "Builds on previous findings"
        }
    ]
)
```

### Batch Processing

```python
# Process multiple experiments with consistent classification
for seed in [100, 200, 300]:
    codex = writer.build_record(
        experiment=f"track_b_sac_seed_{seed}",
        params={"seed": seed},
        metrics=compute_metrics(seed),
        # Defaults (E4, N1, M3) for all
    )
    writer.write(codex, output_dir)
```

---

## Summary

K-Codex v2.0 with Epistemic Charter integration provides:

✅ **Reproducibility**: E4 certification - highest standard  
✅ **Authority**: N1 classification - research community consensus  
✅ **Permanence**: M3 designation - eternal preservation  
✅ **Evolution**: Knowledge graph tracks v1 → v2 → v3  
✅ **Discovery**: Query across experiments and labs  
✅ **Integration**: Ready for Mycelix DHT publication

**Default (E4, N1, M3) is perfect for 99% of Kosmic Lab experiments!**

---

## References

- Mycelix Epistemic Charter v2.0: https://github.com/Luminous-Dynamics/Mycelix-Core/blob/main/docs/architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md
- K-Codex Schema: `schemas/k_codex.json`
- Example Code: `examples/track_c_knowledge_graph_example.py`
- Implementation: `core/kcodex.py`

---

**Last Updated**: November 10, 2025  
**Version**: 2.0.0  
**Status**: Production Ready 🚀
