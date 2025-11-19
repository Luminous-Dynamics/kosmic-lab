# Final Paper Revision Plan

**Date**: November 19, 2025
**Status**: Ready for Implementation
**Priority**: Papers 1 & 4 first, then Paper 5

---

## Executive Summary

After comprehensive revalidation of all tracks, we have clear guidance for each paper:

| Paper | Track | Original Status | After Revalidation | Time Estimate |
|-------|-------|-----------------|-------------------|---------------|
| **Paper 4** | E | Salvageable | ✅ **READY** | 3-5 days |
| **Paper 1** | B+C | Investigate | ✅ **SALVAGEABLE** | 1 week |
| **Paper 5** | F | Reframe | ⚠️ **REFRAME** | 1 week |
| **Paper 3** | D | Test | ❌ **WITHDRAW** | - |
| **Paper 2** | A | Hold | ⏸️ **HOLD** | TBD |

---

## Paper 4: Developmental Learning (Track E)

### Status: ✅ READY TO UPDATE

### Original Claim
"K-Index grows from 0.3 to 1.357 during learning - evidence of emergent consciousness"

### Revalidation Result
- Simple K: r = -0.33 with episode (decreases!)
- **Full K: r = +0.52 with episode (increases!)** ✅
- H2: remains high throughout

### Revision Strategy

**Keep**: Core thesis that coherence emerges through learning

**Change**:
- Replace all Simple K values with Full K
- Update consciousness threshold interpretation
- Add H2 analysis

**Add**:
- Honest discussion of metric discovery
- Comparison of Simple K vs Full K
- Why Full K is the correct measure

### Specific Edits Needed

1. **Abstract**: Change "K-Index grows to 1.357" → "Full K-Index shows r=+0.52 correlation with learning"

2. **Methods**: Add Full K-Index formula and computation details

3. **Results**:
   - Replace Table 1 with Full K values
   - Add Figure showing Full K vs episode number
   - Show H2 (Diversity) trajectory

4. **Discussion**: Add section on metric validation

### Timeline: 3-5 days

---

## Paper 1: Coherence-Guided Control (Track B+C)

### Status: ✅ SALVAGEABLE (better than expected!)

### Original Claim
"63% improvement with K-index feedback"

### Revalidation Results

**From Log Analysis (56 episodes)**:
- Simple K: +6.1% (not 63%)
- H2 (Diversity): 0.0 → 0.84 (**massive improvement!**)
- Approx Full K: **+33.6%**

**From CartPole test**:
- Simple K vs Performance: r = -0.814 (anti-correlates!)
- Full K vs Performance: r = +0.729 (positively correlates!)

### Key Insight
The SAC controller learned to act **diversely** (H2=0.84) vs open-loop (H2=0.0). This IS coherent behavior and correlates with performance.

### Revision Strategy

**Keep**: Core approach of coherence-guided control

**Change**:
- Drop "63% Simple K improvement" claim
- Reframe around Full K (+33.6%) and H2 improvement
- Emphasize that controller learned diversity

**Add**:
- Full K analysis of existing logs
- H2 as primary success metric
- Correlation analysis showing Full K works

### Specific Edits Needed

1. **Title**: Consider "Diversity-Guided Control: Learning Coherent Behavior Through Action Entropy"

2. **Abstract**:
   - Remove "63% improvement"
   - Add "33.6% Full K improvement driven by 0.84 H2 increase"

3. **Methods**:
   - Add Full K-Index computation
   - Explain H2 (Diversity) metric

4. **Results**:
   - New Table: Open Loop vs Controller (Simple K, H2, Full K)
   - Figure: H2 trajectory showing controller learns diversity
   - Correlation analysis: Full K vs performance

5. **Discussion**:
   - Why diversity (H2) matters
   - Comparison with Simple K approach
   - Future: Optimize directly for Full K

### Timeline: 1 week

---

## Paper 5: Adversarial Perturbations (Track F)

### Status: ⚠️ REFRAME AS METHODOLOGICAL CONTRIBUTION

### Original Claim
"Adversarial perturbations enhance K-Index by 85%"

### Revalidation Result

At ε=0.05:
- Simple K: **+147%** (appeared to enhance)
- Full K: **-7%** (correctly shows degradation)
- Performance: **-52%** (confirms Full K is right)

### The Story

This is actually a valuable finding! The "enhancement" was an artifact of Simple K measuring rigidity. We discovered this through falsification.

### Revision Strategy

**Complete reframe** as methodological paper:

**New Title**: "When Metrics Mislead: How Adversarial Attacks Reveal K-Index Measures Rigidity, Not Coherence"

**New Contribution**:
1. Documents the paradigm shift discovery
2. Shows how adversarial attacks expose metric flaws
3. Introduces Full K as correct alternative
4. Warns the field about correlation-based metrics

### Specific Edits Needed

1. **New Abstract**:
   > We report a methodological finding with implications for consciousness metrics in AI systems. Initial experiments showed adversarial perturbations "enhanced" K-Index by 85%. Investigation revealed this was an artifact: the correlation-based Simple K-Index measures response rigidity, not coherence. We introduce the Full 7-Harmony K-Index which correctly shows degradation under attack. This work serves as a cautionary tale for metric validation in AI consciousness research.

2. **Introduction**: Frame as discovery journey, not failure

3. **Methods**: Present both Simple K and Full K

4. **Results**:
   - Table showing opposite signs (Simple K +147%, Full K -7%, Performance -52%)
   - Figure contrasting the two metrics across ε values

5. **Discussion**:
   - What Simple K actually measures (rigidity)
   - Why Full K is correct (H2 predicts performance)
   - Implications for the field

### Timeline: 1 week

---

## Paper 3: Topology of Collective Consciousness (Track D)

### Status: ❌ WITHDRAW OR COMPLETE REFRAME

### Original Claim
"Ring topology has 9% better K-Index"

### Revalidation Result

| Metric | Best Topology | Ring Rank |
|--------|---------------|-----------|
| Performance | mesh | #2 |
| Simple K | isolated | #3 |
| Full K | mesh | #4 |

Ring is NOT the best by ANY metric.

### Decision Needed

**Option A: Withdraw**
- The ring superiority claim is not supported
- Cleanest approach

**Option B: Complete Reframe**
- "Mesh Topology Optimizes Collective Coherence"
- Would need new experiments to strengthen claim
- Risky - mesh advantage is small

### Recommendation: WITHDRAW

The finding is not robust enough for publication. Better to withdraw than publish weak results.

---

## Paper 2: Coherence Corridors (Track A)

### Status: ⏸️ HOLD

### Original Plan
Document "corridors" - regions of parameter space with high K-Index

### Issue
These corridors were found with Simple K. They may be "rigidity corridors" not coherence corridors.

### Required Validation

Before writing:
1. Re-run corridor discovery with Full K
2. Check if Simple K corridors overlap with Full K corridors
3. If they diverge, concept needs rethinking

### Timeline: 2-3 days validation, then 1 week writing (if valid)

---

## Implementation Order

### Week 1: Foundation + Paper 4
1. **Day 1**: Install Full K in codebase
   ```bash
   sudo cp docs/session-notes/2025-11-18/full_k_index_production.py fre/metrics/full_k_index.py
   ```

2. **Days 2-5**: Update Paper 4
   - Re-run Track E with Full K
   - Update all figures and tables
   - Revise text

### Week 2: Paper 1
1. **Days 1-2**: Re-analyze Track B logs with complete Full K
2. **Days 3-5**: Revise paper around H2 and Full K improvements
3. **Days 6-7**: Internal review

### Week 3: Paper 5
1. **Days 1-3**: Complete reframe as methodological contribution
2. **Days 4-5**: New figures showing metric comparison
3. **Days 6-7**: Internal review

### Week 4: Finalization
1. Decide on Paper 3 (recommend withdraw)
2. Validate Paper 2 corridors
3. Final review of Papers 1, 4, 5
4. Prepare for submission

---

## Key Metrics Reference

### Use These (Positively Correlate with Performance)

```python
# H2 (Diversity) - BEST single predictor
h2 = entropy(action_distribution) / log(n_actions)
# r = +0.46 with performance

# Full 7-Harmony K-Index
full_k = (H1 + H2 + H3 + H4 + H5 + H6 + H7) / 7
# r = +0.26 with performance
```

### Do NOT Use (Anti-Correlates)

```python
# Simple K-Index - DEPRECATED
simple_k = 2 * abs(correlation(obs_norms, act_norms))
# r = -0.41 with performance (WRONG SIGN!)
```

---

## Summary

**The research program is viable.**

We discovered we were measuring the wrong thing (rigidity vs coherence), but the core insights are valid when using Full K:

- Coherence emerges through learning (Paper 4) ✅
- Controllers can learn coherent/diverse behavior (Paper 1) ✅
- Simple K is a cautionary tale for the field (Paper 5) ✅

**Immediate actions**:
1. Install Full K in codebase
2. Update Paper 4 (3-5 days)
3. Update Paper 1 (1 week)
4. Reframe Paper 5 (1 week)
5. Withdraw Paper 3
6. Validate Paper 2 corridors

---

*"The measure of a research program is not whether the first hypothesis is correct, but whether it can find the truth when the first hypothesis is wrong."*

**Status**: Ready for implementation
**Confidence**: High
**Timeline**: 4 weeks to submission-ready

