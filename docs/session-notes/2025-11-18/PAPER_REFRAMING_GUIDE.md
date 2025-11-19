# Paper Reframing Guide: From Performance Claims to Validated Findings

**Date**: November 19, 2025
**Purpose**: Concrete guidance for reframing each paper around validated findings
**Timeline**: 2-4 weeks

---

## Executive Summary

All papers currently claim K-Index predicts or improves performance. This is not supported by data. This guide shows exactly how to reframe each paper around what we *can* validate.

---

## Paper 1: Track B+C (Coherence-Guided Control)

### Current Claims (Cannot Publish)

> "K-Index feedback improves controller performance by 63%"
> "Coherence-guided learning achieves better task success"

### Why Invalid

- **No external performance metric exists** for Track B
- Episode length is constant (180)
- Rewards were not logged
- The 63% was corridor discovery rate, not performance

### Reframed Title

**Before**: "Coherence-Guided Learning Improves Controller Performance"
**After**: "Learning Action Diversity Through Coherence Feedback"

### Reframed Abstract

**Before**:
> "We introduce K-Index as a performance predictor for reinforcement learning controllers. Training with K-Index feedback improves task performance by 63%..."

**After**:
> "We introduce a metric for action diversity (H2) that tracks behavioral complexity during controller training. Controllers trained with coherence feedback develop significantly more diverse action repertoires (H2: 0.0 → 0.99, p < 0.001) compared to open-loop baselines. While the relationship between action diversity and task performance requires further investigation, our results demonstrate that coherence feedback effectively shapes controller behavior toward more exploratory policies..."

### Key Claims to Keep

| Claim | Evidence | Status |
|-------|----------|--------|
| Controllers learn diverse actions | H2: 0.0 → 0.99 | ✅ Valid |
| K-Index increases during training | Observed in logs | ✅ Valid |
| Coherence feedback changes behavior | Significant difference from baseline | ✅ Valid |

### Key Claims to Remove

| Claim | Why Remove |
|-------|------------|
| "63% performance improvement" | No performance metric exists |
| "K-Index predicts performance" | Never tested |
| "Better task success" | Not measured |

### Specific Text Changes

**Section 4.2 Results**:

Delete:
> "K-Index feedback improved performance by 63% compared to baseline"

Replace with:
> "K-Index feedback produced controllers with significantly higher action diversity (H2 = 0.99 ± 0.01) compared to open-loop baselines (H2 = 0.00). The relationship between this increased diversity and task performance is a direction for future work."

**Section 5 Discussion**:

Add limitation:
> "A key limitation of this study is the absence of an external performance metric. While we demonstrate that coherence feedback shapes behavior toward greater action diversity, we did not measure whether this translates to improved task performance. Future work should incorporate explicit performance metrics to validate this relationship."

---

## Paper 4: Track E (Developmental Learning)

### Current Claims (Cannot Publish)

> "K-Index grows during learning, indicating improved coherence"
> "Developmental learning improves K-Index by X%"

### Why Invalid

- K-Index does correlate with training (r = +0.59 to +0.77)
- But K-Index does NOT correlate with rewards (r = -0.01)
- Rewards themselves are essentially random noise (no learning trend)

### Reframed Title

**Before**: "Developmental Learning Improves Controller Coherence"
**After**: "K-Index as a Training Progress Metric in Developmental Learning"

### Reframed Abstract

**Before**:
> "We show that developmental learning approaches improve controller coherence as measured by K-Index. This coherence improvement correlates with better task performance..."

**After**:
> "We investigate K-Index as a metric for tracking training progress in developmental learning. Across all conditions, K-Index increases significantly during training (r = +0.59 to +0.77, p < 0.001), suggesting it captures aspects of behavioral change during learning. However, K-Index does not correlate with external task rewards (r = -0.01, n.s.), indicating it measures training dynamics rather than performance. We discuss K-Index as a potential early indicator of learning, independent of task-specific outcomes..."

### Key Claims to Keep

| Claim | Evidence | Status |
|-------|----------|--------|
| K increases during training | r = +0.59 to +0.77, p < 0.001 | ✅ Valid |
| Different conditions show different K trajectories | Observed in data | ✅ Valid |

### Key Claims to Remove

| Claim | Why Remove |
|-------|------------|
| "K predicts performance" | r = -0.01 with rewards |
| "Higher K = better controller" | Not supported |
| Any performance comparison | Rewards are noise |

### Specific Text Changes

**Section 3.1 Methods**:

Add:
> "Note: The Track E environment uses a state that evolves as `0.9 * state + 0.1 * noise`, making optimal behavior difficult to learn. The rewards computed from this state show no significant learning trend (r = +0.20, n.s.) and low autocorrelation (-0.08), suggesting the task may not be fully learnable. We therefore interpret K-Index as a training dynamics metric rather than a performance predictor."

**Section 4 Results**:

Delete:
> "K-Index improvement indicates better task coherence"

Replace with:
> "K-Index increases during training across all conditions, but does not correlate with task rewards (r = -0.01). This suggests K-Index captures training dynamics (e.g., policy stabilization, action repertoire development) rather than task performance."

**Section 5 Discussion**:

Add:
> "Our results suggest K-Index is best understood as a training progress metric rather than a performance predictor. It may be useful for: (1) detecting when learning has plateaued, (2) comparing learning trajectories across conditions, (3) identifying anomalies in training. However, researchers should not assume higher K implies better task performance without explicit validation."

---

## Paper 5: Track F (Adversarial Robustness)

### Current Claims (Cannot Publish)

> "Adversarial training improves K-Index by 85%"
> "K-Index measures controller robustness"

### Why Invalid

- Same fundamental issue: K doesn't predict performance
- No evidence that K relates to robustness

### Reframed Title

**Before**: "Adversarial Training Improves Controller Coherence"
**After**: "Behavioral Changes Under Adversarial Attack: A K-Index Analysis"

### Reframed Abstract

**Before**:
> "We show adversarial training increases K-Index by 85%, indicating more robust controllers..."

**After**:
> "We analyze how adversarial attacks affect controller behavior as measured by K-Index. Adversarial perturbations significantly reduce K-Index (Δ = -X%, p < 0.001), while adversarial training produces controllers with higher K values under attack. However, the relationship between K-Index and actual robustness (task performance under perturbation) was not measured. We discuss K-Index as a behavioral signature that changes under adversarial conditions, requiring further work to establish its relationship to meaningful robustness metrics..."

### Specific Changes

Same pattern as above:
- Remove all "robustness improvement" claims
- Reframe as "behavioral signature" or "behavioral change metric"
- Acknowledge need to validate against actual robustness measures

---

## Paper 3: Track D (Topology Effects)

### Current Claims (Cannot Publish)

> "Ring topology achieves 9% higher K-Index"
> "Topology affects controller coherence"

### Why Invalid

- K doesn't predict performance
- 9% difference may not be meaningful
- No validation that topology matters for actual outcomes

### Recommendation: Withdraw or Major Revision

This paper has the weakest case because:
1. It makes a specific architectural claim (ring > other)
2. This claim is only supported by K-Index
3. K-Index doesn't predict performance

**Options**:
1. **Withdraw** until performance metrics available
2. **Major revision** as exploratory/descriptive study

If revising, reframe as:
> "Preliminary Investigation of Topology Effects on Behavioral Diversity"
> "We observe that ring topology produces higher K-Index values, but the practical significance of this finding is unclear. Future work should investigate whether topology differences translate to measurable performance differences."

---

## Paper 2: Track A (Baseline Studies)

### Status

Not detailed in validation, but same issues apply. Should follow same reframing pattern.

---

## General Reframing Principles

### Language Changes

| Before | After |
|--------|-------|
| "improves performance" | "changes behavior" |
| "predicts success" | "tracks training progress" |
| "coherence indicates quality" | "coherence measures behavioral patterns" |
| "63% improvement" | "significant behavioral difference" |
| "better controller" | "more diverse actions" |

### Required Additions (All Papers)

#### Limitations Section

```markdown
### Limitations

1. **No External Performance Validation**: K-Index was not validated against task-specific performance metrics. The relationship between K-Index and task success requires future investigation.

2. **CartPole Generalization Unknown**: The finding that action diversity (H2) predicts performance in CartPole has not been validated in the environments used in this study.

3. **Metric Interpretation**: K-Index should be interpreted as a behavioral complexity/training dynamics metric, not a performance predictor.
```

#### Future Work Section

```markdown
### Future Work

1. Add external performance metrics to all experimental tracks
2. Validate the H2-performance relationship in non-CartPole environments
3. Investigate what K-Index actually measures (behavioral complexity? policy stability?)
4. Determine which contexts, if any, show K-performance correlation
```

---

## Checklist Before Submission

For each paper:

- [ ] Title reflects validated findings, not performance claims
- [ ] Abstract makes no unvalidated performance claims
- [ ] All "improvement" claims are about behavior, not performance
- [ ] Limitations section acknowledges lack of performance validation
- [ ] Future work section identifies validation needs
- [ ] All specific percentages tied to behavioral metrics, not performance
- [ ] No claim that K predicts or improves performance

---

## Timeline

| Week | Activities |
|------|------------|
| 1 | Rewrite abstracts and titles for Papers 1, 4, 5 |
| 2 | Revise results sections with behavioral framing |
| 2-3 | Add limitations and future work sections |
| 3 | Internal review of reframed papers |
| 4 | Final polish and submission preparation |

---

## Conclusion

The K-Index research found something real: controllers develop more diverse actions during training, and K-Index tracks this development. This is publishable and valuable.

What we cannot claim: that this diversity predicts performance. That claim requires validation we don't have.

By reframing around validated findings, we:
1. Maintain scientific integrity
2. Contribute real knowledge about behavioral metrics
3. Set up future work with clear validation goals
4. Avoid publishing claims we cannot support

**The honest story is still a good story.** Controllers learn diverse actions, and we can measure that. That's worth publishing.

---

*"Science is not about being right. It's about being honest about what we know and don't know."*

