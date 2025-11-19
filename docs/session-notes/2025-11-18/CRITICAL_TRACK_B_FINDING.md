# Critical Finding: Track B Has No External Performance Metric

**Date**: November 19, 2025
**Status**: CRITICAL - Affects All Track B Claims

---

## Executive Summary

**Track B logs contain no external performance metric.**

| Metric | Value | Can Use? |
|--------|-------|----------|
| episode_length | All 180 (constant) | ❌ No variation |
| rewards | Not logged | ❌ Not available |
| K_values | Varies | ❌ Circular (what we optimize) |

**Implication**: We cannot validate that H2 (or any metric) "predicts performance" in Track B because there is no performance to predict.

---

## What the Logs Contain

```
Keys: ['K_values', 'actions', 'metadata', 'observations', 'states']

Episode structure:
- 180 timesteps (constant across all episodes)
- 14-dimensional observations
- 3-dimensional continuous actions
- K_values logged at each step
```

**What's missing**: Any external task reward or success metric.

---

## The Circular Reasoning Problem

### What We Wanted to Show

"H2 predicts performance in Track B"

### What We Can Actually Show

"H2 correlates with K_mean" (r = +0.404)

### Why This is Circular

1. Track B optimizes for K-Index (that's the reward)
2. H2 is a component of the agent's behavior
3. K-Index is also a measure of the agent's behavior
4. Correlating H2 with K is correlating behavior with behavior

**This does not validate that H2 predicts "performance".**

---

## The H2 Finding is Still Real

Despite the circularity issue, we found:

| Condition | H2 (Diversity) |
|-----------|---------------|
| Open Loop | 0.0000 ± 0.0000 |
| Controller | 0.9966 ± 0.0163 |

**p < 0.0001**

**Interpretation**: The controller learned to take diverse actions, while open loop took identical actions every time.

This is consistent with:
- Open loop does `np.zeros_like(raw_action)` (same action always)
- Controller learned varied behavior

But this doesn't tell us if diverse behavior = good behavior.

---

## Where Did "63% Improvement" Come From?

The 63% claim was for "corridor discovery rate" (32% → 52%).

**Question**: How was corridor discovery measured?

**Possibilities**:
1. A separate evaluation metric not in these logs
2. Post-hoc analysis of K-Index trajectories
3. Manual labeling of successful runs

**We need to find** the original code/logs that computed corridor discovery.

---

## Simple K Anomaly

Both open loop and controller have **Simple K = 0.0000**.

This is unexpected and suggests:
- Zero correlation between |obs| and |act| norms
- Or a computation issue

**Investigation needed**: Why is Simple K zero when K_values are ~0.95?

**Possible explanation**: K_values in logs might be computed differently than our Simple K formula.

---

## Impact on Papers

### Paper 1 (Track B+C)

**Original framing**: "63% improvement in corridor discovery with K-index feedback"

**Problem**: We cannot validate this claim because:
1. No external performance metric in logs
2. "Corridor discovery" definition unknown
3. Correlating H2 with K is circular

**Options**:
1. Find original corridor discovery computation
2. Re-run Track B with external metrics
3. Acknowledge limitation and drop performance claims

### What We CAN Claim

"Controller learned diverse actions (H2: 0.0 → 0.99) compared to open loop baseline that produced constant actions."

**What we CANNOT claim**:
"Diverse actions improve performance" (no performance metric)

---

## Required Actions

### Immediate

1. **Find corridor discovery code**: How was 63% computed?
2. **Investigate K_values**: How does K_mean relate to our Simple K?
3. **Check Track E/F logs**: Do they have external performance metrics?

### For Paper 1

Either:
- Find external performance metric
- OR acknowledge limitation
- OR reframe around diversity learning (not performance prediction)

---

## Honest Assessment

### What Track B Shows

1. Controllers learn diverse actions (H2: 0.0 → 0.99) ✅
2. Open loop does nothing (zero action) ✅
3. Controllers achieve higher K_mean (but this is circular)

### What Track B Does NOT Show

1. ❌ That diversity improves performance
2. ❌ That H2 predicts anything external
3. ❌ What "corridor discovery" means or how it was measured

---

## Conclusion

**Track B is fundamentally limited as a validation environment** because it has no external performance metric.

The research may have value in showing that:
- Controllers can learn diverse behavior
- K-Index optimization produces different behavior than open loop

But the claim that this improves "performance" cannot be validated from these logs.

**Next step**: Find Track E/F logs and check if they have external rewards.

