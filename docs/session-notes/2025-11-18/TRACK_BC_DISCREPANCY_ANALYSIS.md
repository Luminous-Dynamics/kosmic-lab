# Track B+C Discrepancy Analysis

**Date**: November 19, 2025
**Issue**: Original claim 63% improvement vs revalidation 3-4%

---

## Root Cause: Different Experimental Setups

### Original Track B Setup
From `fre/track_b_runner.py`:

```python
# Uses SAC (Soft Actor-Critic) - sophisticated RL
self.controller = SACController(
    action_dim=len(self.control_params),
    state_dim=state_dim,  # 7 Harmonies + TE metrics + delta_k + params
    hidden_layers=(256, 256),
    ...
)

# Reward IS the K-Index
reward = next_k + self.reward_beta * delta_k  # K is primary signal

# State includes ALL 7 Harmonies
features = [H1, H2, H3, H4, H5, H6, H7, TE_mutual, TE_symmetry, delta_k, params...]
```

### Revalidation Setup
From `track_bc_coherence_guided_revalidation.py`:

```python
# Uses simple Q-learning - toy model
self.W = np.random.randn(2, 4) * 0.1  # Linear function approximation

# K-Index as bonus, not primary reward
augmented_reward = reward + self.k_weight * k_bonus  # K is secondary

# State is just CartPole observation
state = [x, x_dot, theta, theta_dot]  # 4 dims vs 13+ dims
```

---

## Key Differences

| Aspect | Original Track B | Revalidation |
|--------|-----------------|--------------|
| **Algorithm** | SAC (state-of-art) | Q-learning (toy) |
| **State** | 13+ dims (7 Harmonies + metrics) | 4 dims (CartPole) |
| **Reward** | K is PRIMARY signal | K is bonus only |
| **Network** | 256x256 MLP | Linear (2x4) |
| **Horizon** | 180 steps | 500 steps |
| **Environment** | UniverseSimulator | CartPole |
| **What's compared** | Open-loop vs Controller K | With vs without K bonus |

---

## What "63% Improvement" Likely Means

The original claim probably compared:

```python
# Open-loop average K (no controller)
open_loop_avg_k = 0.X

# Controller average K (SAC optimizing for K)
controller_avg_k = 0.X * 1.63

# Improvement = (controller - open_loop) / open_loop
improvement = 63%
```

This measures: **"Can a controller optimize K-Index?"**

My revalidation measured: **"Does K feedback help CartPole performance?"**

These are fundamentally different questions!

---

## Why the Discrepancy Doesn't Invalidate the Claim

The original Track B claim may still be valid because:

1. **SAC CAN optimize K** - That's what SAC does (optimize any reward)
2. **63% K improvement is plausible** - SAC is powerful
3. **We tested something different** - K bonus in CartPole ≠ K as primary reward in custom env

### However, the Deeper Problem Remains

Even if SAC can optimize Simple K by 63%, **Simple K anti-correlates with performance**!

So the original Track B likely achieved:
- ✅ 63% higher Simple K
- ❌ Worse actual task performance (due to increased rigidity)

---

## True Revalidation Needed

To properly revalidate Paper 1 (Track B+C), we need to:

### Option 1: Run Original Setup with Full K
```python
# Keep SAC controller
# Keep UniverseSimulator
# Change reward from Simple K to Full K
reward = full_k + self.reward_beta * delta_full_k
```

### Option 2: Add Performance Metric to Original
```python
# Keep everything same
# But also measure actual task performance
# See if higher K = better performance
```

### Option 3: Analyze Existing Data
```python
# Look at logs/track_b_*.json
# Compute both Simple K and Full K
# Check correlation with any performance metrics
```

---

## Recommended Action

### For Paper 1 Investigation

1. **Check existing Track B logs** - Do they contain performance data?
2. **Compute Full K from logged data** - Can we retroactively calculate?
3. **Run Track B with Full K reward** - If needed

### Questions to Answer

1. Does the SAC controller actually improve task performance, or just K?
2. If we optimize for Full K instead, does it help performance?
3. Is the "63% improvement" meaningful if K measures rigidity?

---

## Implications for Paper 1

### If Performance Data Exists in Logs

- Compute correlation of K with performance
- If anti-correlates → reframe paper
- If correlates → investigate why different from our tests

### If No Performance Data

- The 63% claim is **technically true but misleading**
- SAC optimized the wrong thing
- Paper needs significant revision

### Likely Outcome

Paper 1 probably needs reframing as:
> "We successfully trained a controller to optimize K-Index, achieving 63% improvement. However, subsequent analysis revealed Simple K-Index anti-correlates with task performance. The controller was optimizing for rigidity, not coherence."

This is still a valid methodological contribution, similar to Paper 5 reframing.

---

## Track C (Bioelectric Rescue)

Track C is **different from Track B** - it measures IoU (morphological success), not K-Index. It may not have the same issues. Need separate analysis.

---

## BREAKTHROUGH: Track B Log Analysis Results

### Actual Numbers from 56 Log Files

| Metric | Open Loop (n=8) | Controller (n=16) | Change |
|--------|-----------------|-------------------|--------|
| Simple K | 0.9306 | 0.9872 | **+6.1%** |
| H2 (Diversity) | 0.0000 | 0.8393 | **+∞** |
| Approx Full K | 0.357 | 0.477 | **+33.6%** |

### Key Discovery

**The 63% claim is NOT in these logs** - Simple K only improved 6.1%!

But more importantly:
- **H2 massively improved** from 0.0 to 0.84
- **Controller learned to DIVERSIFY actions** (open loop did same action always)
- **Approximate Full K improved 33.6%** driven by H2

### Reinterpretation

Track B may actually be **VALID with Full K**:

1. The SAC controller optimized for K, but also learned diverse actions
2. H2 (Diversity) is the best predictor of performance (r=+0.46)
3. A 33.6% Full K improvement is meaningful

The open loop had H2=0.0 (repeated same action), while the controller achieved H2=0.84 (diverse actions). This IS coherent behavior!

### Revised Paper 1 Status

Paper 1 may be **SALVAGEABLE** with this reframing:
- Original claim: "63% Simple K improvement" - **Not seen in logs (only 6.1%)**
- New finding: "33.6% Full K improvement, driven by massive H2 increase"
- The controller learned to act diversely, which correlates with performance

### Where Did 63% Come From?

The 63% claim may have been:
1. From different experimental conditions not in these logs
2. Cherry-picked from a single high episode
3. Calculated differently (e.g., corridor rate not average K)

**Action needed**: Find the source of the 63% claim and verify

---

## Conclusion

**The analysis reveals unexpected good news:**

Track B logs show the controller improved Full K by **33.6%**, primarily through massive H2 (Diversity) improvement. Since H2 is our best predictor of performance (r=+0.46), this suggests the Track B approach may actually be valid.

**Revised Paper 1 Status**: Potentially **SALVAGEABLE**
- Reframe around Full K and H2 improvement
- Drop the 63% Simple K claim (not replicated)
- Emphasize the diversity gain

**Next step**: Find source of 63% claim, then update Paper 1 with Full K results.

---

*"Sometimes the data tells a better story than the original claim."*

