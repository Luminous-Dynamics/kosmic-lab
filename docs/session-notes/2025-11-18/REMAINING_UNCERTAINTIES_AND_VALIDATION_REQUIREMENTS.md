# Remaining Uncertainties and Validation Requirements

**Date**: November 19, 2025
**Purpose**: Honest accounting of what we still don't know

---

## Critical Limitation: All Tests Were in CartPole

Every validation test in this session used CartPole, not the original environments.

| Original Track | Environment | Our Test |
|----------------|-------------|----------|
| Track B | UniverseSimulator | CartPole |
| Track C | Bioelectric simulation | CartPole |
| Track D | Multi-agent collective | CartPole |
| Track E | Developmental learning | CartPole |
| Track F | Adversarial robustness | CartPole |

**CartPole has:**
- 4-dimensional state (vs 14-dim in UniverseSimulator)
- 2 discrete actions (vs continuous)
- Simple pole-balancing task (vs cosmic simulation)
- Clear performance metric (episode length)

**The original environments have:**
- Higher dimensionality
- Continuous actions
- Different dynamics
- K-Index as the "performance" metric (circular!)

### Implication

**Our findings may not transfer to the original environments.**

The H2 correlation (r = +0.71) in CartPole might be:
- Higher in original environments
- Lower in original environments
- Different in sign

We don't know until we test.

---

## Remaining Uncertainties

### 1. Does H2 Predict Performance in UniverseSimulator?

**We showed**: H2 predicts CartPole performance (r = +0.71)

**We don't know**: Does this hold in UniverseSimulator where:
- The "performance" is K-Index itself
- The state space is 14-dimensional
- Actions are continuous

**Risk**: The correlation could vanish or reverse.

### 2. Why Did Simple K Show r = -0.81 Earlier?

**We found**: Simple K is positive in 95% of tests

**But**: One earlier test showed r = -0.81 (strongly negative)

**Possible explanations**:
- Different experimental setup (we didn't fully investigate)
- Random variation (but very extreme)
- Bug in earlier code
- Different K computation

**We should**: Re-run the exact earlier test to understand the discrepancy

### 3. What Does "Performance" Mean in Track B?

Track B optimizes for K-Index. "Success" = high K.

**Question**: What external metric (besides K) measures performance?

**Possibilities**:
- Corridor discovery rate? (but this is defined by K regions)
- Stability? (not clearly defined)
- Nothing? (K is the only metric)

**If K is the only metric**: Correlating K with "performance" is circular.

### 4. Do the Anti-Correlating Harmonies Anti-Correlate in Original Environments?

**We found**: H3, H4, H6 anti-correlate with performance in CartPole

**We don't know**: Is this true in:
- UniverseSimulator
- Bioelectric simulation
- Multi-agent collective

**If not**: Full K might work in original environments but not CartPole

### 5. Does H2 Actually Cause Good Performance?

**We showed**: Correlation between H2 and performance

**We didn't show**: Causation

**Alternative explanations**:
- Both are caused by training progress
- Both are caused by exploration rate
- Correlation is spurious

**To show causation**: Need intervention experiments (e.g., artificially increase/decrease H2)

---

## Validation Requirements Before Publication

### Tier 1: Essential (Must Do)

These must be done before any paper submission.

#### 1.1 Validate H2 in Original Environments

```bash
# For each track, compute H2 and correlate with performance
python fre/track_b_runner.py --compute-h2 --correlate
python fre/track_e_runner.py --compute-h2 --correlate
python fre/track_f_runner.py --compute-h2 --correlate
```

**Expected result**: H2 correlates positively (r > 0.3)

**If it doesn't**: Our finding doesn't generalize

#### 1.2 Define External Performance Metric for Track B

Currently: "Performance" = K-Index (circular)

**Need**: External metric that is not K-based
- Stability of trajectories?
- Predictability?
- Energy efficiency?

**If no external metric exists**: Acknowledge this limitation explicitly

#### 1.3 Investigate r = -0.81 Discrepancy

Re-run the exact test that showed Simple K r = -0.81

```bash
# Find and re-run exact earlier test
python docs/session-notes/2025-11-18/track_bc_coherence_guided_revalidation.py
```

**Understand**: Why did it differ from subsequent tests?

### Tier 2: Important (Should Do)

These strengthen the claims significantly.

#### 2.1 Test H2 Causation

**Experiment**: Artificially constrain H2 and measure performance

```python
# Force agent to have low H2 (repeat same action)
# vs high H2 (diverse actions)
# Compare performance
```

**If causation confirmed**: Can claim "diversity improves performance"

**If not**: Can only claim "correlation, possibly spurious"

#### 2.2 Validate Other Harmonies in Original Environments

Do H3, H4, H6 anti-correlate in UniverseSimulator too?

If they don't, Full K might be valid there (just not in CartPole).

#### 2.3 Run with Meaningful Baselines

Current baseline: Zero action (degenerate)

**Better baselines**:
- Random policy
- Simple heuristic (e.g., move toward center)
- Previously trained agent

### Tier 3: Nice to Have

These would strengthen the paper but aren't essential.

#### 3.1 Multiple Random Seeds

Run all experiments with 5+ random seeds to quantify variance.

#### 3.2 Different RL Algorithms

Test with DQN, PPO, SAC to show H2 finding generalizes.

#### 3.3 Different Environments

Test in Atari, MuJoCo to show H2 finding is general.

---

## Risk Assessment

### Risk: H2 Finding Doesn't Generalize

**Probability**: Medium (30-50%)

**Impact**: Cannot publish main claim

**Mitigation**: Run Tier 1.1 validation immediately

### Risk: No External Performance Metric

**Probability**: High (60-80%)

**Impact**: Track B claims are circular

**Mitigation**: Be honest about limitation; focus on Tracks E, F

### Risk: r = -0.81 Was Valid

**Probability**: Low (10-20%)

**Impact**: Simple K is context-dependent, not unreliable

**Mitigation**: Run Tier 1.3 to understand

### Risk: Causation Not Established

**Probability**: High (if we don't test it)

**Impact**: Weaker claims

**Mitigation**: Run Tier 2.1 or acknowledge limitation

---

## Recommended Validation Schedule

### Week 1: Essential Validations

| Day | Task | Time |
|-----|------|------|
| 1 | Run H2 validation in Track B (1.1) | 4 hours |
| 2 | Run H2 validation in Tracks E, F (1.1) | 4 hours |
| 3 | Investigate r = -0.81 discrepancy (1.3) | 3 hours |
| 4 | Define external performance metric (1.2) | 4 hours |
| 5 | Document findings, update papers | 4 hours |

### Week 2: Important Validations (if Week 1 passes)

| Day | Task | Time |
|-----|------|------|
| 1-2 | Test H2 causation (2.1) | 6 hours |
| 3 | Validate harmonies in original envs (2.2) | 4 hours |
| 4-5 | Run with meaningful baselines (2.3) | 6 hours |

### Week 3: Paper Revision

Based on validation results, revise papers with honest claims.

---

## Decision Points

### After Week 1 Validation

**If H2 correlates in original environments (r > 0.3)**:
→ Proceed with papers focused on H2/diversity

**If H2 doesn't correlate (r < 0.1)**:
→ The CartPole finding doesn't generalize
→ Need to investigate why
→ Papers may not be publishable

**If no external performance metric found**:
→ Acknowledge Track B claims are circular
→ Focus papers on Tracks E, F where performance is defined

### After Week 2 Validation

**If causation established**:
→ Can claim "diversity improves performance"

**If only correlation**:
→ Can claim "diversity correlates with performance"
→ Weaker but still publishable

---

## Honest Assessment of Current State

### What We Know

1. H2 (Action Diversity) correlates with CartPole performance (r = +0.71)
2. Full K is worse than H2 alone (averaging dilutes signal)
3. Simple K is unreliable (high variance)
4. 3 of 7 harmonies anti-correlate in CartPole

### What We Don't Know

1. Does H2 correlate in original environments?
2. Why did Simple K show r = -0.81 once?
3. What is "performance" in Track B?
4. Is the correlation causal?

### What We Should Not Claim (Yet)

1. ❌ "H2 predicts performance in general" (only tested CartPole)
2. ❌ "Simple K anti-correlates" (positive in 95% of tests)
3. ❌ "Full K is the correct metric" (worse than H2 alone)
4. ❌ "Diversity causes good performance" (correlation ≠ causation)

---

## Conclusion

**We have a promising finding (H2 → performance) but have not validated it in the environments that matter.**

Before publication:
1. Must validate H2 in original environments
2. Must define external performance metric
3. Should establish causation

**Timeline**: 1-2 weeks of validation, then 1-2 weeks of revision

**Risk**: Finding may not generalize (30-50% probability)

**Honest path**: Do the validation before making claims

---

*"The most important thing is not to fool yourself. We have not yet shown that our CartPole finding applies to the original research environments."*

