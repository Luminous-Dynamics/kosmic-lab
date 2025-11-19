#!/usr/bin/env python3
"""
Investigate Track E Reward Meaningfulness

Critical question: Are Track E rewards actually a meaningful performance metric,
or are they essentially random noise?

The reward formula is:
  action_quality = dot(action, state[:action_dim]) / (action_dim * difficulty)
  reward = tanh(action_quality)

But the state is random noise that changes each step. Does this create
a meaningful learning signal?
"""

import numpy as np
from scipy import stats

def main():
    print("=" * 70)
    print("INVESTIGATING TRACK E REWARD MEANINGFULNESS")
    print("=" * 70)

    # Load Track E data
    data = np.load('logs/track_e/developmental/track_e_20251111_162703.npz',
                   allow_pickle=True)

    conditions = ['standard_rl', 'curriculum_learning', 'meta_learning', 'full_developmental']

    print("\n1. REWARD STATISTICS")
    print("-" * 70)

    all_rewards = []
    for cond in conditions:
        d = data[cond].item()
        rewards = d['rewards']
        all_rewards.extend(rewards)

        print(f"\n{cond}:")
        print(f"  Mean: {np.mean(rewards):.6f}")
        print(f"  Std:  {np.std(rewards):.6f}")
        print(f"  Min:  {np.min(rewards):.6f}")
        print(f"  Max:  {np.max(rewards):.6f}")

        # Check if mean is significantly different from 0
        t_stat, p_val = stats.ttest_1samp(rewards, 0)
        print(f"  Mean ≠ 0: t={t_stat:.2f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"    → Mean is significantly {'positive' if np.mean(rewards) > 0 else 'negative'}")
        else:
            print(f"    → Mean is NOT significantly different from 0 (random!)")

    print("\n" + "=" * 70)
    print("2. DO REWARDS IMPROVE DURING TRAINING?")
    print("-" * 70)

    for cond in conditions:
        d = data[cond].item()
        rewards = d['rewards']
        episodes = np.arange(len(rewards))

        # Correlation with episode number
        r, p = stats.pearsonr(episodes, rewards)

        # Compare first half vs second half
        mid = len(rewards) // 2
        first_half = rewards[:mid]
        second_half = rewards[mid:]
        t_stat, p_half = stats.ttest_ind(first_half, second_half)

        print(f"\n{cond}:")
        print(f"  Reward trend: r = {r:+.3f} (p = {p:.4f})")
        print(f"  First half mean:  {np.mean(first_half):.6f}")
        print(f"  Second half mean: {np.mean(second_half):.6f}")
        print(f"  Improvement: {(np.mean(second_half) - np.mean(first_half)):.6f}")
        print(f"  Significant? p = {p_half:.4f}")

        if p < 0.05:
            print(f"    → Rewards {'increase' if r > 0 else 'decrease'} during training")
        else:
            print(f"    → No significant learning trend")

    print("\n" + "=" * 70)
    print("3. REWARD AUTOCORRELATION (Is it random noise?)")
    print("-" * 70)

    for cond in conditions:
        d = data[cond].item()
        rewards = d['rewards']

        # Lag-1 autocorrelation
        if len(rewards) > 1:
            autocorr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1]
        else:
            autocorr = 0

        print(f"\n{cond}:")
        print(f"  Lag-1 autocorrelation: {autocorr:.3f}")

        if abs(autocorr) < 0.1:
            print(f"    → Rewards appear to be random (no autocorrelation)")
        elif autocorr > 0.3:
            print(f"    → Rewards show persistence (not random)")
        else:
            print(f"    → Weak autocorrelation")

    print("\n" + "=" * 70)
    print("4. COMPARISON WITH RANDOM BASELINE")
    print("-" * 70)

    # Generate what random rewards would look like
    n_episodes = 50
    random_rewards = np.random.randn(n_episodes) * 0.01  # Similar scale

    print(f"\nRandom baseline (same scale):")
    print(f"  Mean: {np.mean(random_rewards):.6f}")
    print(f"  Std:  {np.std(random_rewards):.6f}")

    all_rewards = np.array(all_rewards)
    print(f"\nActual Track E rewards:")
    print(f"  Mean: {np.mean(all_rewards):.6f}")
    print(f"  Std:  {np.std(all_rewards):.6f}")

    # Statistical test: are they different from random?
    t_stat, p_val = stats.ttest_1samp(all_rewards, 0)
    print(f"\nDifferent from random (mean=0)? p = {p_val:.4f}")

    print("\n" + "=" * 70)
    print("5. CRITICAL ANALYSIS")
    print("=" * 70)

    # Check if any condition shows meaningful learning
    any_learning = False
    for cond in conditions:
        d = data[cond].item()
        rewards = d['rewards']
        episodes = np.arange(len(rewards))
        r, p = stats.pearsonr(episodes, rewards)
        if p < 0.05 and r > 0.2:
            any_learning = True
            break

    if not any_learning:
        print("\n⚠️  CRITICAL FINDING: Rewards show NO learning trend")
        print("   None of the conditions show significant reward improvement.")
        print("   This suggests the 'rewards' may not be a meaningful metric.")

    # Check if rewards are essentially noise
    mean_autocorr = 0
    for cond in conditions:
        d = data[cond].item()
        rewards = d['rewards']
        if len(rewards) > 1:
            mean_autocorr += np.corrcoef(rewards[:-1], rewards[1:])[0, 1]
    mean_autocorr /= len(conditions)

    if abs(mean_autocorr) < 0.15:
        print("\n⚠️  CRITICAL FINDING: Rewards appear to be random noise")
        print(f"   Mean autocorrelation: {mean_autocorr:.3f}")
        print("   Sequential rewards are uncorrelated (like random numbers).")

    # Final assessment
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    overall_mean = np.mean(all_rewards)
    t_stat, p_val = stats.ttest_1samp(all_rewards, 0)

    if p_val > 0.05 and abs(mean_autocorr) < 0.15:
        print("\n❌ TRACK E REWARDS ARE NOT A MEANINGFUL PERFORMANCE METRIC")
        print("\n   Evidence:")
        print(f"   1. Mean reward ≈ 0 ({overall_mean:.6f}, p = {p_val:.4f})")
        print(f"   2. No learning trend in any condition")
        print(f"   3. Low autocorrelation ({mean_autocorr:.3f}) - like random noise")
        print("\n   Implication:")
        print("   Finding that 'K doesn't predict rewards' is MEANINGLESS")
        print("   because the rewards themselves don't measure anything.")
        print("\n   Track E does NOT provide a valid test of K-Index performance.")
    else:
        print("\n✅ Track E rewards appear to be a meaningful metric")
        print("   The finding that K doesn't predict them is valid.")

    print("\n✅ Analysis complete\n")

if __name__ == '__main__':
    main()
