#!/usr/bin/env python3
"""
Validate K-Index Correlation with ACTUAL Rewards in Track E

Track E has both k_indices AND rewards - this is the test we need!

Critical questions:
1. Does K-Index correlate with rewards (external performance)?
2. Is the correlation positive or negative?
3. Does this validate or invalidate our paradigm shift narrative?
"""

import numpy as np
from scipy import stats

def main():
    print("=" * 70)
    print("VALIDATING K-INDEX vs REWARDS IN TRACK E")
    print("=" * 70)
    print("\nThis is the CRITICAL test: Does K correlate with external performance?\n")

    # Load Track E data
    data = np.load('logs/track_e/developmental/track_e_20251111_162703.npz',
                   allow_pickle=True)

    conditions = ['standard_rl', 'curriculum_learning', 'meta_learning', 'full_developmental']

    all_k = []
    all_rewards = []

    print("=" * 70)
    print("DATA BY CONDITION")
    print("=" * 70)

    for cond in conditions:
        d = data[cond].item()
        k_indices = d['k_indices']
        rewards = d['rewards']

        all_k.extend(k_indices)
        all_rewards.extend(rewards)

        # Correlation within condition
        r, p = stats.pearsonr(k_indices, rewards)

        print(f"\n{cond}:")
        print(f"  N = {len(k_indices)}")
        print(f"  K-Index: {np.mean(k_indices):.3f} ± {np.std(k_indices):.3f}")
        print(f"  Rewards: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"  K vs Rewards: r = {r:+.3f} (p = {p:.4f})")

        if r > 0.3 and p < 0.05:
            print(f"  → ✅ POSITIVE correlation")
        elif r < -0.3 and p < 0.05:
            print(f"  → ❌ NEGATIVE correlation")
        else:
            print(f"  → Weak/no significant correlation")

    # Overall correlation
    print("\n" + "=" * 70)
    print("OVERALL CORRELATION (All Conditions Combined)")
    print("=" * 70)

    all_k = np.array(all_k)
    all_rewards = np.array(all_rewards)

    r_overall, p_overall = stats.pearsonr(all_k, all_rewards)
    r_spearman, p_spearman = stats.spearmanr(all_k, all_rewards)

    print(f"\nN = {len(all_k)}")
    print(f"K-Index: {np.mean(all_k):.3f} ± {np.std(all_k):.3f}")
    print(f"  Range: [{np.min(all_k):.3f}, {np.max(all_k):.3f}]")
    print(f"Rewards: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"  Range: [{np.min(all_rewards):.3f}, {np.max(all_rewards):.3f}]")

    print(f"\nPearson correlation:  r = {r_overall:+.3f} (p = {p_overall:.4f})")
    print(f"Spearman correlation: ρ = {r_spearman:+.3f} (p = {p_spearman:.4f})")

    # Effect size
    r2 = r_overall ** 2
    print(f"\nVariance explained: {r2*100:.1f}%")

    print("\n" + "=" * 70)
    print("CRITICAL INTERPRETATION")
    print("=" * 70)

    if r_overall > 0.3 and p_overall < 0.05:
        print("\n✅ K-INDEX POSITIVELY CORRELATES WITH REWARDS")
        print("   This validates that K predicts external performance!")
        print("   The paradigm shift narrative needs revision:")
        print("   - Simple K may actually work (in some contexts)")
        print("   - Or this is a different K computation")
    elif r_overall < -0.3 and p_overall < 0.05:
        print("\n❌ K-INDEX NEGATIVELY CORRELATES WITH REWARDS")
        print("   This validates our paradigm shift finding!")
        print("   Higher K = worse performance")
        print("   We were right that Simple K measures rigidity")
    elif abs(r_overall) < 0.1:
        print("\n⚠️  K-INDEX HAS NO CORRELATION WITH REWARDS")
        print("   K-Index does not predict external performance")
        print("   This is a serious problem for all K-Index papers")
    else:
        print(f"\n⚠️  WEAK CORRELATION (r = {r_overall:+.3f})")
        print("   K-Index has limited predictive value")

    # Check trajectory - does K grow with learning?
    print("\n" + "=" * 70)
    print("LEARNING TRAJECTORY ANALYSIS")
    print("=" * 70)

    for cond in conditions:
        d = data[cond].item()
        k_indices = d['k_indices']
        rewards = d['rewards']
        episodes = np.arange(len(k_indices))

        # Correlation with episode number (learning progress)
        r_k_ep, p_k_ep = stats.pearsonr(episodes, k_indices)
        r_rew_ep, p_rew_ep = stats.pearsonr(episodes, rewards)

        print(f"\n{cond}:")
        print(f"  K vs Episode:      r = {r_k_ep:+.3f} (p = {p_k_ep:.4f})")
        print(f"  Reward vs Episode: r = {r_rew_ep:+.3f} (p = {p_rew_ep:.4f})")

        if r_k_ep > 0 and r_rew_ep > 0:
            print(f"  → Both increase during learning")
        elif r_k_ep < 0 and r_rew_ep > 0:
            print(f"  → K decreases but reward increases!")
        elif r_k_ep > 0 and r_rew_ep < 0:
            print(f"  → K increases but reward decreases!")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if r_overall > 0.1:
        print("\n✅ K-Index POSITIVELY correlates with external rewards")
        print("   This contradicts our CartPole finding that Simple K anti-correlates")
        print("\n   Possible explanations:")
        print("   1. Track E uses a different K computation")
        print("   2. The relationship is context-dependent")
        print("   3. Track E's task structure is different")
    elif r_overall < -0.1:
        print("\n✅ K-Index NEGATIVELY correlates with external rewards")
        print("   This CONFIRMS our paradigm shift finding!")
    else:
        print("\n⚠️  K-Index does NOT predict rewards")
        print("   This undermines all K-Index performance claims")

    print("\n✅ Analysis complete\n")

if __name__ == '__main__':
    main()
