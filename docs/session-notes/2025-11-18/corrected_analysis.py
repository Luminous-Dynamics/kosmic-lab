#!/usr/bin/env python3
"""
Corrected Analysis

The flexibility metric returns negative values where MORE negative = MORE flexible.
This creates confusing correlation signs.

This script uses ABSOLUTE flexibility so the correlation has intuitive sign.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u corrected_analysis.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Policy:
    def __init__(self, obs_dim, action_dim):
        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
        self.obs_history = []
        self.action_history = []

    def get_action(self, obs):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        action = mean + std * np.random.randn(len(mean))
        action = np.clip(action, -1, 1)
        self.obs_history.append(obs.flatten())
        self.action_history.append(action.flatten())
        return action

    def get_flexibility_raw(self):
        """Returns raw flexibility: more negative = more flexible."""
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-50:])
        actions = np.concatenate(self.action_history[-50:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def get_flexibility_absolute(self):
        """Returns absolute flexibility: higher = more flexible."""
        return abs(self.get_flexibility_raw())

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


class Env:
    def __init__(self):
        self.n_agents = 4
        self.obs_dim = 6
        self.action_dim = 3
        self.adj = np.ones((4, 4)) - np.eye(4)

    def reset(self):
        self.state = np.random.randn(6) * 0.1
        self.target = np.random.randn(6)
        self.steps = 0
        return [self.state + np.random.randn(6) * 0.2 for _ in range(4)]

    def step(self, actions):
        action_sum = np.zeros(3)
        for i in range(4):
            for j in range(4):
                if i != j:
                    action_sum += self.adj[i, j] * actions[j]
        action_mean = action_sum / 12

        if len(action_mean) < 6:
            action_mean = np.pad(action_mean, (0, 6 - len(action_mean)))

        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)

        alignments = []
        for i in range(4):
            for j in range(i+1, 4):
                alignments.append(np.dot(actions[i], actions[j]))
        coord = np.mean(alignments)

        reward = -dist * 0.5 + coord * 0.5
        self.steps += 1
        done = dist < 0.3 or self.steps >= 60

        return [self.state + np.random.randn(6) * 0.2 for _ in range(4)], [reward] * 4, done


def main():
    print("\n" + "=" * 60)
    print("CORRECTED FLEXIBILITY ANALYSIS")
    print("=" * 60)
    print("\nUsing ABSOLUTE flexibility (higher = more flexible)")

    n_teams = 150
    results = []

    print(f"\nEvaluating {n_teams} random teams...")
    for i in range(n_teams):
        env = Env()
        policies = [Policy(6, 3) for _ in range(4)]

        rewards = []
        flex_raw = []
        flex_abs = []

        for _ in range(3):
            obs_list = env.reset()
            for p in policies:
                p.reset_history()
            total = 0
            for step in range(60):
                actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
                obs_list, rews, done = env.step(actions)
                total += rews[0]
                if done:
                    break
            rewards.append(total)
            flex_raw.append(np.mean([p.get_flexibility_raw() for p in policies]))
            flex_abs.append(np.mean([p.get_flexibility_absolute() for p in policies]))

        results.append({
            'flex_raw': np.mean(flex_raw),
            'flex_abs': np.mean(flex_abs),
            'reward': np.mean(rewards)
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_teams} complete")

    flex_raw = np.array([x['flex_raw'] for x in results])
    flex_abs = np.array([x['flex_abs'] for x in results])
    rewards = np.array([x['reward'] for x in results])

    r_raw, p_raw = stats.pearsonr(flex_raw, rewards)
    r_abs, p_abs = stats.pearsonr(flex_abs, rewards)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    sig_raw = '***' if p_raw < 0.001 else '**' if p_raw < 0.01 else '*' if p_raw < 0.05 else 'ns'
    sig_abs = '***' if p_abs < 0.001 else '**' if p_abs < 0.01 else '*' if p_abs < 0.05 else 'ns'

    print(f"\nRaw flexibility (more negative = more flexible):")
    print(f"  r = {r_raw:+.3f}, p = {p_raw:.4f} {sig_raw}")
    print(f"  Range: [{flex_raw.min():.3f}, {flex_raw.max():.3f}]")

    print(f"\nAbsolute flexibility (higher = more flexible):")
    print(f"  r = {r_abs:+.3f}, p = {p_abs:.4f} {sig_abs}")
    print(f"  Range: [{flex_abs.min():.3f}, {flex_abs.max():.3f}]")

    # Effect size with absolute
    median = np.median(flex_abs)
    high = rewards[flex_abs > median]
    low = rewards[flex_abs <= median]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    print(f"\nEffect size (Cohen's d): {d:+.3f}")

    # Bootstrap CI
    boot_r = []
    for _ in range(500):
        idx = np.random.choice(len(flex_abs), len(flex_abs), replace=True)
        boot_r.append(stats.pearsonr(flex_abs[idx], rewards[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])
    print(f"95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")

    # Quartile breakdown
    print("\nQuartile Analysis:")
    q1, q2, q3 = np.percentile(flex_abs, [25, 50, 75])
    for lo, hi, name in [
        (flex_abs.min(), q1, "Q1 (least flex)"),
        (q1, q2, "Q2"),
        (q2, q3, "Q3"),
        (q3, flex_abs.max(), "Q4 (most flex)")
    ]:
        mask = (flex_abs >= lo) & (flex_abs <= hi)
        mean_r = rewards[mask].mean()
        print(f"  {name}: reward = {mean_r:.1f}")

    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    if p_abs < 0.05:
        if r_abs > 0:
            print("✓ SIGNIFICANT POSITIVE: More flexibility → better coordination")
        else:
            print("✓ SIGNIFICANT NEGATIVE: More flexibility → worse coordination")
    elif p_abs < 0.10:
        print("⚠️ MARGINAL: Trend visible but not significant")
    else:
        print("→ Not significant at this sample size")

    print(f"\nConclusion: r = {r_abs:+.3f} with n={n_teams}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"corrected_analysis_{timestamp}.npz"
    np.savez(filename, flex_abs=flex_abs, flex_raw=flex_raw, rewards=rewards,
             r_abs=r_abs, p_abs=p_abs, d=d)
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
