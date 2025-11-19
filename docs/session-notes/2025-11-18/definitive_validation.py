#!/usr/bin/env python3
"""
Definitive Validation (n=500)

Large-scale test to achieve statistical significance for r≈0.12.
Optimized for speed: fewer episodes, smaller eval.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u definitive_validation.py"
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

    def get_flexibility(self):
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-50:])
        actions = np.concatenate(self.action_history[-50:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return abs(corr) * 2.0 if not np.isnan(corr) else 0.0  # Absolute: higher = more flexible

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


class Env:
    def __init__(self):
        self.n_agents = 4
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
        done = dist < 0.3 or self.steps >= 50  # Shorter episodes

        return [self.state + np.random.randn(6) * 0.2 for _ in range(4)], [reward] * 4, done


def eval_team():
    """Evaluate one random team quickly."""
    env = Env()
    policies = [Policy(6, 3) for _ in range(4)]

    # Single evaluation episode
    obs_list = env.reset()
    for p in policies:
        p.reset_history()
    total = 0
    for step in range(50):
        actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
        obs_list, rews, done = env.step(actions)
        total += rews[0]
        if done:
            break

    flex = np.mean([p.get_flexibility() for p in policies])
    return flex, total


def main():
    print("\n" + "=" * 60)
    print("DEFINITIVE VALIDATION (n=500)")
    print("=" * 60)

    n_teams = 500
    results = []

    print(f"\nEvaluating {n_teams} random teams...")
    for i in range(n_teams):
        f, r = eval_team()
        results.append({'flex': f, 'reward': r})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_teams} complete")

    flexibility = np.array([x['flex'] for x in results])
    rewards = np.array([x['reward'] for x in results])

    r, p = stats.pearsonr(flexibility, rewards)
    rho, p_rho = stats.spearmanr(flexibility, rewards)

    # Bootstrap CI
    boot_r = []
    for _ in range(1000):
        idx = np.random.choice(len(flexibility), len(flexibility), replace=True)
        boot_r.append(stats.pearsonr(flexibility[idx], rewards[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    # Effect size
    median = np.median(flexibility)
    high = rewards[flexibility > median]
    low = rewards[flexibility <= median]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\nPearson r = {r:+.3f}, p = {p:.6f} {sig}")
    print(f"95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"Spearman ρ = {rho:+.3f}, p = {p_rho:.6f}")
    print(f"Cohen's d = {d:+.3f}")

    # Quartile analysis
    print("\nQuartile Analysis:")
    q1, q2, q3 = np.percentile(flexibility, [25, 50, 75])
    quartile_means = []
    for lo, hi, name in [
        (flexibility.min(), q1, "Q1 (least flex)"),
        (q1, q2, "Q2"),
        (q2, q3, "Q3"),
        (q3, flexibility.max(), "Q4 (most flex)")
    ]:
        mask = (flexibility >= lo) & (flexibility <= hi)
        mean_r = rewards[mask].mean()
        quartile_means.append(mean_r)
        print(f"  {name}: reward = {mean_r:.1f}")

    # Monotonicity test
    monotonic = all(quartile_means[i] <= quartile_means[i+1] for i in range(3))
    print(f"\nMonotonic progression: {'Yes' if monotonic else 'No'}")

    print("\n" + "-" * 60)
    print("CONCLUSION")
    print("-" * 60)

    if p < 0.05:
        print(f"✓ SIGNIFICANT at n={n_teams}")
        print(f"  Flexibility positively predicts coordination")
        print(f"  r = {r:+.3f}, d = {d:+.3f}")
        if p < 0.01:
            print(f"  Strong evidence (p < 0.01)")
        elif p < 0.05:
            print(f"  Moderate evidence (p < 0.05)")
    elif p < 0.10:
        print(f"⚠️ MARGINAL at n={n_teams}")
        print(f"  r = {r:+.3f}, p = {p:.4f}")
        print(f"  Need n≈{int(500 * (0.05/p)**2)} for p<0.05")
    else:
        print(f"✗ NOT SIGNIFICANT at n={n_teams}")
        print(f"  r = {r:+.3f}, p = {p:.4f}")
        print(f"  Effect may not exist or is too small")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"definitive_validation_{timestamp}.npz"
    np.savez(filename, flexibility=flexibility, rewards=rewards,
             r=r, p=p, d=d, ci_low=ci_low, ci_high=ci_high,
             quartile_means=quartile_means)
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
