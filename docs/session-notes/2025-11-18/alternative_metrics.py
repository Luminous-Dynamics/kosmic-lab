#!/usr/bin/env python3
"""
Alternative Flexibility Metrics

Tests multiple metrics to find one that actually predicts coordination.

Metrics tested:
1. Obs-action correlation (current)
2. Action entropy (exploration)
3. Response diversity (output variance)
4. Policy weight variance (parameter spread)

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u alternative_metrics.py"
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

    def get_flex_correlation(self):
        """Original metric: obs-action correlation (higher = more flexible)."""
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-50:])
        actions = np.concatenate(self.action_history[-50:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def get_action_entropy(self):
        """Metric 2: Entropy of action distribution."""
        if len(self.action_history) < 10:
            return 0.0
        actions = np.array(self.action_history[-50:])
        # Approximate entropy as variance
        return np.mean(np.var(actions, axis=0))

    def get_response_diversity(self):
        """Metric 3: Variance in outputs."""
        if len(self.action_history) < 10:
            return 0.0
        actions = np.array(self.action_history[-50:])
        return np.std(actions)

    def get_weight_variance(self):
        """Metric 4: Spread in policy parameters."""
        return np.std(self.weights)

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
        done = dist < 0.3 or self.steps >= 50

        return [self.state + np.random.randn(6) * 0.2 for _ in range(4)], [reward] * 4, done


def main():
    print("\n" + "=" * 60)
    print("ALTERNATIVE FLEXIBILITY METRICS")
    print("=" * 60)

    n_teams = 400
    results = []

    print(f"\nEvaluating {n_teams} random teams...")
    for i in range(n_teams):
        env = Env()
        policies = [Policy(6, 3) for _ in range(4)]

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

        results.append({
            'reward': total,
            'flex_corr': np.mean([p.get_flex_correlation() for p in policies]),
            'entropy': np.mean([p.get_action_entropy() for p in policies]),
            'diversity': np.mean([p.get_response_diversity() for p in policies]),
            'weight_var': np.mean([p.get_weight_variance() for p in policies]),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_teams} complete")

    # Extract arrays
    rewards = np.array([x['reward'] for x in results])
    metrics = {
        'Obs-Action Corr': np.array([x['flex_corr'] for x in results]),
        'Action Entropy': np.array([x['entropy'] for x in results]),
        'Response Diversity': np.array([x['diversity'] for x in results]),
        'Weight Variance': np.array([x['weight_var'] for x in results]),
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'r':>10} {'p':>12} {'d':>10}")
    print("-" * 60)

    best_metric = None
    best_r = 0

    for name, values in metrics.items():
        r, p = stats.pearsonr(values, rewards)

        # Effect size
        median = np.median(values)
        high = rewards[values > median]
        low = rewards[values <= median]
        d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{name:<25} {r:>+10.3f} {p:>12.4f}{sig:>3} {d:>+10.3f}")

        if abs(r) > abs(best_r):
            best_r = r
            best_metric = name

    print("\n" + "-" * 60)
    print("BEST METRIC")
    print("-" * 60)
    print(f"{best_metric}: r = {best_r:+.3f}")

    # Detailed analysis of best metric
    best_values = metrics[best_metric]
    r, p = stats.pearsonr(best_values, rewards)

    if p < 0.05:
        print(f"✓ SIGNIFICANT (p = {p:.4f})")

        # Quartile analysis
        print("\nQuartile Analysis:")
        q1, q2, q3 = np.percentile(best_values, [25, 50, 75])
        for lo, hi, name in [
            (best_values.min(), q1, "Q1"),
            (q1, q2, "Q2"),
            (q2, q3, "Q3"),
            (q3, best_values.max(), "Q4")
        ]:
            mask = (best_values >= lo) & (best_values <= hi)
            mean_r = rewards[mask].mean()
            print(f"  {name}: {mean_r:.1f}")
    else:
        print(f"✗ Not significant (p = {p:.4f})")
        print("\nNo metric predicts coordination performance.")
        print("Possible explanations:")
        print("  1. Random policies lack the structure needed")
        print("  2. The environment doesn't reward flexibility")
        print("  3. Need trained policies for meaningful metrics")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alternative_metrics_{timestamp}.npz"
    np.savez(filename, rewards=rewards, **metrics)
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
