#!/usr/bin/env python3
"""
Diagnostic Analysis

Investigates why the flex-reward correlation is not appearing:
1. Check flexibility variance
2. Check reward variance
3. Test different environment structures
4. Compare to original experiment setup

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u diagnostic_analysis.py"
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
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


class OriginalEnv:
    """Environment matching original experiments more closely."""

    def __init__(self, n_agents=4, obs_dim=6, action_dim=3):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Adjacency matrix for agent interactions
        self.adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def reset(self):
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.target = np.random.randn(self.obs_dim)
        self.steps = 0
        # Give each agent a different view
        return [self.state + np.random.randn(self.obs_dim) * 0.2 for _ in range(self.n_agents)]

    def step(self, actions):
        # Weighted action combination based on adjacency
        action_sum = np.zeros(self.action_dim)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    action_sum += self.adj[i, j] * actions[j]
        action_mean = action_sum / (self.n_agents * (self.n_agents - 1))

        if len(action_mean) < self.obs_dim:
            action_mean = np.pad(action_mean, (0, self.obs_dim - len(action_mean)))

        self.state += action_mean * 0.1

        # Distance reward
        dist = np.linalg.norm(self.state - self.target)

        # Coordination reward (more complex - based on alignment)
        action_vecs = [a.flatten() for a in actions]
        alignments = []
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                dot = np.dot(action_vecs[i], action_vecs[j])
                alignments.append(dot)
        coord = np.mean(alignments) if alignments else 0

        # Combined reward
        reward = -dist * 0.5 + coord * 0.5

        self.steps += 1
        done = dist < 0.3 or self.steps >= 60

        # Different observations for each agent
        return [self.state + np.random.randn(self.obs_dim) * 0.2 for _ in range(self.n_agents)], [reward] * self.n_agents, done


class SimpleEnv:
    """Our current simple environment."""

    def __init__(self):
        self.n_agents = 4
        self.obs_dim = 6
        self.action_dim = 3

    def reset(self):
        self.state = np.random.randn(6) * 0.1
        self.target = np.random.randn(6)
        self.steps = 0
        return [self.state + np.random.randn(6) * 0.05 for _ in range(4)]

    def step(self, actions):
        action_mean = np.mean(actions, axis=0)
        if len(action_mean) < 6:
            action_mean = np.pad(action_mean, (0, 6 - len(action_mean)))
        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)
        coord = 1 - np.std([np.mean(a) for a in actions])
        reward = -dist + coord * 0.2
        self.steps += 1
        done = dist < 0.3 or self.steps >= 60
        return [self.state + np.random.randn(6) * 0.05 for _ in range(4)], [reward] * 4, done


def eval_with_env(env_class, n_teams=50):
    """Evaluate teams in a given environment."""
    results = []
    for _ in range(n_teams):
        env = env_class()
        policies = [Policy(6, 3) for _ in range(4)]

        rewards = []
        flexes = []
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
            flexes.append(np.mean([p.get_flexibility() for p in policies]))

        results.append({'flex': np.mean(flexes), 'reward': np.mean(rewards)})

    return results


def main():
    print("\n" + "=" * 60)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 60)

    # ================================================================
    # Test 1: Simple vs Original Environment
    # ================================================================
    print("\n" + "-" * 60)
    print("Test 1: Environment Comparison")
    print("-" * 60)

    print("\nSimple Environment:")
    simple_results = eval_with_env(SimpleEnv, n_teams=50)
    simple_flex = np.array([x['flex'] for x in simple_results])
    simple_reward = np.array([x['reward'] for x in simple_results])
    r_simple, p_simple = stats.pearsonr(simple_flex, simple_reward)
    print(f"  r = {r_simple:+.3f}, p = {p_simple:.3f}")
    print(f"  Flex range: [{simple_flex.min():.3f}, {simple_flex.max():.3f}]")
    print(f"  Flex std: {simple_flex.std():.3f}")
    print(f"  Reward range: [{simple_reward.min():.1f}, {simple_reward.max():.1f}]")

    print("\nOriginal-style Environment:")
    orig_results = eval_with_env(OriginalEnv, n_teams=50)
    orig_flex = np.array([x['flex'] for x in orig_results])
    orig_reward = np.array([x['reward'] for x in orig_results])
    r_orig, p_orig = stats.pearsonr(orig_flex, orig_reward)
    print(f"  r = {r_orig:+.3f}, p = {p_orig:.3f}")
    print(f"  Flex range: [{orig_flex.min():.3f}, {orig_flex.max():.3f}]")
    print(f"  Flex std: {orig_flex.std():.3f}")
    print(f"  Reward range: [{orig_reward.min():.1f}, {orig_reward.max():.1f}]")

    # ================================================================
    # Test 2: Check for range restriction
    # ================================================================
    print("\n" + "-" * 60)
    print("Test 2: Variance Analysis")
    print("-" * 60)

    # Combine all
    all_flex = np.concatenate([simple_flex, orig_flex])
    all_reward = np.concatenate([simple_reward, orig_reward])

    print(f"\nFlexibility coefficient of variation: {simple_flex.std() / abs(simple_flex.mean()):.2f}")
    print(f"Reward coefficient of variation: {simple_reward.std() / abs(simple_reward.mean()):.2f}")

    # ================================================================
    # Test 3: Correlation by flexibility quartile
    # ================================================================
    print("\n" + "-" * 60)
    print("Test 3: Quartile Analysis (Original Env)")
    print("-" * 60)

    # Run more for better quartiles
    large_results = eval_with_env(OriginalEnv, n_teams=100)
    flex = np.array([x['flex'] for x in large_results])
    reward = np.array([x['reward'] for x in large_results])

    q1, q2, q3 = np.percentile(flex, [25, 50, 75])

    print(f"\nFlexibility quartiles: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}")

    for i, (lo, hi, name) in enumerate([
        (flex.min(), q1, "Q1 (lowest)"),
        (q1, q2, "Q2"),
        (q2, q3, "Q3"),
        (q3, flex.max(), "Q4 (highest)")
    ]):
        mask = (flex >= lo) & (flex <= hi)
        mean_reward = reward[mask].mean()
        print(f"  {name}: mean reward = {mean_reward:.1f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    r_combined, p_combined = stats.pearsonr(flex, reward)
    sig = '***' if p_combined < 0.001 else '**' if p_combined < 0.01 else '*' if p_combined < 0.05 else 'ns'

    print(f"\nOriginal-style env (n=100): r = {r_combined:+.3f}, p = {p_combined:.4f} {sig}")

    if p_combined < 0.05:
        print("\n✓ Original-style environment shows significant correlation")
        print("  The issue was the simple environment structure")
    else:
        print("\n→ Issue not resolved by environment change")
        print("  Need to investigate flexibility metric itself")

    # Effect size
    median = np.median(flex)
    high = reward[flex > median]
    low = reward[flex <= median]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)
    print(f"\nEffect size: d = {d:+.3f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnostic_{timestamp}.npz"
    np.savez(filename, flex=flex, reward=reward, r=r_combined, p=p_combined)
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
