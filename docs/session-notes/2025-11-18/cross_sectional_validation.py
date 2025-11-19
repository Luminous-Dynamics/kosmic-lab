#!/usr/bin/env python3
"""
Cross-Sectional Validation with Trained Agents

Tests the key insight: flexibility-reward correlation is cross-sectional
(comparing different agents) not longitudinal (during training).

Approach:
1. Train many independent agent teams
2. Evaluate each at end of training
3. Correlate flexibility with performance across teams

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 cross_sectional_validation.py"
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from datetime import datetime


class FlexibilityTracker:
    def __init__(self):
        self.obs_history = []
        self.action_history = []

    def record(self, obs, action):
        self.obs_history.append(np.array(obs).flatten())
        self.action_history.append(np.array(action).flatten())

    def get_flexibility(self):
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-100:])
        actions = np.concatenate(self.action_history[-100:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def reset(self):
        self.obs_history = []
        self.action_history = []


class SimplePolicy:
    def __init__(self, obs_dim, action_dim):
        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
        self.lr = 0.003

    def get_action(self, obs):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        action = mean + std * np.random.randn(len(mean))
        return np.clip(action, -1, 1)

    def update(self, obs, action, advantage):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)
        self.weights += self.lr * np.clip(advantage, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)


class CoordinationEnv:
    def __init__(self, n_agents=4, obs_dim=6, action_dim=3):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def reset(self):
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.target = np.random.randn(self.obs_dim)
        self.steps = 0
        return [self.state + np.random.randn(self.obs_dim) * 0.05 for _ in range(self.n_agents)]

    def step(self, actions):
        action_mean = np.mean(actions, axis=0)
        if len(action_mean) < self.obs_dim:
            action_mean = np.pad(action_mean, (0, self.obs_dim - len(action_mean)))
        self.state += action_mean * 0.1

        dist = np.linalg.norm(self.state - self.target)
        coord = 1 - np.std([np.mean(a) for a in actions])
        reward = -dist + coord * 0.2

        self.steps += 1
        done = dist < 0.3 or self.steps >= 80

        return [self.state + np.random.randn(self.obs_dim) * 0.05 for _ in range(self.n_agents)], [reward] * self.n_agents, done


def train_team(n_training_episodes=100):
    """Train one team and return final performance."""
    n_agents = 4
    obs_dim = 6
    action_dim = 3

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [SimplePolicy(obs_dim, action_dim) for _ in range(n_agents)]

    # Training
    for ep in range(n_training_episodes):
        obs_list = env.reset()
        episode_data = [[] for _ in range(n_agents)]

        for step in range(80):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            next_obs, rewards, done = env.step(actions)

            for i in range(n_agents):
                episode_data[i].append({'obs': obs_list[i], 'action': actions[i], 'reward': rewards[i]})

            obs_list = next_obs
            if done:
                break

        # Update
        for i in range(n_agents):
            returns = []
            G = 0
            for t in reversed(range(len(episode_data[i]))):
                G = episode_data[i][t]['reward'] + 0.99 * G
                returns.insert(0, G)
            if len(returns) > 1:
                returns = np.array(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                for t, G in enumerate(returns):
                    policies[i].update(episode_data[i][t]['obs'], episode_data[i][t]['action'], G)

    # Evaluation (5 episodes)
    eval_rewards = []
    eval_flex = []

    for _ in range(5):
        obs_list = env.reset()
        trackers = [FlexibilityTracker() for _ in range(n_agents)]
        total_reward = 0

        for step in range(80):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            for i, (obs, action) in enumerate(zip(obs_list, actions)):
                trackers[i].record(obs, action)

            obs_list, rewards, done = env.step(actions)
            total_reward += rewards[0]

            if done:
                break

        eval_rewards.append(total_reward)
        eval_flex.append(np.mean([t.get_flexibility() for t in trackers]))

    return np.mean(eval_flex), np.mean(eval_rewards)


def main():
    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL VALIDATION WITH TRAINED AGENTS")
    print("=" * 70)

    n_teams = 100  # Number of independent teams to train

    print(f"\nTraining {n_teams} independent teams...")

    results = []
    for i in range(n_teams):
        flex, reward = train_team(n_training_episodes=100)
        results.append({'flexibility': flex, 'reward': reward})

        if (i + 1) % 20 == 0:
            print(f"  Teams {i+1}/{n_teams} complete")

    # Analysis
    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])

    r, p = stats.pearsonr(flexibility, rewards)
    rho, p_rho = stats.spearmanr(flexibility, rewards)

    # Effect size
    median_flex = np.median(flexibility)
    high = rewards[flexibility > median_flex]
    low = rewards[flexibility <= median_flex]
    if len(high) > 1 and len(low) > 1:
        d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)
        t, p_t = stats.ttest_ind(high, low)
    else:
        d = 0
        t, p_t = 0, 1

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL RESULTS")
    print("=" * 70)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\nPearson r = {r:+.3f}, p = {p:.2e} {sig}")
    print(f"Spearman ρ = {rho:+.3f}, p = {p_rho:.2e}")
    print(f"Cohen's d = {d:+.3f}")
    print(f"High-flex mean: {high.mean():+.3f}, Low-flex mean: {low.mean():+.3f}")

    print("\n" + "-" * 70)

    if p < 0.05 and r > 0:
        print("✓ VALIDATED: Flexibility predicts coordination in trained agents")
        print(f"  {n_teams} independently trained teams show r = {r:+.3f}")
    elif p < 0.05 and r < 0:
        print("⚠️ UNEXPECTED: Negative correlation in trained agents")
    else:
        print("✗ No significant cross-sectional correlation in trained agents")

    # Compare to random
    print("\n" + "-" * 70)
    print("TRAINED vs RANDOM COMPARISON")
    print("-" * 70)

    # Quick random baseline
    random_results = []
    env = CoordinationEnv()
    for _ in range(50):
        obs_list = env.reset()
        trackers = [FlexibilityTracker() for _ in range(4)]
        total = 0
        for step in range(80):
            actions = [np.random.randn(3) * 0.5 for _ in range(4)]
            for i, (obs, act) in enumerate(zip(obs_list, actions)):
                trackers[i].record(obs, act)
            obs_list, rewards, done = env.step(actions)
            total += rewards[0]
            if done:
                break
        random_results.append({'flexibility': np.mean([t.get_flexibility() for t in trackers]), 'reward': total})

    random_flex = np.array([r['flexibility'] for r in random_results])
    random_reward = np.array([r['reward'] for r in random_results])
    r_random, p_random = stats.pearsonr(random_flex, random_reward)

    print(f"Trained teams: r = {r:+.3f} (n={n_teams})")
    print(f"Random policies: r = {r_random:+.3f} (n=50)")

    if r > r_random:
        print(f"\n✓ Trained agents show stronger flex-reward relationship")
    else:
        print(f"\n→ Similar relationship in trained and random agents")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cross_sectional_validation_{timestamp}.npz"
    np.savez(filename,
             flexibility=flexibility,
             rewards=rewards,
             r=r, p=p, d=d,
             random_r=r_random)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
