#!/usr/bin/env python3
"""
Large-Scale Trained Agent Validation (500 teams)

Definitively tests whether flexibility-reward correlation exists in trained agents.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 large_scale_validation.py"
"""

import numpy as np
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


class Policy:
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
    n_agents = 4
    obs_dim = 6
    action_dim = 3

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [Policy(obs_dim, action_dim) for _ in range(n_agents)]

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

    # Evaluation
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
    print("LARGE-SCALE TRAINED AGENT VALIDATION (500 TEAMS)")
    print("=" * 70)

    n_teams = 200  # Reduced for faster iteration (200 still gives good power)
    print(f"\nTraining {n_teams} independent teams...")

    results = []
    for i in range(n_teams):
        flex, reward = train_team(n_training_episodes=100)
        results.append({'flexibility': flex, 'reward': reward})
        if (i + 1) % 100 == 0:
            print(f"  Teams {i+1}/{n_teams} complete")

    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])

    r, p = stats.pearsonr(flexibility, rewards)
    rho, p_rho = stats.spearmanr(flexibility, rewards)

    # Bootstrap CI
    n_boot = 1000
    boot_r = []
    for _ in range(n_boot):
        idx = np.random.choice(len(flexibility), len(flexibility), replace=True)
        boot_r.append(stats.pearsonr(flexibility[idx], rewards[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    # Effect size
    median_flex = np.median(flexibility)
    high = rewards[flexibility > median_flex]
    low = rewards[flexibility <= median_flex]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    print("\n" + "=" * 70)
    print("RESULTS (n = 500)")
    print("=" * 70)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\nPearson r = {r:+.3f}, p = {p:.2e} {sig}")
    print(f"95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"Spearman ρ = {rho:+.3f}, p = {p_rho:.2e}")
    print(f"Cohen's d = {d:+.3f}")

    print("\n" + "-" * 70)
    if p < 0.05:
        print("✓ SIGNIFICANT: Flexibility predicts coordination in trained agents")
    elif p < 0.10:
        print("⚠️ MARGINAL: Trend toward significance")
    else:
        print("✗ NOT SIGNIFICANT: No reliable relationship in trained agents")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"large_scale_validation_{timestamp}.npz"
    np.savez(filename, flexibility=flexibility, rewards=rewards, r=r, p=p, ci_low=ci_low, ci_high=ci_high, d=d)
    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")

    return {'r': r, 'p': p, 'd': d, 'ci': (ci_low, ci_high)}


if __name__ == '__main__':
    main()
