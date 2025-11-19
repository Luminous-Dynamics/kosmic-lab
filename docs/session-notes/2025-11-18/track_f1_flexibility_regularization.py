#!/usr/bin/env python3
"""
Track F1: Flexibility Regularization (Causal Test)

Does IMPOSING flexibility through regularization improve coordination?
Tests causality: if flexibility causes better coordination, then encouraging
it during training should improve outcomes.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_f1_flexibility_regularization.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class RegularizedAgent:
    """Agent with entropy bonus to encourage flexibility."""

    def __init__(self, agent_id, lambda_flex=0.0):
        self.id = agent_id
        self.lambda_flex = lambda_flex
        self.policy_weights = np.random.randn(10, 15) * 0.1
        self.obs_history = []
        self.action_history = []
        self.rewards = []

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        mean = self.policy_weights @ combined

        # Exploration noise (higher lambda = more exploration)
        noise_scale = 0.3 + self.lambda_flex * 0.5
        action = np.tanh(mean + np.random.randn(10) * noise_scale)

        self.obs_history.append(obs)
        self.action_history.append(action)
        return action

    def get_flexibility(self):
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-50:]).flatten()
        actions = np.array(self.action_history[-50:]).flatten()
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def update(self, learning_rate=0.01):
        if not self.rewards:
            return

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = np.array(returns)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update with flexibility bonus
        for i in range(min(len(self.obs_history), len(returns))):
            obs = self.obs_history[i]
            action = self.action_history[i]
            ret = returns[i]

            # Entropy bonus: reward action variance
            if i > 0:
                prev_action = self.action_history[i-1]
                entropy_bonus = np.sum((action - prev_action)**2)
            else:
                entropy_bonus = 0

            modified_return = ret + self.lambda_flex * entropy_bonus

            gradient = np.outer(action, np.concatenate([obs, np.zeros(5)]))
            self.policy_weights += learning_rate * modified_return * gradient[:, :15]

        self.obs_history = []
        self.action_history = []
        self.rewards = []

    def create_message(self, obs):
        return obs[:5]


class Network:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def exchange(self, messages):
        return [np.mean([messages[j] for j in range(self.n_agents)
                        if self.adj[i,j] > 0], axis=0)
                for i in range(self.n_agents)]


class Environment:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.state = np.zeros(10)
        self.target = np.random.randn(10)

    def reset(self):
        self.state = np.random.randn(10) * 0.1
        self.target = np.random.randn(10)
        return self.state

    def step(self, actions):
        action_mean = np.mean(actions, axis=0)
        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)
        coord = -np.mean([np.linalg.norm(a - action_mean) for a in actions])
        return self.state, -dist + 0.5 * coord


def run_training(lambda_flex, n_agents=4, n_episodes=100, n_steps=200):
    """Train agents with given flexibility regularization."""
    agents = [RegularizedAgent(i, lambda_flex) for i in range(n_agents)]
    network = Network(n_agents)
    env = Environment(n_agents)

    episode_rewards = []
    episode_flexibilities = []

    for ep in range(n_episodes):
        state = env.reset()
        for a in agents:
            a.obs_history = []
            a.action_history = []
            a.rewards = []

        total_reward = 0
        for step in range(n_steps):
            observations = [state + np.random.randn(10) * 0.1 for _ in range(n_agents)]
            messages = [a.create_message(o) for a, o in zip(agents, observations)]
            received = network.exchange(messages)
            actions = [a.act(o, m) for a, o, m in zip(agents, observations, received)]
            state, reward = env.step(actions)
            for a in agents:
                a.rewards.append(reward)
            total_reward += reward

        for a in agents:
            a.update()

        episode_rewards.append(total_reward)
        mean_flex = np.mean([a.get_flexibility() for a in agents])
        episode_flexibilities.append(mean_flex)

    return episode_rewards, episode_flexibilities


def main():
    print("\n" + "=" * 70)
    print("TRACK F1: FLEXIBILITY REGULARIZATION (CAUSAL TEST)")
    print("=" * 70)

    np.random.seed(42)

    # Test different regularization strengths
    lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    n_runs = 5

    print(f"\nTesting λ ∈ {lambdas} with {n_runs} runs each...")

    results = {}

    for lam in lambdas:
        all_final_rewards = []
        all_final_flex = []

        for run in range(n_runs):
            np.random.seed(run * 1000 + int(lam * 100))
            rewards, flexibilities = run_training(lam, n_episodes=80)

            # Final performance (last 20 episodes)
            final_reward = np.mean(rewards[-20:])
            final_flex = np.mean(flexibilities[-20:])

            all_final_rewards.append(final_reward)
            all_final_flex.append(final_flex)

        results[lam] = {
            'reward_mean': np.mean(all_final_rewards),
            'reward_std': np.std(all_final_rewards),
            'flex_mean': np.mean(all_final_flex),
            'flex_std': np.std(all_final_flex)
        }

        print(f"  λ = {lam:.2f}: reward = {results[lam]['reward_mean']:.1f} ± {results[lam]['reward_std']:.1f}, "
              f"flex = {results[lam]['flex_mean']:.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Find optimal lambda
    best_lam = max(results.keys(), key=lambda l: results[l]['reward_mean'])
    baseline = results[0.0]['reward_mean']
    best_reward = results[best_lam]['reward_mean']
    improvement = best_reward - baseline

    print(f"\nBaseline (λ=0): {baseline:.1f}")
    print(f"Best (λ={best_lam}): {best_reward:.1f}")
    print(f"Improvement: {improvement:+.1f} ({100*improvement/abs(baseline):+.1f}%)")

    # Check if flexibility correlates with reward across lambdas
    flex_values = [results[l]['flex_mean'] for l in lambdas]
    reward_values = [results[l]['reward_mean'] for l in lambdas]
    r, p = stats.pearsonr(flex_values, reward_values)

    print(f"\nFlexibility ↔ Reward across λ: r = {r:+.3f}, p = {p:.4f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if improvement > 0 and best_lam > 0:
        print(f"\n✓ CAUSAL EVIDENCE: Imposing flexibility (λ={best_lam}) improves coordination")
        print(f"  Improvement: {improvement:+.1f} reward units")
        if r > 0.5:
            print(f"  Strong correlation: r = {r:+.3f}")
    elif improvement < 0:
        print(f"\n✗ Flexibility regularization HURTS performance")
        print(f"  Best is baseline (λ=0)")
    else:
        print(f"\n→ No clear effect of flexibility regularization")

    # Check for inverted-U
    mid_lambdas = [0.05, 0.1]
    mid_reward = np.mean([results[l]['reward_mean'] for l in mid_lambdas])
    high_reward = results[0.5]['reward_mean']

    if mid_reward > baseline and mid_reward > high_reward:
        print(f"\n  Pattern: INVERTED-U (moderate λ best)")
    elif high_reward > mid_reward > baseline:
        print(f"\n  Pattern: MONOTONIC (more flexibility = better)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_f1_regularization_{timestamp}.npz"
    np.savez(filename, results=results, lambdas=lambdas)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
