#!/usr/bin/env python3
"""
Track E Quick: Developmental Dynamics (Faster Version)

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_e_quick.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class DevelopingAgent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.policy_weights = np.random.randn(10, 15) * 0.1
        self.obs_history = []
        self.action_history = []
        self.rewards = []
        self.flexibility_trajectory = []

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        mean = self.policy_weights @ combined
        action = np.tanh(mean + np.random.randn(10) * 0.3)
        self.obs_history.append(obs)
        self.action_history.append(action)
        return action

    def get_flexibility(self):
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-30:]).flatten()
        actions = np.array(self.action_history[-30:]).flatten()
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def update(self, learning_rate=0.01):
        if not self.rewards:
            return
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for i in range(min(len(self.obs_history), len(returns))):
            obs = self.obs_history[i]
            action = self.action_history[i]
            ret = returns[i]
            gradient = np.outer(action, np.concatenate([obs, np.zeros(5)]))
            self.policy_weights += learning_rate * ret * gradient[:, :15]

        self.flexibility_trajectory.append(self.get_flexibility())
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


def run_training(n_agents=4, n_episodes=80, n_steps=100, learning_rate=0.01):
    agents = [DevelopingAgent(i) for i in range(n_agents)]
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
            a.update(learning_rate)

        episode_rewards.append(total_reward)
        mean_flex = np.mean([a.flexibility_trajectory[-1] if a.flexibility_trajectory else 0
                            for a in agents])
        episode_flexibilities.append(mean_flex)

    return episode_rewards, episode_flexibilities


def main():
    print("\n" + "=" * 70)
    print("TRACK E QUICK: DEVELOPMENTAL DYNAMICS")
    print("=" * 70)

    np.random.seed(42)

    # Experiment 1: Development curve
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: FLEXIBILITY DEVELOPMENT")
    print("-" * 70)

    rewards, flexibilities = run_training(n_episodes=80)

    early = np.mean(flexibilities[:20])
    late = np.mean(flexibilities[-20:])
    print(f"\n  Early (0-20):  {early:+.3f}")
    print(f"  Late (60-80):  {late:+.3f}")
    print(f"  Change: {late - early:+.3f}")

    # Experiment 2: Early predicts final
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: EARLY → FINAL PREDICTION")
    print("-" * 70)

    n_teams = 20
    early_flex = []
    final_perf = []

    for team in range(n_teams):
        np.random.seed(team * 1000)
        rewards, flexibilities = run_training(n_episodes=60)
        early_flex.append(np.mean(flexibilities[:15]))
        final_perf.append(np.mean(rewards[-15:]))

    r, p = stats.pearsonr(early_flex, final_perf)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"\n  Early flex → Final reward: r = {r:+.3f}, p = {p:.4f}{sig}")

    # Experiment 3: Optimal level
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: OPTIMAL FLEXIBILITY LEVEL")
    print("-" * 70)

    n_teams = 25
    final_flex = []
    final_rewards = []

    for team in range(n_teams):
        np.random.seed(team * 2000)
        rewards, flexibilities = run_training(n_episodes=100)
        final_flex.append(np.mean(flexibilities[-20:]))
        final_rewards.append(np.mean(rewards[-20:]))

    flex_arr = np.array(final_flex)
    perf_arr = np.array(final_rewards)
    sorted_idx = np.argsort(flex_arr)
    thirds = np.array_split(sorted_idx, 3)

    low = np.mean(perf_arr[thirds[0]])
    mid = np.mean(perf_arr[thirds[1]])
    high = np.mean(perf_arr[thirds[2]])

    print(f"\n  Low flexibility:  {low:.1f}")
    print(f"  Mid flexibility:  {mid:.1f}")
    print(f"  High flexibility: {high:.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\n1. Flexibility development: {early:+.3f} → {late:+.3f}")
    print(f"2. Predictive power: r = {r:+.3f}")
    print(f"3. Optimal level: ", end="")
    if mid > max(low, high):
        print("MODERATE (inverted-U)")
    elif high > mid > low:
        print("MORE IS BETTER (monotonic)")
    else:
        print("UNCLEAR pattern")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_e_quick_{timestamp}.npz"
    np.savez(filename, early_flex=early_flex, final_perf=final_perf,
             final_flex=final_flex, final_rewards=final_rewards)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
