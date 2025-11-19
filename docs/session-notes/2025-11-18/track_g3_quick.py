#!/usr/bin/env python3
"""
Track G3 Quick: Environment Perturbation (10 teams)

Faster version with 10 teams instead of 20.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_g3_quick.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.policy_weights = np.random.randn(10, 15) * 0.1
        self.obs_history = []
        self.action_history = []
        self.rewards = []

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined + np.random.randn(10) * 0.3)
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

        self.obs_history = []
        self.action_history = []
        self.rewards = []

    def create_message(self, obs):
        return obs[:5]


class Network:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def exchange(self, messages, dropout_rate=0.0):
        received = []
        for i in range(self.n_agents):
            incoming = []
            for j in range(self.n_agents):
                if self.adj[i, j] > 0:
                    if np.random.rand() > dropout_rate:
                        incoming.append(messages[j])
                    else:
                        incoming.append(np.zeros(5))
            received.append(np.mean(incoming, axis=0) if incoming else np.zeros(5))
        return received


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


def train_team(n_agents=4, n_episodes=50, n_steps=150):
    agents = [Agent(i) for i in range(n_agents)]
    network = Network(n_agents)
    env = Environment(n_agents)

    for ep in range(n_episodes):
        state = env.reset()
        for a in agents:
            a.obs_history = []
            a.action_history = []
            a.rewards = []

        for step in range(n_steps):
            observations = [state + np.random.randn(10) * 0.1 for _ in range(n_agents)]
            messages = [a.create_message(o) for a, o in zip(agents, observations)]
            received = network.exchange(messages)
            actions = [a.act(o, m) for a, o, m in zip(agents, observations, received)]
            state, reward = env.step(actions)
            for a in agents:
                a.rewards.append(reward)

        for a in agents:
            a.update()

    flexibility = np.mean([a.get_flexibility() for a in agents])
    return agents, flexibility


def test_with_perturbation(agents, obs_noise=0.0, comm_dropout=0.0, n_episodes=20, n_steps=150):
    n_agents = len(agents)
    network = Network(n_agents)
    env = Environment(n_agents)

    episode_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        for a in agents:
            a.obs_history = []
            a.action_history = []

        total_reward = 0
        for step in range(n_steps):
            observations = [state + np.random.randn(10) * (0.1 + obs_noise)
                           for _ in range(n_agents)]
            messages = [a.create_message(o) for a, o in zip(agents, observations)]
            received = network.exchange(messages, dropout_rate=comm_dropout)
            actions = [a.act(o, m) for a, o, m in zip(agents, observations, received)]
            state, reward = env.step(actions)
            total_reward += reward

        episode_rewards.append(total_reward)

    return np.mean(episode_rewards)


def main():
    print("\n" + "=" * 70)
    print("TRACK G3 QUICK: ENVIRONMENT PERTURBATION (10 TEAMS)")
    print("=" * 70)

    np.random.seed(42)
    n_teams = 10

    print(f"\nTraining {n_teams} teams...")

    teams = []
    flexibilities = []

    for t in range(n_teams):
        np.random.seed(t * 1000)
        agents, flex = train_team(n_episodes=50)
        teams.append(agents)
        flexibilities.append(flex)

    flexibilities = np.array(flexibilities)
    print(f"  Flexibility range: [{flexibilities.min():.3f}, {flexibilities.max():.3f}]")

    perturbations = {
        'baseline': {'obs_noise': 0.0, 'comm_dropout': 0.0},
        'obs_5x': {'obs_noise': 0.4, 'comm_dropout': 0.0},
        'comm_50%': {'obs_noise': 0.0, 'comm_dropout': 0.5},
        'combined': {'obs_noise': 0.2, 'comm_dropout': 0.3},
    }

    print("\nTesting perturbations...")

    results = {}

    for name, params in perturbations.items():
        performances = []
        for agents in teams:
            perf = test_with_perturbation(agents, **params)
            performances.append(perf)

        performances = np.array(performances)
        r, p = stats.pearsonr(flexibilities, performances)

        results[name] = {'r': r, 'p': p, 'mean': np.mean(performances)}
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {name:12s}: r = {r:+.3f}{sig}, perf = {np.mean(performances):.1f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    baseline_r = results['baseline']['r']
    perturb_rs = [results[k]['r'] for k in results if k != 'baseline']
    mean_perturb_r = np.mean(perturb_rs)

    print(f"\nBaseline r = {baseline_r:+.3f}")
    print(f"Mean perturbed r = {mean_perturb_r:+.3f}")

    if mean_perturb_r > baseline_r:
        print("\n✓ Flexibility MORE important under perturbation")
    elif mean_perturb_r > 0.2:
        print("\n✓ Flexibility maintains importance under perturbation")
    else:
        print("\n→ Results inconclusive (need more teams)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_g3_quick_{timestamp}.npz"
    np.savez(filename, results=results, flexibilities=flexibilities)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
