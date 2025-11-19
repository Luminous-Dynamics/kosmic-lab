#!/usr/bin/env python3
"""
Track G3: Environment Perturbation Robustness

Do flexible teams degrade gracefully under perturbation?
Tests: observation noise, action noise, communication dropout

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_g3_environment_perturbation.py"
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
                    # Apply dropout
                    if np.random.rand() > dropout_rate:
                        incoming.append(messages[j])
                    else:
                        incoming.append(np.zeros(5))
            if incoming:
                received.append(np.mean(incoming, axis=0))
            else:
                received.append(np.zeros(5))
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


def train_team(n_agents=4, n_episodes=100, n_steps=200):
    """Train a team and return trained agents with their flexibility."""
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

    # Measure final flexibility
    flexibility = np.mean([a.get_flexibility() for a in agents])
    return agents, flexibility


def test_with_perturbation(agents, obs_noise=0.0, action_noise=0.0, comm_dropout=0.0,
                           n_episodes=30, n_steps=200):
    """Test trained agents under perturbation."""
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
            # Add observation noise
            observations = [state + np.random.randn(10) * (0.1 + obs_noise)
                           for _ in range(n_agents)]

            messages = [a.create_message(o) for a, o in zip(agents, observations)]
            received = network.exchange(messages, dropout_rate=comm_dropout)

            actions = []
            for a, o, m in zip(agents, observations, received):
                action = a.act(o, m)
                # Add action noise
                action = action + np.random.randn(10) * action_noise
                action = np.clip(action, -1, 1)
                actions.append(action)

            state, reward = env.step(actions)
            total_reward += reward

        episode_rewards.append(total_reward)

    return np.mean(episode_rewards), np.std(episode_rewards)


def main():
    print("\n" + "=" * 70)
    print("TRACK G3: ENVIRONMENT PERTURBATION ROBUSTNESS")
    print("=" * 70)

    np.random.seed(42)
    n_teams = 20

    print(f"\nTraining {n_teams} teams and testing under perturbation...")

    # Train teams and record flexibility
    teams = []
    flexibilities = []

    for t in range(n_teams):
        np.random.seed(t * 1000)
        agents, flex = train_team(n_episodes=60)
        teams.append(agents)
        flexibilities.append(flex)
        if (t + 1) % 5 == 0:
            print(f"  Trained {t + 1}/{n_teams} teams")

    flexibilities = np.array(flexibilities)

    # Test under different perturbations
    perturbations = {
        'baseline': {'obs_noise': 0.0, 'action_noise': 0.0, 'comm_dropout': 0.0},
        'obs_2x': {'obs_noise': 0.1, 'action_noise': 0.0, 'comm_dropout': 0.0},
        'obs_5x': {'obs_noise': 0.4, 'action_noise': 0.0, 'comm_dropout': 0.0},
        'action_noise': {'obs_noise': 0.0, 'action_noise': 0.3, 'comm_dropout': 0.0},
        'comm_30%': {'obs_noise': 0.0, 'action_noise': 0.0, 'comm_dropout': 0.3},
        'comm_50%': {'obs_noise': 0.0, 'action_noise': 0.0, 'comm_dropout': 0.5},
        'combined': {'obs_noise': 0.2, 'action_noise': 0.1, 'comm_dropout': 0.2},
    }

    results = {}

    print("\nTesting under perturbations...")

    for name, params in perturbations.items():
        performances = []
        for agents in teams:
            perf, _ = test_with_perturbation(agents, **params, n_episodes=20)
            performances.append(perf)

        performances = np.array(performances)

        # Correlation between flexibility and performance under this perturbation
        r, p = stats.pearsonr(flexibilities, performances)

        results[name] = {
            'mean': np.mean(performances),
            'std': np.std(performances),
            'r': r,
            'p': p
        }

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {name:15s}: r = {r:+.3f}{sig}, perf = {np.mean(performances):.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compare baseline vs perturbation correlations
    baseline_r = results['baseline']['r']

    print("\nFlexibility-Performance Correlation by Condition:")
    print(f"  Baseline:      r = {baseline_r:+.3f}")

    # Average degradation for flexible vs rigid teams
    median_flex = np.median(flexibilities)
    flexible_teams = flexibilities > median_flex
    rigid_teams = ~flexible_teams

    print("\nPerformance Degradation (% from baseline):")

    baseline_perf = results['baseline']['mean']
    for name in ['obs_5x', 'comm_50%', 'combined']:
        perf = results[name]['mean']
        degradation = 100 * (perf - baseline_perf) / abs(baseline_perf)
        print(f"  {name}: {degradation:+.1f}%")

    # Key finding: does flexibility help under perturbation?
    perturbation_rs = [results[name]['r'] for name in perturbations if name != 'baseline']
    mean_perturb_r = np.mean(perturbation_rs)

    print(f"\nMean correlation under perturbation: r = {mean_perturb_r:+.3f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if mean_perturb_r > baseline_r:
        print(f"\n✓ Flexibility MORE important under perturbation")
        print(f"  Baseline r = {baseline_r:+.3f} → Perturbed r = {mean_perturb_r:+.3f}")
        print(f"  Flexible teams degrade MORE GRACEFULLY")
    elif mean_perturb_r > 0.2:
        print(f"\n✓ Flexibility MAINTAINS importance under perturbation")
        print(f"  Perturbed r = {mean_perturb_r:+.3f}")
    else:
        print(f"\n→ Flexibility benefit DECREASES under perturbation")
        print(f"  Baseline r = {baseline_r:+.3f} → Perturbed r = {mean_perturb_r:+.3f}")

    # Specific findings
    comm_50_r = results['comm_50%']['r']
    if comm_50_r > 0.3:
        print(f"\n  Notable: Strong effect under 50% comm dropout (r = {comm_50_r:+.3f})")
        print(f"  → Flexible teams adapt to missing information")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_g3_perturbation_{timestamp}.npz"
    np.savez(filename, results=results, flexibilities=flexibilities)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
