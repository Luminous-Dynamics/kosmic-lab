#!/usr/bin/env python3
"""
Track F1 Curriculum: Annealing Flexibility Over Training

Test if annealing λ (high early, low late) works better than constant.
Hypothesis: Flexible early for exploration, rigid late for exploitation.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_f1_curriculum.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class CurriculumAgent:
    def __init__(self, agent_id, schedule='constant', lambda_init=0.1):
        self.id = agent_id
        self.schedule = schedule
        self.lambda_init = lambda_init
        self.policy_weights = np.random.randn(10, 15) * 0.1
        self.obs_history = []
        self.action_history = []
        self.rewards = []
        self.episode = 0

    def get_lambda(self, episode, max_episodes):
        if self.schedule == 'constant':
            return self.lambda_init
        elif self.schedule == 'anneal':
            # Linear decay from lambda_init to 0
            return self.lambda_init * (1 - episode / max_episodes)
        elif self.schedule == 'warmup':
            # Start at 0, increase to lambda_init
            return self.lambda_init * (episode / max_episodes)
        elif self.schedule == 'cosine':
            # Cosine annealing
            return self.lambda_init * 0.5 * (1 + np.cos(np.pi * episode / max_episodes))

    def act(self, obs, messages, episode, max_episodes):
        combined = np.concatenate([obs, messages])
        mean = self.policy_weights @ combined

        lam = self.get_lambda(episode, max_episodes)
        noise_scale = 0.3 + lam * 0.5
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

    def update(self, learning_rate=0.01, episode=0, max_episodes=100):
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

        lam = self.get_lambda(episode, max_episodes)

        for i in range(min(len(self.obs_history), len(returns))):
            obs = self.obs_history[i]
            action = self.action_history[i]
            ret = returns[i]

            if i > 0:
                prev_action = self.action_history[i-1]
                entropy_bonus = np.sum((action - prev_action)**2)
            else:
                entropy_bonus = 0

            modified_return = ret + lam * entropy_bonus
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


def run_training(schedule, lambda_init=0.1, n_agents=4, n_episodes=80, n_steps=150):
    agents = [CurriculumAgent(i, schedule, lambda_init) for i in range(n_agents)]
    network = Network(n_agents)
    env = Environment(n_agents)

    episode_rewards = []

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
            actions = [a.act(o, m, ep, n_episodes) for a, o, m in zip(agents, observations, received)]
            state, reward = env.step(actions)
            for a in agents:
                a.rewards.append(reward)
            total_reward += reward

        for a in agents:
            a.update(episode=ep, max_episodes=n_episodes)

        episode_rewards.append(total_reward)

    return episode_rewards


def main():
    print("\n" + "=" * 70)
    print("TRACK F1 CURRICULUM: ANNEALING FLEXIBILITY")
    print("=" * 70)

    np.random.seed(42)

    schedules = ['constant', 'anneal', 'warmup', 'cosine']
    n_runs = 5

    print(f"\nTesting schedules with {n_runs} runs each...")

    results = {}

    for schedule in schedules:
        all_final_rewards = []

        for run in range(n_runs):
            np.random.seed(run * 1000)
            rewards = run_training(schedule, lambda_init=0.2)
            final_reward = np.mean(rewards[-20:])
            all_final_rewards.append(final_reward)

        results[schedule] = {
            'mean': np.mean(all_final_rewards),
            'std': np.std(all_final_rewards)
        }

        print(f"  {schedule:10s}: {results[schedule]['mean']:.1f} ± {results[schedule]['std']:.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best = max(schedules, key=lambda s: results[s]['mean'])
    worst = min(schedules, key=lambda s: results[s]['mean'])

    print(f"\nBest: {best} ({results[best]['mean']:.1f})")
    print(f"Worst: {worst} ({results[worst]['mean']:.1f})")

    improvement = results[best]['mean'] - results['constant']['mean']
    print(f"Improvement over constant: {improvement:+.1f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if best == 'anneal':
        print("\n✓ ANNEAL wins: Flexible early, rigid late is optimal")
        print("  Supports: Explore early, exploit late")
    elif best == 'warmup':
        print("\n✓ WARMUP wins: Rigid early, flexible late is optimal")
        print("  Suggests: Learn basics first, then adapt")
    elif best == 'cosine':
        print("\n✓ COSINE wins: Smooth annealing is optimal")
    else:
        print("\n→ CONSTANT is best (no curriculum benefit)")

    if abs(improvement) > 50:
        print(f"\n  Curriculum effect is SUBSTANTIAL ({improvement:+.1f})")
    else:
        print(f"\n  Curriculum effect is modest ({improvement:+.1f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_f1_curriculum_{timestamp}.npz"
    np.savez(filename, results=results)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
