#!/usr/bin/env python3
"""
🧒 Track E Re-Validation: Developmental Learning with Full K-Index

Original Finding: K-Index grows from 0.3 to 1.357 during learning
Original Claim: "Evidence that consciousness emerges through learning"

Question: Does Full K also grow during learning?
If YES → The finding is valid, just measured wrong
If NO → The "emergence" was actually increasing rigidity
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from collections import Counter


class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = 1.1
        self.length = 0.5
        self.polemass_length = 0.05
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 0.2095
        self.x_threshold = 2.4
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4/3 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        self.state = np.array([
            x + self.tau * x_dot,
            x_dot + self.tau * xacc,
            theta + self.tau * theta_dot,
            theta_dot + self.tau * thetaacc
        ])
        done = abs(self.state[0]) > self.x_threshold or abs(self.state[2]) > self.theta_threshold
        return self.state.copy(), 0 if done else 1, done


class DevelopmentalAgent:
    """Agent that learns through simple Q-learning (developmental paradigm)."""

    def __init__(self, lr=0.1, gamma=0.99, epsilon=0.3):
        # Simple linear function approximation
        self.W = np.random.randn(2, 4) * 0.1
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_values(self, state):
        return self.W @ state

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q_values)

        # TD error
        td_error = target - q_values[action]

        # Update weights
        self.W[action] += self.lr * td_error * state

        # Decay epsilon
        self.epsilon *= 0.9995
        self.epsilon = max(0.01, self.epsilon)

        return td_error


# K-Index computation functions
def compute_h1(hidden_states):
    if len(hidden_states) < 5:
        return 0.5
    states = np.array(hidden_states)
    if states.ndim == 1 or states.shape[1] < 2:
        return 0.5
    correlations = []
    for i in range(min(states.shape[1], 4)):
        for j in range(i+1, min(states.shape[1], 4)):
            if np.std(states[:, i]) > 1e-10 and np.std(states[:, j]) > 1e-10:
                r = np.corrcoef(states[:, i], states[:, j])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))
    return float(np.mean(correlations)) if correlations else 0.5


def compute_h2(actions):
    if len(actions) < 5:
        return 0.5
    counts = Counter(actions)
    if len(counts) <= 1:
        return 0.0
    probs = np.array([c / len(actions) for c in counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    return float(h / np.log(len(counts)))


def compute_h4(q_values_history):
    if len(q_values_history) < 10:
        return 0.5
    arr = np.array(q_values_history)
    n_bins = min(5, len(arr) // 5)
    if n_bins < 2:
        return 0.5
    entropies = []
    for dim in range(arr.shape[1]):
        col = arr[:, dim]
        if np.std(col) < 1e-10:
            continue
        hist, _ = np.histogram(col, bins=n_bins)
        hist = hist + 1e-10
        probs = hist / hist.sum()
        entropies.append(-np.sum(probs * np.log(probs)))
    return float(np.mean(entropies) / np.log(n_bins)) if entropies else 0.5


def compute_h5(q_values_history):
    if len(q_values_history) < 10:
        return 0.5
    series = np.array(q_values_history)
    if series.shape[1] < 2:
        return 0.5
    influences = []
    for i in range(series.shape[1]):
        for j in range(i+1, series.shape[1]):
            if np.std(series[:-1, i]) > 1e-10 and np.std(series[1:, j]) > 1e-10:
                c_ij = abs(np.corrcoef(series[:-1, i], series[1:, j])[0, 1])
                c_ji = abs(np.corrcoef(series[:-1, j], series[1:, i])[0, 1])
                if not (np.isnan(c_ij) or np.isnan(c_ji)):
                    influences.append(2 * min(c_ij, c_ji) / (c_ij + c_ji + 1e-10))
    return float(np.mean(influences)) if influences else 0.5


def compute_h6(obs_norms):
    if len(obs_norms) < 6:
        return 0.5
    mid = len(obs_norms) // 2
    out = np.array(obs_norms[:mid])
    inc = np.array(obs_norms[mid:])
    min_len = min(len(out), len(inc))
    out, inc = out[:min_len], inc[:min_len]
    out_p = (np.abs(out) + 1e-10) / (np.abs(out).sum() + 1e-10)
    in_p = (np.abs(inc) + 1e-10) / (np.abs(inc).sum() + 1e-10)
    m = 0.5 * (out_p + in_p)
    js = 0.5 * (np.sum(out_p * np.log(out_p / (m + 1e-10) + 1e-10)) +
                np.sum(in_p * np.log(in_p / (m + 1e-10) + 1e-10)))
    return float(1.0 - np.sqrt(np.clip(js, 0, 1)))


def compute_h7(metric_history):
    if len(metric_history) < 10:
        return 0.0
    recent = metric_history[-50:] if len(metric_history) > 50 else metric_history
    times = np.arange(len(recent))
    values = np.array(recent)
    if np.std(values) < 1e-10:
        return 0.0
    slope = np.polyfit(times / (times.max() + 1e-10), values, deg=1)[0]
    return float(np.tanh(slope / (np.std(values) + 1e-10)))


def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate agent performance and compute K-indices."""
    total_reward = 0
    obs_norms, act_norms = [], []
    actions = []
    q_values_history = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(500):
            q_values = agent.get_q_values(state)
            action = np.argmax(q_values)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(q_values))
            actions.append(action)
            q_values_history.append(q_values.copy())

            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break

        total_reward += episode_reward

    avg_performance = total_reward / n_episodes

    # Simple K
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        simple_k = 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        simple_k = 0.0

    # Full K (simplified 6 harmonies since we don't have hidden layers)
    h1 = compute_h1(q_values_history)  # Q-value integration
    h2 = compute_h2(actions)
    h4 = compute_h4(q_values_history)
    h5 = compute_h5(q_values_history)
    h6 = compute_h6(obs_norms)
    h7 = 0.0  # Need history across checkpoints

    full_k = (h1 + h2 + 0.5 + h4 + h5 + h6 + h7) / 7.0

    return {
        'performance': avg_performance,
        'simple_k': simple_k,
        'full_k': full_k,
        'h1': h1, 'h2': h2, 'h4': h4, 'h5': h5, 'h6': h6
    }


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧒 Track E Re-Validation: Developmental Learning             ║")
    print("║                                                                ║")
    print("║  Question: Does Full K grow during learning like Simple K?    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)
    env = CartPoleEnv()
    agent = DevelopmentalAgent(lr=0.1, gamma=0.99, epsilon=0.5)

    # Training with checkpoints
    n_episodes = 500
    checkpoint_interval = 50

    developmental_history = []

    print("Training developmental agent...\n")
    print("Episode │  Perf  │ Simple K │ Full K │  H2   │ Epsilon")
    print("────────┼────────┼──────────┼────────┼───────┼─────────")

    episode = 0
    while episode < n_episodes:
        # Train for checkpoint_interval episodes
        for _ in range(checkpoint_interval):
            state = env.reset()

            for _ in range(500):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

            episode += 1

        # Evaluate at checkpoint
        metrics = evaluate_agent(agent, env, n_episodes=10)
        metrics['episode'] = episode
        metrics['epsilon'] = agent.epsilon
        developmental_history.append(metrics)

        print(f"  {episode:4d}  │ {metrics['performance']:6.1f} │ {metrics['simple_k']:8.4f} │ "
              f"{metrics['full_k']:6.3f} │ {metrics['h2']:5.3f} │ {agent.epsilon:.3f}")

    # Analyze developmental trajectory
    print("\n" + "═" * 70)
    print("\n📊 DEVELOPMENTAL ANALYSIS:\n")

    # Compare start vs end
    start = developmental_history[0]
    end = developmental_history[-1]

    print("Metric      │    Start    │     End     │   Change")
    print("────────────┼─────────────┼─────────────┼───────────")

    for metric in ['performance', 'simple_k', 'full_k', 'h2']:
        start_val = start[metric]
        end_val = end[metric]
        change = ((end_val / (start_val + 0.001)) - 1) * 100
        print(f"{metric:11s} │ {start_val:11.3f} │ {end_val:11.3f} │ {change:+8.1f}%")

    # Compute correlations with episode number (development)
    episodes = [d['episode'] for d in developmental_history]

    print("\n📈 CORRELATION WITH DEVELOPMENT (Episode Number):\n")

    correlations = {}
    for metric in ['performance', 'simple_k', 'full_k', 'h2']:
        values = [d[metric] for d in developmental_history]
        try:
            r, p = pearsonr(episodes, values)
            correlations[metric] = (r, p)
        except:
            correlations[metric] = (0.0, 1.0)

    print("  Metric      │  r      │ p-value │ Grows with Learning?")
    print("  ────────────┼─────────┼─────────┼─────────────────────")

    for metric, (r, p) in sorted(correlations.items(), key=lambda x: -x[1][0]):
        grows = "✅ YES" if r > 0.3 and p < 0.05 else ("❌ NO" if r < -0.3 else "~ Flat")
        print(f"  {metric:11s} │ {r:+.4f} │ {p:.4f}  │ {grows}")

    # Key finding
    print("\n🔑 KEY FINDING:\n")

    perf_r = correlations['performance'][0]
    simple_r = correlations['simple_k'][0]
    full_r = correlations['full_k'][0]
    h2_r = correlations['h2'][0]

    if full_r > 0.3:
        print("  ✅ Full K GROWS with learning!")
        print(f"     Performance: r = {perf_r:+.3f}")
        print(f"     Full K:      r = {full_r:+.3f}")
        print(f"     H2:          r = {h2_r:+.3f}")
        print("\n  → Track E finding is VALID with Full K!")
        print("  → 'Coherence emerges through learning' is correct!")
        conclusion = "VALID"
    elif simple_r > 0.3 and full_r < 0.3:
        print("  ⚠️  Simple K grows but Full K does not!")
        print(f"     Simple K: r = {simple_r:+.3f}")
        print(f"     Full K:   r = {full_r:+.3f}")
        print("\n  → Track E measured RIGIDITY growth, not coherence")
        conclusion = "INVALID"
    else:
        print(f"  🔶 Mixed results:")
        print(f"     Simple K: r = {simple_r:+.3f}")
        print(f"     Full K:   r = {full_r:+.3f}")
        conclusion = "MIXED"

    # Save results
    Path('logs/track_e_revalidation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_e_revalidation/results.json', 'w') as f:
        json.dump({
            'developmental_history': developmental_history,
            'correlations': {k: {'r': float(v[0]), 'p': float(v[1])}
                           for k, v in correlations.items()},
            'conclusion': conclusion,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_e_revalidation/")
    print('\n"Does coherence truly emerge through learning?" 💚\n')


if __name__ == '__main__':
    main()
