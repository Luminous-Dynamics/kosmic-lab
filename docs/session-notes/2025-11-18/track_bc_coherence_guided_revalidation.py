#!/usr/bin/env python3
"""
🎯 Track B+C Re-Validation: Coherence-Guided Control with Full K-Index

Original Finding: 63% improvement with K-index feedback
Original Claim: "Coherence-guided control enhances performance"

Question: Does Full K also improve performance when used as feedback?
If YES → The finding is valid, just measured wrong
If NO → The "improvement" came from optimizing rigidity
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


class CoherenceGuidedAgent:
    """Agent that uses K-Index as part of its learning signal."""

    def __init__(self, lr=0.1, gamma=0.99, epsilon=0.3, k_weight=0.0, use_full_k=False):
        self.W = np.random.randn(2, 4) * 0.1
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_weight = k_weight  # How much to weight K-Index in reward
        self.use_full_k = use_full_k  # Use Full K or Simple K

    def get_q_values(self, state):
        return self.W @ state

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done, k_bonus=0.0):
        """Learn with optional K-Index bonus in reward."""
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        # Augmented reward with K-Index feedback
        augmented_reward = reward + self.k_weight * k_bonus

        if done:
            target = augmented_reward
        else:
            target = augmented_reward + self.gamma * np.max(next_q_values)

        td_error = target - q_values[action]
        self.W[action] += self.lr * td_error * state

        self.epsilon *= 0.9995
        self.epsilon = max(0.01, self.epsilon)

        return td_error


# K-Index computation functions (same as track_e)
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


def compute_simple_k(obs_norms, act_norms):
    """Simple K-Index: correlation-based."""
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


def compute_full_k(obs_norms, act_norms, actions, q_values_history):
    """Full 7-Harmony K-Index."""
    h1 = compute_h1(q_values_history)
    h2 = compute_h2(actions)
    h3 = 0.5  # Prediction accuracy placeholder
    h4 = compute_h4(q_values_history)
    h5 = compute_h5(q_values_history)
    h6 = compute_h6(obs_norms)
    h7 = 0.0  # Growth rate needs history

    return (h1 + h2 + h3 + h4 + h5 + h6 + h7) / 7.0, h2


def train_and_evaluate(env, k_weight, use_full_k, n_episodes=300, seed=None):
    """Train agent with given K-weight and metric type."""
    if seed is not None:
        np.random.seed(seed)

    agent = CoherenceGuidedAgent(
        lr=0.1, gamma=0.99, epsilon=0.5,
        k_weight=k_weight, use_full_k=use_full_k
    )

    performance_history = []
    k_history = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        obs_norms, act_norms = [], []
        actions = []
        q_values_history = []

        for step in range(500):
            q_values = agent.get_q_values(state)
            action = agent.act(state)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(q_values))
            actions.append(action)
            q_values_history.append(q_values.copy())

            next_state, reward, done = env.step(action)

            # Compute K-bonus based on metric type
            if len(obs_norms) > 10:
                if use_full_k:
                    k_val, _ = compute_full_k(obs_norms, act_norms, actions, q_values_history)
                else:
                    k_val = compute_simple_k(obs_norms, act_norms)
                k_bonus = k_val - 0.5  # Centered around baseline
            else:
                k_bonus = 0.0

            agent.learn(state, action, reward, next_state, done, k_bonus)
            episode_reward += reward
            state = next_state

            if done:
                break

        performance_history.append(episode_reward)

        # Compute final K values for episode
        if use_full_k:
            k_val, _ = compute_full_k(obs_norms, act_norms, actions, q_values_history)
        else:
            k_val = compute_simple_k(obs_norms, act_norms)
        k_history.append(k_val)

    # Return average of last 50 episodes
    final_perf = np.mean(performance_history[-50:])
    final_k = np.mean(k_history[-50:])

    return final_perf, final_k, performance_history, k_history


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🎯 Track B+C Re-Validation: Coherence-Guided Control         ║")
    print("║                                                                ║")
    print("║  Question: Does Full K feedback also improve performance?      ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    env = CartPoleEnv()

    # Experiment configurations
    configs = [
        # (name, k_weight, use_full_k)
        ("Baseline (no K)", 0.0, False),
        ("Simple K (w=0.1)", 0.1, False),
        ("Simple K (w=0.5)", 0.5, False),
        ("Full K (w=0.1)", 0.1, True),
        ("Full K (w=0.5)", 0.5, True),
    ]

    n_trials = 5
    results = {}

    print("Running experiments (5 trials each)...\n")
    print("Configuration      │  Perf (mean±std)  │  K-Index (mean)  │ vs Baseline")
    print("───────────────────┼───────────────────┼──────────────────┼────────────")

    baseline_perf = None

    for name, k_weight, use_full_k in configs:
        perfs = []
        ks = []

        for trial in range(n_trials):
            perf, k_val, _, _ = train_and_evaluate(
                env, k_weight, use_full_k,
                n_episodes=300, seed=42 + trial
            )
            perfs.append(perf)
            ks.append(k_val)

        mean_perf = np.mean(perfs)
        std_perf = np.std(perfs)
        mean_k = np.mean(ks)

        if name == "Baseline (no K)":
            baseline_perf = mean_perf
            improvement = "—"
        else:
            pct_change = ((mean_perf / baseline_perf) - 1) * 100
            improvement = f"{pct_change:+.1f}%"

        results[name] = {
            'perf_mean': float(mean_perf),
            'perf_std': float(std_perf),
            'k_mean': float(mean_k),
            'trials': perfs
        }

        print(f"{name:18s} │ {mean_perf:6.1f} ± {std_perf:5.1f}  │ {mean_k:16.4f} │ {improvement:>10s}")

    print("\n" + "═" * 70)
    print("\n📊 ANALYSIS:\n")

    # Compare Simple K vs Full K feedback
    simple_k_01 = results["Simple K (w=0.1)"]['perf_mean']
    simple_k_05 = results["Simple K (w=0.5)"]['perf_mean']
    full_k_01 = results["Full K (w=0.1)"]['perf_mean']
    full_k_05 = results["Full K (w=0.5)"]['perf_mean']
    baseline = results["Baseline (no K)"]['perf_mean']

    print("  Performance vs Baseline:\n")
    print(f"    Simple K (w=0.1): {((simple_k_01/baseline)-1)*100:+.1f}%")
    print(f"    Simple K (w=0.5): {((simple_k_05/baseline)-1)*100:+.1f}%")
    print(f"    Full K (w=0.1):   {((full_k_01/baseline)-1)*100:+.1f}%")
    print(f"    Full K (w=0.5):   {((full_k_05/baseline)-1)*100:+.1f}%")

    print("\n🔑 KEY FINDING:\n")

    # Determine which K-feedback helps more
    best_simple = max(simple_k_01, simple_k_05)
    best_full = max(full_k_01, full_k_05)

    simple_helps = best_simple > baseline * 1.05
    full_helps = best_full > baseline * 1.05

    if full_helps and simple_helps:
        if best_full > best_simple:
            print("  ✅ Full K feedback improves performance MORE than Simple K!")
            print(f"     Best Full K: {((best_full/baseline)-1)*100:+.1f}%")
            print(f"     Best Simple K: {((best_simple/baseline)-1)*100:+.1f}%")
            print("\n  → Track B+C finding is ENHANCED with Full K!")
            conclusion = "FULL_K_BETTER"
        else:
            print("  ✅ Both improve performance, Simple K slightly better")
            print(f"     Best Simple K: {((best_simple/baseline)-1)*100:+.1f}%")
            print(f"     Best Full K: {((best_full/baseline)-1)*100:+.1f}%")
            print("\n  → Track B+C finding valid, but Full K also works")
            conclusion = "BOTH_WORK"
    elif full_helps:
        print("  ✅ Full K feedback improves performance!")
        print("  ⚠️  Simple K feedback does NOT help")
        print(f"     Full K improvement: {((best_full/baseline)-1)*100:+.1f}%")
        print("\n  → Track B+C finding INVALID with Simple K, but VALID with Full K!")
        conclusion = "ONLY_FULL_K"
    elif simple_helps:
        print("  ⚠️  Simple K feedback helps, but Full K does NOT")
        print(f"     Simple K improvement: {((best_simple/baseline)-1)*100:+.1f}%")
        print("\n  → Track B+C finding may have measured rigidity guidance")
        conclusion = "ONLY_SIMPLE_K"
    else:
        print("  ❌ Neither K-feedback improves performance significantly")
        print("\n  → Track B+C finding NOT replicated with either metric")
        conclusion = "NEITHER_HELPS"

    # Check if Full K correlates with performance
    print("\n📈 CORRELATION CHECK:\n")

    all_perfs = []
    all_ks_full = []
    all_ks_simple = []

    for trial in range(10):
        # Run without K-feedback to get pure correlation
        perf, _, perf_hist, _ = train_and_evaluate(
            env, 0.0, False, n_episodes=100, seed=100 + trial
        )

        # Get final episode metrics
        state = env.reset()
        obs_norms, act_norms = [], []
        actions = []
        q_values_history = []

        agent = CoherenceGuidedAgent(lr=0.1, gamma=0.99, epsilon=0.1)
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

        all_perfs.append(episode_reward)
        all_ks_simple.append(compute_simple_k(obs_norms, act_norms))
        full_k, _ = compute_full_k(obs_norms, act_norms, actions, q_values_history)
        all_ks_full.append(full_k)

    r_simple, p_simple = pearsonr(all_perfs, all_ks_simple)
    r_full, p_full = pearsonr(all_perfs, all_ks_full)

    print(f"  Simple K vs Performance: r = {r_simple:+.3f} (p = {p_simple:.3f})")
    print(f"  Full K vs Performance:   r = {r_full:+.3f} (p = {p_full:.3f})")

    if r_full > r_simple:
        print("\n  → Full K correlates better with performance")

    # Save results
    Path('logs/track_bc_revalidation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_bc_revalidation/results.json', 'w') as f:
        json.dump({
            'experiment_results': results,
            'correlations': {
                'simple_k': {'r': float(r_simple), 'p': float(p_simple)},
                'full_k': {'r': float(r_full), 'p': float(p_full)}
            },
            'conclusion': conclusion,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_bc_revalidation/")
    print('\n"Does coherence-guided control truly enhance performance?" 💚\n')


if __name__ == '__main__':
    main()
