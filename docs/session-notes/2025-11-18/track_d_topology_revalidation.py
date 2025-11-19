#!/usr/bin/env python3
"""
🕸️ Track D Re-Validation: Topology of Collective Consciousness

Original Finding: Ring topology has 9% better K-Index than other topologies
Original Claim: "Network topology affects emergent coherence"

Question: Does Full K also show ring topology superiority?
If YES → The finding is valid with correct metric
If NO → The topology effect was an artifact of Simple K
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


class CollectiveAgent:
    """Individual agent in a collective."""

    def __init__(self, agent_id, lr=0.1, gamma=0.99, epsilon=0.3):
        self.agent_id = agent_id
        self.W = np.random.randn(2, 4) * 0.1
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_values(self, state):
        return self.W @ state

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.get_q_values(state))

    def learn(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        target = reward if done else reward + self.gamma * np.max(next_q_values)
        td_error = target - q_values[action]
        self.W[action] += self.lr * td_error * state
        self.epsilon *= 0.999
        self.epsilon = max(0.01, self.epsilon)
        return td_error

    def share_knowledge(self, other_agent, weight=0.1):
        """Share knowledge with another agent."""
        self.W = (1 - weight) * self.W + weight * other_agent.W


def create_topology(n_agents, topology_type):
    """Create adjacency matrix for different topologies."""
    adj = np.zeros((n_agents, n_agents))

    if topology_type == "ring":
        # Each agent connected to neighbors
        for i in range(n_agents):
            adj[i, (i + 1) % n_agents] = 1
            adj[i, (i - 1) % n_agents] = 1

    elif topology_type == "star":
        # All connected to central hub (agent 0)
        for i in range(1, n_agents):
            adj[0, i] = 1
            adj[i, 0] = 1

    elif topology_type == "mesh":
        # Fully connected
        adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    elif topology_type == "line":
        # Linear chain
        for i in range(n_agents - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1

    elif topology_type == "isolated":
        # No connections
        pass

    return adj


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


def compute_simple_k(obs_norms, act_norms):
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


def compute_full_k(obs_norms, act_norms, actions, q_values_history):
    h1 = compute_h1(q_values_history)
    h2 = compute_h2(actions)
    h3 = 0.5
    h4 = compute_h4(q_values_history)
    h5 = compute_h5(q_values_history)
    h6 = compute_h6(obs_norms)
    h7 = 0.0
    return (h1 + h2 + h3 + h4 + h5 + h6 + h7) / 7.0, h2


def run_collective_experiment(topology_type, n_agents=5, n_episodes=200, seed=None):
    """Run experiment with a specific topology."""
    if seed is not None:
        np.random.seed(seed)

    env = CartPoleEnv()
    agents = [CollectiveAgent(i) for i in range(n_agents)]
    adj = create_topology(n_agents, topology_type)

    # Collective metrics
    all_obs_norms = []
    all_act_norms = []
    all_actions = []
    all_q_values = []
    total_performance = 0

    for episode in range(n_episodes):
        # Each agent takes a turn
        for agent_idx, agent in enumerate(agents):
            state = env.reset()
            episode_reward = 0

            for step in range(200):
                q_values = agent.get_q_values(state)
                action = agent.act(state)

                all_obs_norms.append(np.linalg.norm(state))
                all_act_norms.append(np.linalg.norm(q_values))
                all_actions.append(action)
                all_q_values.append(q_values.copy())

                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                if done:
                    break

            total_performance += episode_reward

            # Knowledge sharing based on topology
            neighbors = np.where(adj[agent_idx] > 0)[0]
            for neighbor_idx in neighbors:
                agent.share_knowledge(agents[neighbor_idx], weight=0.05)

    avg_performance = total_performance / (n_episodes * n_agents)

    # Compute K-indices
    simple_k = compute_simple_k(all_obs_norms, all_act_norms)
    full_k, h2 = compute_full_k(all_obs_norms, all_act_norms, all_actions, all_q_values)

    return {
        'performance': float(avg_performance),
        'simple_k': float(simple_k),
        'full_k': float(full_k),
        'h2': float(h2)
    }


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🕸️ Track D Re-Validation: Topology of Collective Consciousness║")
    print("║                                                                ║")
    print("║  Question: Does Full K also show ring topology superiority?    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    topologies = ["ring", "star", "mesh", "line", "isolated"]
    n_trials = 5

    results = {}

    print("Running topology experiments (5 trials each)...\n")
    print("Topology   │  Perf (mean±std)  │ Simple K │ Full K │   H2")
    print("───────────┼───────────────────┼──────────┼────────┼───────")

    for topology in topologies:
        perfs = []
        simple_ks = []
        full_ks = []
        h2s = []

        for trial in range(n_trials):
            result = run_collective_experiment(
                topology, n_agents=5, n_episodes=150, seed=42 + trial
            )
            perfs.append(result['performance'])
            simple_ks.append(result['simple_k'])
            full_ks.append(result['full_k'])
            h2s.append(result['h2'])

        results[topology] = {
            'perf_mean': float(np.mean(perfs)),
            'perf_std': float(np.std(perfs)),
            'simple_k_mean': float(np.mean(simple_ks)),
            'full_k_mean': float(np.mean(full_ks)),
            'h2_mean': float(np.mean(h2s))
        }

        print(f"{topology:10s} │ {np.mean(perfs):6.1f} ± {np.std(perfs):5.1f}  │ "
              f"{np.mean(simple_ks):8.4f} │ {np.mean(full_ks):6.3f} │ {np.mean(h2s):6.3f}")

    print("\n" + "═" * 70)
    print("\n📊 TOPOLOGY RANKING:\n")

    # Rank by each metric
    print("  By Performance:")
    by_perf = sorted(results.items(), key=lambda x: -x[1]['perf_mean'])
    for i, (topo, data) in enumerate(by_perf, 1):
        print(f"    {i}. {topo}: {data['perf_mean']:.1f}")

    print("\n  By Simple K:")
    by_simple = sorted(results.items(), key=lambda x: -x[1]['simple_k_mean'])
    for i, (topo, data) in enumerate(by_simple, 1):
        print(f"    {i}. {topo}: {data['simple_k_mean']:.4f}")

    print("\n  By Full K:")
    by_full = sorted(results.items(), key=lambda x: -x[1]['full_k_mean'])
    for i, (topo, data) in enumerate(by_full, 1):
        print(f"    {i}. {topo}: {data['full_k_mean']:.3f}")

    print("\n  By H2 (Diversity):")
    by_h2 = sorted(results.items(), key=lambda x: -x[1]['h2_mean'])
    for i, (topo, data) in enumerate(by_h2, 1):
        print(f"    {i}. {topo}: {data['h2_mean']:.3f}")

    # Check if ring is best
    print("\n🔑 KEY FINDING:\n")

    ring_data = results['ring']
    best_perf_topo = by_perf[0][0]
    best_simple_topo = by_simple[0][0]
    best_full_topo = by_full[0][0]

    # Check original claim: ring has 9% better K
    if best_simple_topo == "ring":
        second_simple = by_simple[1][1]['simple_k_mean']
        ring_simple = ring_data['simple_k_mean']
        simple_advantage = ((ring_simple / second_simple) - 1) * 100
        print(f"  Simple K: Ring IS best ({simple_advantage:+.1f}% vs {by_simple[1][0]})")
    else:
        ring_rank_simple = [t[0] for t in by_simple].index('ring') + 1
        print(f"  Simple K: Ring is #{ring_rank_simple} (best is {best_simple_topo})")

    if best_full_topo == "ring":
        second_full = by_full[1][1]['full_k_mean']
        ring_full = ring_data['full_k_mean']
        full_advantage = ((ring_full / second_full) - 1) * 100
        print(f"  Full K: Ring IS best ({full_advantage:+.1f}% vs {by_full[1][0]})")
        conclusion = "RING_BEST_FULL_K"
    else:
        ring_rank_full = [t[0] for t in by_full].index('ring') + 1
        print(f"  Full K: Ring is #{ring_rank_full} (best is {best_full_topo})")
        conclusion = f"BEST_IS_{best_full_topo.upper()}"

    if best_perf_topo == "ring":
        print(f"  Performance: Ring IS best")
    else:
        ring_rank_perf = [t[0] for t in by_perf].index('ring') + 1
        print(f"  Performance: Ring is #{ring_rank_perf} (best is {best_perf_topo})")

    # Correlation between metrics and performance
    print("\n📈 METRIC-PERFORMANCE CORRELATION:\n")

    perfs = [results[t]['perf_mean'] for t in topologies]
    simple_ks = [results[t]['simple_k_mean'] for t in topologies]
    full_ks = [results[t]['full_k_mean'] for t in topologies]
    h2s = [results[t]['h2_mean'] for t in topologies]

    r_simple, _ = pearsonr(perfs, simple_ks)
    r_full, _ = pearsonr(perfs, full_ks)
    r_h2, _ = pearsonr(perfs, h2s)

    print(f"  Simple K vs Performance: r = {r_simple:+.3f}")
    print(f"  Full K vs Performance:   r = {r_full:+.3f}")
    print(f"  H2 vs Performance:       r = {r_h2:+.3f}")

    # Final interpretation
    print("\n🎯 INTERPRETATION:\n")

    if best_simple_topo == "ring" and best_full_topo != "ring":
        print("  ⚠️  Ring is best by Simple K but NOT by Full K")
        print("  → Original finding was artifact of wrong metric")
        print(f"  → With Full K, {best_full_topo} topology is superior")
    elif best_simple_topo == "ring" and best_full_topo == "ring":
        print("  ✅ Ring is best by BOTH Simple K and Full K!")
        print("  → Track D finding is VALID")
    elif best_full_topo == "ring":
        print("  ✅ Ring is best by Full K (and performance)")
        print("  → Track D finding is VALID with correct metric")
    else:
        print(f"  🔶 {best_full_topo} topology is best by Full K")
        print(f"  → Track D needs reframing around {best_full_topo}")

    # Save results
    Path('logs/track_d_revalidation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_d_revalidation/results.json', 'w') as f:
        json.dump({
            'topology_results': results,
            'rankings': {
                'by_performance': [t[0] for t in by_perf],
                'by_simple_k': [t[0] for t in by_simple],
                'by_full_k': [t[0] for t in by_full],
                'by_h2': [t[0] for t in by_h2]
            },
            'correlations': {
                'simple_k': float(r_simple),
                'full_k': float(r_full),
                'h2': float(r_h2)
            },
            'conclusion': conclusion,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_d_revalidation/")
    print('\n"Does network topology truly affect emergent coherence?" 💚\n')


if __name__ == '__main__':
    main()
