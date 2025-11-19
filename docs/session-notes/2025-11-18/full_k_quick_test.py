#!/usr/bin/env python3
"""
🧠 Quick Full K-Index vs Simple K Performance Test

Faster version: fewer generations, smaller population.
Tests if Full K predicts CartPole performance better than Simple K.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from collections import Counter


class CMAES:
    def __init__(self, dim, population_size=15, sigma=0.5):
        self.dim = dim
        self.pop_size = population_size
        self.sigma = sigma
        self.mean = np.random.randn(dim) * 0.1
        self.C = np.eye(dim)
        self.c_mu = 0.3

    def ask(self):
        L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        return [self.mean + self.sigma * (L @ np.random.randn(self.dim))
                for _ in range(self.pop_size)]

    def tell(self, population, fitness):
        indices = np.argsort(fitness)[::-1]
        elite_size = max(3, self.pop_size // 4)
        weights = np.log(elite_size + 0.5) - np.log(np.arange(1, elite_size + 1))
        weights = weights / weights.sum()
        elite_pop = np.array([population[i] for i in indices[:elite_size]])
        new_mean = np.sum(weights[:, None] * elite_pop, axis=0)
        y = (elite_pop - self.mean) / self.sigma
        self.C = (1 - self.c_mu) * self.C + self.c_mu * (y.T @ np.diag(weights) @ y)
        eigvals = np.linalg.eigvalsh(self.C)
        if np.min(eigvals) < 1e-10:
            self.C += (1e-8 - np.min(eigvals)) * np.eye(self.dim)
        self.mean = new_mean
        self.sigma *= 1.01 if (np.max(fitness) - np.mean(fitness)) > 0.02 else 0.99
        self.sigma = np.clip(self.sigma, 0.01, 1.0)
        return np.max(fitness)


def compute_h2_diversity(action_history):
    if len(action_history) < 3:
        return 0.5
    actions = np.array(action_history)
    discretized = [tuple(int(min(4, max(0, (x + 1) / 2 * 5))) for x in a) for a in actions]
    type_counts = Counter(discretized)
    n_types = len(type_counts)
    if n_types <= 1:
        return 0.0
    total = len(discretized)
    probs = np.array([c / total for c in type_counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    return h / np.log(n_types) if np.log(n_types) > 0 else 0.0


def compute_h4_entropy(action_history):
    if len(action_history) < 5:
        return 0.5
    actions = np.array(action_history)
    n_bins = min(5, len(actions) // 3)
    if n_bins < 2:
        return 0.5
    entropies = []
    for dim in range(actions.shape[1]):
        hist, _ = np.histogram(actions[:, dim], bins=n_bins)
        hist = hist + 1e-10
        probs = hist / hist.sum()
        entropies.append(-np.sum(probs * np.log(probs)))
    return np.mean(entropies) / np.log(n_bins) if np.log(n_bins) > 0 else 0.0


def compute_h6_balance(action_history):
    if len(action_history) < 3:
        return 0.5
    actions = np.array(action_history)
    imbalance = np.linalg.norm(np.mean(actions, axis=0))
    return max(0.0, 1.0 - imbalance)


def compute_full_k(action_history):
    """Simplified full K using H2, H4, H6."""
    h2 = compute_h2_diversity(action_history)
    h4 = compute_h4_entropy(action_history)
    h6 = compute_h6_balance(action_history)
    return 2.0 * (h2 + h4 + h6) / 3.0


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


class Policy:
    def __init__(self):
        # Smaller network: 4→8→2
        self.W1 = np.random.randn(8, 4) * 0.5
        self.b1 = np.zeros(8)
        self.W2 = np.random.randn(2, 8) * 0.5
        self.b2 = np.zeros(2)

    def forward(self, obs):
        h = np.maximum(0, self.W1 @ obs + self.b1)
        return self.W2 @ h + self.b2

    def act(self, obs):
        return np.argmax(self.forward(obs))

    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params):
        self.W1 = params[:32].reshape(8, 4)
        self.b1 = params[32:40]
        self.W2 = params[40:56].reshape(2, 8)
        self.b2 = params[56:58]


def evaluate_policy(policy, env):
    total = 0
    for _ in range(3):
        obs = env.reset()
        for _ in range(300):
            obs, r, done = env.step(policy.act(obs))
            total += r
            if done:
                break
    return total / 3


def compute_k_indices(policy, env):
    obs_norms, act_norms, actions = [], [], []
    for _ in range(2):
        obs = env.reset()
        for _ in range(100):
            logits = policy.forward(obs)
            obs_norms.append(np.linalg.norm(obs))
            act_norms.append(np.linalg.norm(logits))
            actions.append(logits.copy())
            obs, _, done = env.step(np.argmax(logits))
            if done:
                break

    # Simple K
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        simple_k = 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        simple_k = 0.0

    # Full K
    full_k = compute_full_k(actions)

    return full_k, simple_k


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Quick Full K vs Simple K Performance Test                 ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)
    env = CartPoleEnv()
    policy = Policy()

    cmaes = CMAES(dim=58, population_size=15, sigma=0.5)
    history = []

    print("Gen │ Performance │  Full K  │ Simple K")
    print("────┼─────────────┼──────────┼──────────")

    for gen in range(1, 26):
        pop = cmaes.ask()
        fitness, full_ks, simple_ks = [], [], []

        for params in pop:
            policy.set_params(params)
            perf = evaluate_policy(policy, env)
            full_k, simple_k = compute_k_indices(policy, env)
            fitness.append(perf)
            full_ks.append(full_k)
            simple_ks.append(simple_k)

        best_fit = cmaes.tell(pop, fitness)
        best_idx = np.argmax(fitness)

        history.append({
            'gen': gen,
            'performance': float(best_fit),
            'full_k': float(full_ks[best_idx]),
            'simple_k': float(simple_ks[best_idx])
        })

        if gen % 5 == 0 or gen == 1:
            print(f"{gen:3d} │ {best_fit:11.1f} │ {full_ks[best_idx]:8.4f} │ {simple_ks[best_idx]:8.4f}")

    # Correlations
    perfs = [h['performance'] for h in history]
    full_ks = [h['full_k'] for h in history]
    simple_ks = [h['simple_k'] for h in history]

    corr_full, p_full = pearsonr(perfs, full_ks)
    corr_simple, p_simple = pearsonr(perfs, simple_ks)

    print("\n" + "═" * 55)
    print("\n📊 Performance Correlation:")
    print(f"  Full K vs Performance:   r = {corr_full:+.4f} (p = {p_full:.4f})")
    print(f"  Simple K vs Performance: r = {corr_simple:+.4f} (p = {p_simple:.4f})")

    if abs(corr_full) > abs(corr_simple):
        print(f"\n  ✅ Full K is BETTER predictor!")
        winner = "FULL"
    else:
        print(f"\n  ⚠️  Simple K is better predictor")
        winner = "SIMPLE"

    Path('logs/full_k_quick').mkdir(parents=True, exist_ok=True)
    with open('logs/full_k_quick/results.json', 'w') as f:
        json.dump({
            'history': history,
            'corr_full': float(corr_full),
            'corr_simple': float(corr_simple),
            'winner': winner,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/full_k_quick/")
    print('\n"Measure what matters." 💚\n')


if __name__ == '__main__':
    main()
