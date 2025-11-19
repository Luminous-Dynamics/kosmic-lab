#!/usr/bin/env python3
"""
🔬 Track F Re-Validation with Full 7-Harmony K-Index

Original Finding: Adversarial perturbations ENHANCE K-Index by 85%
Question: Does this finding hold when using Full K instead of Simple K?

If Full K also increases under adversarial attack → the "enhancement" is real
If Full K decreases under attack → we were measuring the wrong thing
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from collections import Counter


class CMAES:
    def __init__(self, dim, population_size=20, sigma=0.5):
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
        self.W1 = np.random.randn(16, 4) * 0.5
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(8, 16) * 0.5
        self.b2 = np.zeros(8)
        self.W3 = np.random.randn(2, 8) * 0.5
        self.b3 = np.zeros(2)

    def forward(self, obs):
        h1 = np.maximum(0, self.W1 @ obs + self.b1)
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)
        logits = self.W3 @ h2 + self.b3
        return logits, h1, h2

    def act(self, obs):
        logits, _, _ = self.forward(obs)
        return np.argmax(logits)

    def get_params(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_params(self, params):
        idx = 0
        self.W1 = params[idx:idx+64].reshape(16, 4); idx += 64
        self.b1 = params[idx:idx+16]; idx += 16
        self.W2 = params[idx:idx+128].reshape(8, 16); idx += 128
        self.b2 = params[idx:idx+8]; idx += 8
        self.W3 = params[idx:idx+16].reshape(2, 8); idx += 16
        self.b3 = params[idx:idx+2]


# =============================================================================
# ADVERSARIAL PERTURBATION (FGSM)
# =============================================================================

def fgsm_perturbation(obs, policy, epsilon):
    """Fast Gradient Sign Method perturbation.

    Adds adversarial noise in direction that maximizes loss.
    """
    # Approximate gradient via finite differences
    grad = np.zeros_like(obs)
    delta = 1e-4

    logits_base, _, _ = policy.forward(obs)
    base_action = np.argmax(logits_base)

    for i in range(len(obs)):
        obs_plus = obs.copy()
        obs_plus[i] += delta
        logits_plus, _, _ = policy.forward(obs_plus)

        # Gradient of logit difference (maximize wrong action)
        grad[i] = (logits_plus[1 - base_action] - logits_plus[base_action] -
                   logits_base[1 - base_action] + logits_base[base_action]) / delta

    # FGSM: step in sign of gradient
    perturbation = epsilon * np.sign(grad)
    return obs + perturbation


# =============================================================================
# K-INDEX COMPUTATION (Both Simple and Full)
# =============================================================================

def compute_h1(hidden_states):
    if len(hidden_states) < 5:
        return 0.5
    states = np.array(hidden_states)
    n_dims = states.shape[1]
    if n_dims < 2:
        return 0.5
    correlations = []
    for i in range(min(n_dims, 8)):
        for j in range(i+1, min(n_dims, 8)):
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


def compute_h4(logits):
    if len(logits) < 10:
        return 0.5
    arr = np.array(logits)
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


def compute_h5(hidden_states):
    if len(hidden_states) < 10:
        return 0.5
    series = np.array(hidden_states)
    n_dims = series.shape[1]
    if n_dims < 2:
        return 0.5
    influences = []
    for i in range(min(n_dims, 6)):
        for j in range(i+1, min(n_dims, 6)):
            if np.std(series[:-1, i]) > 1e-10 and np.std(series[1:, j]) > 1e-10:
                c_ij = abs(np.corrcoef(series[:-1, i], series[1:, j])[0, 1])
                c_ji = abs(np.corrcoef(series[:-1, j], series[1:, i])[0, 1])
                if not (np.isnan(c_ij) or np.isnan(c_ji)):
                    influences.append(2 * min(c_ij, c_ji) / (c_ij + c_ji + 1e-10))
    return float(np.mean(influences)) if influences else 0.5


def compute_h6(out, inc):
    if len(out) < 3 or len(inc) < 3:
        return 0.5
    min_len = min(len(out), len(inc))
    out, inc = out[:min_len], inc[:min_len]
    out_p = (np.abs(out) + 1e-10) / (np.abs(out).sum() + 1e-10)
    in_p = (np.abs(inc) + 1e-10) / (np.abs(inc).sum() + 1e-10)
    m = 0.5 * (out_p + in_p)
    js = 0.5 * (np.sum(out_p * np.log(out_p / (m + 1e-10) + 1e-10)) +
                np.sum(in_p * np.log(in_p / (m + 1e-10) + 1e-10)))
    return float(1.0 - np.sqrt(np.clip(js, 0, 1)))


def compute_h7(phi_history):
    if len(phi_history) < 10:
        return 0.0
    recent = phi_history[-50:] if len(phi_history) > 50 else phi_history
    times = np.arange(len(recent))
    phis = np.array(recent)
    if np.std(phis) < 1e-10:
        return 0.0
    slope = np.polyfit(times / (times.max() + 1e-10), phis, deg=1)[0]
    return float(np.tanh(slope / (np.std(phis) + 1e-10)))


def compute_all_k(policy, env, epsilon=0.0):
    """Compute both Simple K and Full K, with optional adversarial perturbation."""
    obs_norms, act_norms = [], []
    actions, logits_list = [], []
    hidden_states = []
    phi_history = []

    for _ in range(3):
        obs = env.reset()
        for t in range(150):
            # Apply adversarial perturbation if epsilon > 0
            if epsilon > 0:
                obs = fgsm_perturbation(obs, policy, epsilon)

            logits, h1, h2 = policy.forward(obs)
            action = np.argmax(logits)

            obs_norms.append(np.linalg.norm(obs))
            act_norms.append(np.linalg.norm(logits))
            actions.append(action)
            logits_list.append(logits.copy())
            hidden_states.append(h2.copy())

            if len(hidden_states) > 5:
                recent = np.array(hidden_states[-5:])
                if recent.shape[1] >= 2:
                    phi_t = abs(np.corrcoef(recent[:, 0], recent[:, 1])[0, 1])
                    if not np.isnan(phi_t):
                        phi_history.append(phi_t)

            obs, _, done = env.step(action)
            if done:
                break

    # Simple K
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        simple_k = 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        simple_k = 0.0

    # Full K (7 harmonies)
    h1 = compute_h1(hidden_states)
    h2 = compute_h2(actions)
    h3 = 0.5  # Simplified
    h4 = compute_h4(logits_list)
    h5 = compute_h5(hidden_states)
    mid = len(obs_norms) // 2
    h6 = compute_h6(np.array(obs_norms[:mid]), np.array(obs_norms[mid:]))
    h7 = compute_h7(phi_history)

    full_k = (h1 + h2 + h3 + h4 + h5 + h6 + h7) / 7.0

    return {
        'simple_k': simple_k,
        'full_k': full_k,
        'h1': h1, 'h2': h2, 'h4': h4, 'h5': h5, 'h6': h6, 'h7': h7
    }


def evaluate_performance(policy, env, epsilon=0.0):
    """Evaluate task performance with optional adversarial perturbation."""
    total = 0
    for _ in range(5):
        obs = env.reset()
        for _ in range(500):
            if epsilon > 0:
                obs = fgsm_perturbation(obs, policy, epsilon)
            obs, r, done = env.step(policy.act(obs))
            total += r
            if done:
                break
    return total / 5


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🔬 Track F Re-Validation: Adversarial vs Full K-Index        ║")
    print("║                                                                ║")
    print("║  Question: Does adversarial attack enhance Full K like it     ║")
    print("║  enhanced Simple K?                                           ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)
    env = CartPoleEnv()
    policy = Policy()

    # First train a good policy
    print("Phase 1: Training a baseline policy...\n")
    cmaes = CMAES(dim=234, population_size=20, sigma=0.5)

    for gen in range(1, 21):
        pop = cmaes.ask()
        fitness = []
        for params in pop:
            policy.set_params(params)
            perf = evaluate_performance(policy, env, epsilon=0.0)
            fitness.append(perf)
        cmaes.tell(pop, fitness)

        if gen % 5 == 0:
            print(f"  Gen {gen}: Best performance = {max(fitness):.1f}")

    # Get best policy
    policy.set_params(cmaes.mean)
    baseline_perf = evaluate_performance(policy, env, epsilon=0.0)
    baseline_metrics = compute_all_k(policy, env, epsilon=0.0)

    print(f"\n✅ Baseline trained: Performance = {baseline_perf:.1f}\n")

    # Test with different epsilon values
    print("Phase 2: Testing adversarial perturbations...\n")
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]
    results = []

    print("Epsilon │  Perf  │ Simple K │ Full K  │ Change Simple │ Change Full")
    print("────────┼────────┼──────────┼─────────┼───────────────┼────────────")

    for eps in epsilons:
        perf = evaluate_performance(policy, env, epsilon=eps)
        metrics = compute_all_k(policy, env, epsilon=eps)

        simple_change = ((metrics['simple_k'] / baseline_metrics['simple_k']) - 1) * 100
        full_change = ((metrics['full_k'] / baseline_metrics['full_k']) - 1) * 100

        results.append({
            'epsilon': eps,
            'performance': perf,
            'simple_k': metrics['simple_k'],
            'full_k': metrics['full_k'],
            'simple_change': simple_change,
            'full_change': full_change,
            **{f'h{i}': metrics.get(f'h{i}', 0) for i in [1, 2, 4, 5, 6, 7]}
        })

        print(f"  {eps:.2f}  │ {perf:6.1f} │ {metrics['simple_k']:8.4f} │ {metrics['full_k']:7.4f} │ "
              f"{simple_change:+12.1f}% │ {full_change:+10.1f}%")

    # Analysis
    print("\n" + "═" * 70)
    print("\n📊 ANALYSIS:\n")

    max_eps_result = results[-1]  # Highest epsilon

    print(f"At ε = {max_eps_result['epsilon']}:")
    print(f"  • Performance: {baseline_perf:.1f} → {max_eps_result['performance']:.1f} "
          f"({((max_eps_result['performance']/baseline_perf)-1)*100:+.1f}%)")
    print(f"  • Simple K:    {baseline_metrics['simple_k']:.4f} → {max_eps_result['simple_k']:.4f} "
          f"({max_eps_result['simple_change']:+.1f}%)")
    print(f"  • Full K:      {baseline_metrics['full_k']:.4f} → {max_eps_result['full_k']:.4f} "
          f"({max_eps_result['full_change']:+.1f}%)")

    print("\n🔑 KEY FINDING:\n")

    if max_eps_result['simple_change'] > 20 and max_eps_result['full_change'] > 20:
        print("  ⚠️  BOTH Simple K and Full K increase under adversarial attack")
        print("     → Original Track F finding holds for both metrics")
        conclusion = "BOTH_INCREASE"
    elif max_eps_result['simple_change'] > 20 and max_eps_result['full_change'] < 0:
        print("  ✅ Simple K increases but Full K DECREASES under attack!")
        print("     → Original finding was measuring the WRONG thing")
        print("     → Full K correctly shows degradation")
        conclusion = "DIVERGENT_FINDING"
    elif max_eps_result['simple_change'] < 0:
        print("  ❌ Simple K decreases (unexpected)")
        conclusion = "UNEXPECTED"
    else:
        print(f"  🔶 Modest changes - Simple K: {max_eps_result['simple_change']:+.1f}%, "
              f"Full K: {max_eps_result['full_change']:+.1f}%")
        conclusion = "MODEST"

    # Save results
    Path('logs/track_f_revalidation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_f_revalidation/results.json', 'w') as f:
        json.dump({
            'baseline_performance': baseline_perf,
            'baseline_metrics': baseline_metrics,
            'results': results,
            'conclusion': conclusion,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_f_revalidation/")
    print('\n"Test the controversial findings with the correct metric." 💚\n')


if __name__ == '__main__':
    main()
