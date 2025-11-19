#!/usr/bin/env python3
"""
🧠 Normalization Necessity Test

Hypothesis H2: Layer normalization is necessary for K > 1.8.
Without it, K will cap at ~1.5.

Tests our optimal 4-layer with and without layer normalization.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr


class CMAES:
    def __init__(self, dim, population_size=30, sigma=0.5):
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
        elite_size = max(4, self.pop_size // 4)
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


def k_index(obs_norms, act_norms):
    if len(obs_norms) < 2:
        return 0.0
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 0.0 if np.isnan(r) else 2.0 * abs(r)
    except:
        return 0.0


def evaluate_with_norm(params):
    """4-layer WITH layer normalization."""
    obs_dim, h1, h2, h3, act_dim = 8, 12, 10, 6, 4
    total = 332

    if len(params) < total:
        params = np.concatenate([params, np.zeros(total - len(params))])

    offset = 0
    W1 = params[offset:offset + h1*obs_dim].reshape(h1, obs_dim)
    offset += h1 * obs_dim
    b1 = params[offset:offset + h1]
    offset += h1
    W2 = params[offset:offset + h2*h1].reshape(h2, h1)
    offset += h2 * h1
    b2 = params[offset:offset + h2]
    offset += h2
    W3 = params[offset:offset + h3*h2].reshape(h3, h2)
    offset += h3 * h2
    b3 = params[offset:offset + h3]
    offset += h3
    W4 = params[offset:offset + act_dim*h3].reshape(act_dim, h3)
    offset += act_dim * h3
    b4 = params[offset:offset + act_dim]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            h1_out = np.maximum(0, W1 @ state + b1)
            h1_out = h1_out / (np.linalg.norm(h1_out) + 1e-8)  # NORMALIZED
            h2_out = np.maximum(0, W2 @ h1_out + b2)
            h2_out = h2_out / (np.linalg.norm(h2_out) + 1e-8)  # NORMALIZED
            h3_out = np.maximum(0, W3 @ h2_out + b3)
            h3_out = h3_out / (np.linalg.norm(h3_out) + 1e-8)  # NORMALIZED
            action = np.tanh(W4 @ h3_out + b4)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def evaluate_without_norm(params):
    """4-layer WITHOUT layer normalization."""
    obs_dim, h1, h2, h3, act_dim = 8, 12, 10, 6, 4
    total = 332

    if len(params) < total:
        params = np.concatenate([params, np.zeros(total - len(params))])

    offset = 0
    W1 = params[offset:offset + h1*obs_dim].reshape(h1, obs_dim)
    offset += h1 * obs_dim
    b1 = params[offset:offset + h1]
    offset += h1
    W2 = params[offset:offset + h2*h1].reshape(h2, h1)
    offset += h2 * h1
    b2 = params[offset:offset + h2]
    offset += h2
    W3 = params[offset:offset + h3*h2].reshape(h3, h2)
    offset += h3 * h2
    b3 = params[offset:offset + h3]
    offset += h3
    W4 = params[offset:offset + act_dim*h3].reshape(act_dim, h3)
    offset += act_dim * h3
    b4 = params[offset:offset + act_dim]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            h1_out = np.maximum(0, W1 @ state + b1)
            # NO NORMALIZATION
            h2_out = np.maximum(0, W2 @ h1_out + b2)
            # NO NORMALIZATION
            h3_out = np.maximum(0, W3 @ h2_out + b3)
            # NO NORMALIZATION
            action = np.tanh(W4 @ h3_out + b4)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Normalization Necessity Test                              ║")
    print("║                                                                ║")
    print("║  H2: Layer normalization is necessary for K > 1.8             ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    GENS = 60
    results = []

    # Test WITH normalization
    print("Testing WITH layer normalization...")
    np.random.seed(42)
    cmaes = CMAES(dim=332, population_size=30, sigma=0.5)
    best_with = 0.0
    for _ in range(GENS):
        pop = cmaes.ask()
        fit = [evaluate_with_norm(p) for p in pop]
        k = cmaes.tell(pop, fit)
        if k > best_with:
            best_with = k
    results.append({'condition': 'WITH normalization', 'k': float(best_with)})

    # Test WITHOUT normalization
    print("Testing WITHOUT layer normalization...")
    np.random.seed(42)
    cmaes = CMAES(dim=332, population_size=30, sigma=0.5)
    best_without = 0.0
    for _ in range(GENS):
        pop = cmaes.ask()
        fit = [evaluate_without_norm(p) for p in pop]
        k = cmaes.tell(pop, fit)
        if k > best_without:
            best_without = k
    results.append({'condition': 'WITHOUT normalization', 'k': float(best_without)})

    print("\n" + "═" * 55)
    print("\nCondition              │  Best K  │ Correlation")
    print("───────────────────────┼──────────┼────────────")

    for r in results:
        status = "🌟🌟🌟" if r['k'] > 1.9 else ("🌟🌟" if r['k'] > 1.8 else ("🌟" if r['k'] > 1.5 else ""))
        print(f"{r['condition']:22s} │ {r['k']:8.4f} │ {r['k']/2:10.3f} {status}")

    # Analyze
    diff = best_with - best_without
    print(f"\n📊 Analysis:")
    print(f"  Difference: {diff:+.4f}")
    print(f"  Improvement from normalization: {diff/best_without*100:.1f}%")

    if best_with > 1.8 and best_without < 1.6:
        print(f"\n  ✅ H2 CONFIRMED: Normalization IS necessary for K > 1.8")
        verdict = "CONFIRMED"
    elif best_with > best_without + 0.2:
        print(f"\n  🔶 H2 PARTIALLY CONFIRMED: Normalization helps significantly")
        verdict = "PARTIAL"
    else:
        print(f"\n  ❌ H2 REJECTED: Normalization is NOT necessary")
        verdict = "REJECTED"

    # Save
    Path('logs/normalization_test').mkdir(parents=True, exist_ok=True)
    with open('logs/normalization_test/results.json', 'w') as f:
        json.dump({
            'results': results,
            'verdict': verdict,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/normalization_test/")
    print('\n"Architecture choices reveal hidden assumptions." 💚\n')


if __name__ == '__main__':
    main()
