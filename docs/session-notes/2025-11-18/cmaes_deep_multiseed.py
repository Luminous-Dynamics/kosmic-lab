#!/usr/bin/env python3
"""
🧠 Deep Architecture Multi-Seed Validation
Validates K=1.8812 finding with seeds [42, 123, 456]
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

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
        return [self.mean + self.sigma * (L @ np.random.randn(self.dim)) for _ in range(self.pop_size)]

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


def evaluate_deep(params, obs_dim=8, h1_dim=12, h2_dim=8, act_dim=4):
    l1 = h1_dim * obs_dim + h1_dim
    l2 = h2_dim * h1_dim + h2_dim
    l3 = act_dim * h2_dim + act_dim
    total = l1 + l2 + l3

    if len(params) < total:
        params = np.concatenate([params, np.zeros(total - len(params))])

    offset = 0
    W1 = params[offset:offset + h1_dim*obs_dim].reshape(h1_dim, obs_dim)
    offset += h1_dim * obs_dim
    b1 = params[offset:offset + h1_dim]
    offset += h1_dim
    W2 = params[offset:offset + h2_dim*h1_dim].reshape(h2_dim, h1_dim)
    offset += h2_dim * h1_dim
    b2 = params[offset:offset + h2_dim]
    offset += h2_dim
    W3 = params[offset:offset + act_dim*h2_dim].reshape(act_dim, h2_dim)
    offset += act_dim * h2_dim
    b3 = params[offset:offset + act_dim]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            h1 = np.maximum(0, W1 @ state + b1)
            h1 = h1 / (np.linalg.norm(h1) + 1e-8)
            h2 = np.maximum(0, W2 @ h1 + b2)
            h2 = h2 / (np.linalg.norm(h2) + 1e-8)
            action = np.tanh(W3 @ h2 + b3)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def run_single_seed(seed, gens=50):
    np.random.seed(seed)
    cmaes = CMAES(dim=248, population_size=20, sigma=0.5)
    best = 0.0
    best_gen = 0
    for gen in range(1, gens + 1):
        pop = cmaes.ask()
        fit = [evaluate_deep(p) for p in pop]
        best_k = cmaes.tell(pop, fit)
        if best_k > best:
            best = best_k
            best_gen = gen
    return best, best_gen


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Deep Architecture Multi-Seed Validation                   ║")
    print("║                                                                ║")
    print("║  3-layer: 8 → 12 → 8 → 4 (248 params)                         ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    SEEDS = [42, 123, 456]
    results = []

    print("Seed │  Best K  │ Gen │ Status")
    print("─────┼──────────┼─────┼──────────────")

    for seed in SEEDS:
        best_k, best_gen = run_single_seed(seed)
        results.append({'seed': seed, 'k': float(best_k), 'gen': best_gen})
        status = "🌟🌟 K>1.8!" if best_k > 1.8 else ("🌟" if best_k > 1.5 else "⭐")
        print(f" {seed:3d} │ {best_k:8.4f} │ {best_gen:3d} │ {status}")

    k_vals = [r['k'] for r in results]
    mean_k = np.mean(k_vals)
    std_k = np.std(k_vals)

    print("\n" + "═" * 50)
    print(f"\n🎯 Deep Architecture Validation:")
    print(f"  Mean K: {mean_k:.4f} ± {std_k:.4f}")
    print(f"  Range: [{min(k_vals):.4f}, {max(k_vals):.4f}]")
    print(f"  Threshold hits: {sum(1 for k in k_vals if k > 1.5)}/{len(k_vals)}")
    print(f"  K > 1.8 hits: {sum(1 for k in k_vals if k > 1.8)}/{len(k_vals)}")

    if all(k > 1.8 for k in k_vals):
        print(f"\n  🌟🌟✨ ALL SEEDS K > 1.8! ✨🌟🌟")
    elif mean_k > 1.8:
        print(f"\n  🌟🌟 MEAN K > 1.8!")

    # Save
    Path('logs/track_g_deep_validation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_deep_validation/deep_multiseed.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {'mean': float(mean_k), 'std': float(std_k)},
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_deep_validation/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
