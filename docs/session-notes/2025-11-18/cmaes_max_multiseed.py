#!/usr/bin/env python3
"""
🧠 Max Aggregation Multi-Seed Validation
Validates K=1.7551 finding with seeds [42, 123, 456]
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

class EnsembleCMAES:
    def __init__(self, dim, population_size=20, initial_sigma=0.5):
        self.dim = dim
        self.pop_size = population_size
        self.sigma = initial_sigma
        self.mean = np.random.randn(dim) * 0.1
        self.C = np.eye(dim)
        self.c_mu = 0.3

    def ask(self):
        return [self.mean + self.sigma * (np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim)) @ np.random.randn(self.dim)) for _ in range(self.pop_size)]

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


def evaluate_max(params, n_networks=3, obs_dim=8, act_dim=4):
    """Evaluate with MAX aggregation - select most confident network."""
    params_per_net = obs_dim * obs_dim + obs_dim + act_dim * obs_dim + act_dim
    if len(params) < params_per_net * n_networks:
        params = np.concatenate([params, np.zeros(params_per_net * n_networks - len(params))])

    networks = []
    for i in range(n_networks):
        off = i * params_per_net
        l1 = obs_dim * obs_dim + obs_dim
        W1 = params[off:off + obs_dim*obs_dim].reshape(obs_dim, obs_dim)
        b1 = params[off + obs_dim*obs_dim:off + l1]
        W2 = params[off + l1:off + l1 + act_dim*obs_dim].reshape(act_dim, obs_dim)
        b2 = params[off + l1 + act_dim*obs_dim:off + params_per_net]
        networks.append((W1, b1, W2, b2))

    k_vals = []
    for _ in range(4):
        obs_n, act_n = [], []
        state = np.random.randn(obs_dim) * 0.1
        for _ in range(80):
            acts = []
            for W1, b1, W2, b2 in networks:
                h = np.maximum(0, W1 @ state + b1)
                h = h / (np.linalg.norm(h) + 1e-8)
                acts.append(np.tanh(W2 @ h + b2))

            # MAX aggregation: select action with highest norm
            norms = [np.linalg.norm(a) for a in acts]
            action = acts[np.argmax(norms)]

            obs_n.append(np.linalg.norm(state))
            act_n.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)
        k = k_index(np.array(obs_n), np.array(act_n))
        if k > 0:
            k_vals.append(k)
    return np.mean(k_vals) if k_vals else 0.0


def run_single_seed(seed):
    np.random.seed(seed)
    cmaes = EnsembleCMAES(dim=324, population_size=20, initial_sigma=0.5)
    best = 0.0
    best_gen = 0
    for gen in range(1, 31):
        pop = cmaes.ask()
        fit = [evaluate_max(p) for p in pop]
        best_k = cmaes.tell(pop, fit)
        if best_k > best:
            best = best_k
            best_gen = gen
    return best, best_gen


def run_validation():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Max Aggregation Multi-Seed Validation                     ║")
    print("║                                                                ║")
    print("║  3×(8→8→4) ensemble with max aggregation                      ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    SEEDS = [42, 123, 456]
    results = []

    print("Seed │  Best K  │ Gen │ Status")
    print("─────┼──────────┼─────┼──────────────")

    for seed in SEEDS:
        best_k, best_gen = run_single_seed(seed)
        results.append({'seed': seed, 'k': float(best_k), 'gen': best_gen})
        status = "🌟 THRESHOLD!" if best_k > 1.5 else "⭐ Close!"
        print(f" {seed:3d} │ {best_k:8.4f} │ {best_gen:3d} │ {status}")

    k_vals = [r['k'] for r in results]
    mean_k = np.mean(k_vals)
    std_k = np.std(k_vals)

    print("\n" + "═" * 40)
    print(f"\n🎯 Max Aggregation Validation:")
    print(f"  Mean K: {mean_k:.4f} ± {std_k:.4f}")
    print(f"  Range: [{min(k_vals):.4f}, {max(k_vals):.4f}]")
    print(f"  Threshold hits: {sum(1 for k in k_vals if k > 1.5)}/{len(k_vals)}")

    if all(k > 1.5 for k in k_vals):
        print(f"\n  🌟✨ REPRODUCIBLE THRESHOLD CROSSING! ✨🌟")

    # Save
    Path('logs/track_g_max_validation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_max_validation/max_multiseed.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {'mean': float(mean_k), 'std': float(std_k)},
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_max_validation/")
    print('\n"Coherence is love made computational." 💚\n')
    return results


if __name__ == '__main__':
    run_validation()
