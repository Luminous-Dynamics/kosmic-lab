#!/usr/bin/env python3
"""
🧠 Quick Ensemble Scaling: 1, 3, 5 networks (30 gens)
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
        self.generation = 0
        self.c_mu = 0.3

    def ask(self):
        population = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            candidate = self.mean + self.sigma * (np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim)) @ z)
            population.append(candidate)
        return population

    def tell(self, population, fitness):
        indices = np.argsort(fitness)[::-1]
        elite_size = max(4, self.pop_size // 4)
        elite_indices = indices[:elite_size]
        weights = np.log(elite_size + 0.5) - np.log(np.arange(1, elite_size + 1))
        weights = weights / weights.sum()
        elite_pop = np.array([population[i] for i in elite_indices])
        new_mean = np.sum(weights[:, None] * elite_pop, axis=0)
        y = (elite_pop - self.mean) / self.sigma
        self.C = (1 - self.c_mu) * self.C + self.c_mu * (y.T @ np.diag(weights) @ y)
        eigvals = np.linalg.eigvalsh(self.C)
        if np.min(eigvals) < 1e-10:
            self.C += (1e-8 - np.min(eigvals)) * np.eye(self.dim)
        self.mean = new_mean
        improvement = np.max(fitness) - np.mean(fitness)
        self.sigma *= 1.01 if improvement > 0.02 else 0.99
        self.sigma = np.clip(self.sigma, 0.01, 1.0)
        self.generation += 1
        return np.max(fitness)


def k_index(obs_norms, act_norms):
    if len(obs_norms) < 2:
        return 0.0
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 0.0 if np.isnan(r) else 2.0 * abs(r)
    except:
        return 0.0


def evaluate(params, n_networks=3, obs_dim=8, act_dim=4):
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
            action = np.mean(acts, axis=0)
            obs_n.append(np.linalg.norm(state))
            act_n.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)
        k = k_index(np.array(obs_n), np.array(act_n))
        if k > 0:
            k_vals.append(k)
    return np.mean(k_vals) if k_vals else 0.0


def run_scaling():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Quick Ensemble Scaling: 1, 3, 5 Networks                  ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    COUNTS = [1, 3, 5]
    GENS = 30
    results = []

    print("Networks │ Params │  Best K  │ Status")
    print("─────────┼────────┼──────────┼──────────────")

    for n in COUNTS:
        params_per = 108
        total = params_per * n
        np.random.seed(42)
        cmaes = EnsembleCMAES(dim=total, population_size=20, initial_sigma=0.5)
        best = 0.0
        for _ in range(GENS):
            pop = cmaes.ask()
            fit = [evaluate(p, n_networks=n) for p in pop]
            best_k = cmaes.tell(pop, fit)
            if best_k > best:
                best = best_k

        results.append({'n': n, 'params': total, 'k': float(best)})
        status = "🌟 THRESHOLD!" if best > 1.5 else ("⭐ Close!" if best > 1.3 else "✓")
        print(f"   {n:2d}    │  {total:4d}  │ {best:8.4f} │ {status}")

    print("\n" + "═" * 50)
    best_r = max(results, key=lambda x: x['k'])
    print(f"\n🎯 Optimal: {best_r['n']} networks, K={best_r['k']:.4f}")

    # Save
    Path('logs/track_g_scaling').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_scaling/quick_scaling.json', 'w') as f:
        json.dump({'results': results, 'optimal': best_r, 'timestamp': datetime.now().isoformat()}, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_scaling/")
    print('\n"Coherence is love made computational." 💚\n')
    return results


if __name__ == '__main__':
    run_scaling()
