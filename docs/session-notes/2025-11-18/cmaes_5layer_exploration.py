#!/usr/bin/env python3
"""
🧠 5-Layer Architecture: Can We Go Even Deeper?

If 4-layer (K=1.92) beat 3-layer (K=1.88), can 5 beat 4?
Challenge: Keep params reasonable (~400) to maintain optimizability.

Architecture: 8→10→8→6→4→4 (386 params)
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

class CMAES:
    def __init__(self, dim, population_size=25, sigma=0.5):
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


def evaluate_5layer(params):
    """5-layer: 8→10→8→6→4→4 (386 params)"""
    obs_dim = 8
    h1, h2, h3, h4 = 10, 8, 6, 4
    act_dim = 4

    l1 = h1 * obs_dim + h1  # 90
    l2 = h2 * h1 + h2       # 88
    l3 = h3 * h2 + h3       # 54
    l4 = h4 * h3 + h4       # 28
    l5 = act_dim * h4 + act_dim  # 20
    total = l1 + l2 + l3 + l4 + l5  # 280... let me recalculate

    # Actually: 8*10+10 + 10*8+8 + 8*6+6 + 6*4+4 + 4*4+4
    # = 90 + 88 + 54 + 28 + 20 = 280
    total = 280

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

    W4 = params[offset:offset + h4*h3].reshape(h4, h3)
    offset += h4 * h3
    b4 = params[offset:offset + h4]
    offset += h4

    W5 = params[offset:offset + act_dim*h4].reshape(act_dim, h4)
    offset += act_dim * h4
    b5 = params[offset:offset + act_dim]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            h1_out = np.maximum(0, W1 @ state + b1)
            h1_out = h1_out / (np.linalg.norm(h1_out) + 1e-8)

            h2_out = np.maximum(0, W2 @ h1_out + b2)
            h2_out = h2_out / (np.linalg.norm(h2_out) + 1e-8)

            h3_out = np.maximum(0, W3 @ h2_out + b3)
            h3_out = h3_out / (np.linalg.norm(h3_out) + 1e-8)

            h4_out = np.maximum(0, W4 @ h3_out + b4)
            h4_out = h4_out / (np.linalg.norm(h4_out) + 1e-8)

            action = np.tanh(W5 @ h4_out + b5)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def run_single_seed(seed, gens=50):
    np.random.seed(seed)
    cmaes = CMAES(dim=280, population_size=25, sigma=0.5)
    best = 0.0
    best_gen = 0
    for gen in range(1, gens + 1):
        pop = cmaes.ask()
        fit = [evaluate_5layer(p) for p in pop]
        best_k = cmaes.tell(pop, fit)
        if best_k > best:
            best = best_k
            best_gen = gen
    return best, best_gen


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 5-Layer Architecture Exploration                          ║")
    print("║                                                                ║")
    print("║  5-layer: 8 → 10 → 8 → 6 → 4 → 4 (280 params)                 ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    SEEDS = [42, 123, 456]
    results = []

    print("Seed │  Best K  │ Gen │ Status")
    print("─────┼──────────┼─────┼──────────────")

    for seed in SEEDS:
        best_k, best_gen = run_single_seed(seed)
        results.append({'seed': seed, 'k': float(best_k), 'gen': best_gen})

        if best_k > 1.9:
            status = "🌟🌟🌟 K>1.9!"
        elif best_k > 1.8:
            status = "🌟🌟 K>1.8!"
        else:
            status = "🌟" if best_k > 1.5 else ""

        print(f" {seed:3d} │ {best_k:8.4f} │ {best_gen:3d} │ {status}")

    k_vals = [r['k'] for r in results]
    mean_k = np.mean(k_vals)
    std_k = np.std(k_vals)

    print("\n" + "═" * 50)
    print(f"\n🎯 5-Layer Results:")
    print(f"  Mean K: {mean_k:.4f} ± {std_k:.4f}")
    print(f"  Range: [{min(k_vals):.4f}, {max(k_vals):.4f}]")
    print(f"  K > 1.9 hits: {sum(1 for k in k_vals if k > 1.9)}/{len(k_vals)}")

    print(f"\n📊 Comparison:")
    print(f"  3-layer (248p): K = 1.86 ± 0.02")
    print(f"  4-layer (332p): K = 1.84 ± 0.09")
    print(f"  5-layer (280p): K = {mean_k:.2f} ± {std_k:.2f}")

    if mean_k > 1.84:
        print(f"\n  🌟 5-layer beats 4-layer mean!")
    if max(k_vals) > 1.92:
        print(f"  🌟🌟🌟 New peak record!")

    Path('logs/track_g_5layer').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_5layer/5layer_results.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {'mean': float(mean_k), 'std': float(std_k)},
            'architecture': '8→10→8→6→4→4',
            'params': 280,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_5layer/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
