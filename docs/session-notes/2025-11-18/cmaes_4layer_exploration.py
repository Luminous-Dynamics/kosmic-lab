#!/usr/bin/env python3
"""
🧠 4-Layer Architecture Exploration: Can Deeper Beat K=1.88?

Hypothesis: If 3-layer (K=1.88) beat 2-layer (K=1.79), maybe 4-layer can go higher.
Challenge: Keep params in sweet spot (~200-350) to avoid optimization difficulty.

Architectures to test:
- 8→10→8→6→4 (242 params) - Similar to optimal
- 8→8→6→4→4 (200 params) - Smaller
- 8→12→10→6→4 (322 params) - Slightly larger
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


def evaluate_4layer(params, dims):
    """Evaluate 4-layer network with given dimensions."""
    obs_dim, h1, h2, h3, act_dim = dims

    # Calculate parameter counts
    l1 = h1 * obs_dim + h1
    l2 = h2 * h1 + h2
    l3 = h3 * h2 + h3
    l4 = act_dim * h3 + act_dim
    total = l1 + l2 + l3 + l4

    if len(params) < total:
        params = np.concatenate([params, np.zeros(total - len(params))])

    # Extract weights
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
            h1_out = h1_out / (np.linalg.norm(h1_out) + 1e-8)

            h2_out = np.maximum(0, W2 @ h1_out + b2)
            h2_out = h2_out / (np.linalg.norm(h2_out) + 1e-8)

            h3_out = np.maximum(0, W3 @ h2_out + b3)
            h3_out = h3_out / (np.linalg.norm(h3_out) + 1e-8)

            action = np.tanh(W4 @ h3_out + b4)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def count_params(dims):
    """Count parameters for 4-layer network."""
    obs_dim, h1, h2, h3, act_dim = dims
    return (h1 * obs_dim + h1 + h2 * h1 + h2 + h3 * h2 + h3 + act_dim * h3 + act_dim)


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 4-Layer Architecture Exploration                          ║")
    print("║                                                                ║")
    print("║  Can we beat 3-layer K=1.88 with 4 layers?                    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    # Architectures to test (obs, h1, h2, h3, act)
    architectures = [
        ((8, 10, 8, 6, 4), "8→10→8→6→4"),
        ((8, 8, 6, 4, 4), "8→8→6→4→4"),
        ((8, 12, 10, 6, 4), "8→12→10→6→4"),
    ]

    GENS = 50
    results = []

    print("Architecture        │ Params │  Best K  │ Correlation │ vs 3-layer")
    print("────────────────────┼────────┼──────────┼─────────────┼───────────")

    for dims, name in architectures:
        n_params = count_params(dims)

        np.random.seed(42)
        cmaes = CMAES(dim=n_params, population_size=20, sigma=0.5)

        best_k = 0.0
        for gen in range(1, GENS + 1):
            pop = cmaes.ask()
            fit = [evaluate_4layer(p, dims) for p in pop]
            k = cmaes.tell(pop, fit)
            if k > best_k:
                best_k = k

        diff = best_k - 1.88
        status = "🌟" if best_k > 1.88 else ("≈" if best_k > 1.85 else "↓")

        results.append({
            'name': name,
            'dims': dims,
            'params': n_params,
            'k': float(best_k)
        })

        print(f"{name:19s} │ {n_params:6d} │ {best_k:8.4f} │ {best_k/2:11.3f} │ {diff:+.4f} {status}")

    print("\n" + "═" * 65)

    best = max(results, key=lambda x: x['k'])
    print(f"\n🎯 Best 4-Layer: {best['name']}")
    print(f"  K-Index: {best['k']:.4f}")
    print(f"  Parameters: {best['params']}")

    if best['k'] > 1.88:
        print(f"\n  🌟🌟 4-LAYER BEATS 3-LAYER! 🌟🌟")
    elif best['k'] > 1.85:
        print(f"\n  ≈ Comparable to 3-layer (1.88)")
    else:
        print(f"\n  ↓ 3-layer (248p, K=1.88) remains optimal")

    # Save results
    Path('logs/track_g_4layer').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_4layer/4layer_results.json', 'w') as f:
        json.dump({
            'results': results,
            'baseline': {'name': '3-layer 8→12→8→4', 'params': 248, 'k': 1.88},
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_4layer/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
