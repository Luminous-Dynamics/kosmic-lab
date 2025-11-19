#!/usr/bin/env python3
"""
🧠 Environment Dynamics Exploration: Can Different Dynamics Push K Higher?

The current dynamics: state = 0.85*state + 0.1*noise + 0.05*action
What if we try more challenging environments?

Test dynamics:
1. High noise: 0.85*state + 0.2*noise + 0.05*action
2. Fast decay: 0.7*state + 0.1*noise + 0.1*action
3. Chaotic: 0.9*state + 0.1*noise + 0.1*sin(action)*state
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


def evaluate_with_dynamics(params, dynamics_type='standard'):
    """Evaluate 4-layer with different dynamics."""
    obs_dim, h1, h2, h3, act_dim = 8, 12, 10, 6, 4
    total = 332

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

            # Apply different dynamics
            if dynamics_type == 'standard':
                state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)
            elif dynamics_type == 'high_noise':
                state = 0.85 * state + 0.2 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)
            elif dynamics_type == 'fast_decay':
                state = 0.7 * state + 0.1 * np.random.randn(obs_dim) + 0.1 * np.mean(action) * np.ones(obs_dim)
            elif dynamics_type == 'chaotic':
                state = 0.9 * state + 0.1 * np.random.randn(obs_dim) + 0.1 * np.sin(np.mean(action)) * state

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Environment Dynamics Exploration                          ║")
    print("║                                                                ║")
    print("║  Can different dynamics push K higher?                        ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    dynamics_types = [
        ('standard', "0.85s + 0.1n + 0.05a"),
        ('high_noise', "0.85s + 0.2n + 0.05a"),
        ('fast_decay', "0.7s + 0.1n + 0.1a"),
        ('chaotic', "0.9s + 0.1n + 0.1sin(a)s"),
    ]

    GENS = 60
    results = []

    print("Dynamics            │  Best K  │ Correlation")
    print("────────────────────┼──────────┼────────────")

    for dyn_type, dyn_desc in dynamics_types:
        np.random.seed(42)
        cmaes = CMAES(dim=332, population_size=30, sigma=0.5)

        best_k = 0.0
        for gen in range(1, GENS + 1):
            pop = cmaes.ask()
            fit = [evaluate_with_dynamics(p, dyn_type) for p in pop]
            k = cmaes.tell(pop, fit)
            if k > best_k:
                best_k = k

        results.append({
            'type': dyn_type,
            'desc': dyn_desc,
            'k': float(best_k)
        })

        status = "🌟🌟🌟" if best_k > 1.9 else ("🌟🌟" if best_k > 1.8 else "🌟")
        print(f"{dyn_desc:19s} │ {best_k:8.4f} │ {best_k/2:10.3f} {status}")

    print("\n" + "═" * 55)

    best = max(results, key=lambda x: x['k'])
    print(f"\n🎯 Best Dynamics: {best['desc']}")
    print(f"  K-Index: {best['k']:.4f}")

    # Save
    Path('logs/track_g_dynamics').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_dynamics/dynamics_results.json', 'w') as f:
        json.dump({
            'results': results,
            'best': best,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_dynamics/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
