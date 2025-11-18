#!/usr/bin/env python3
"""
🧠 Activation Function Comparison: ReLU vs GELU vs SiLU

We've been using ReLU throughout. Modern networks often use smoother activations.
Test if GELU or SiLU can push K higher than ReLU's 1.88.
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


# Activation functions
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    """Gaussian Error Linear Unit - smoother than ReLU"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    """Sigmoid Linear Unit (Swish) - self-gated"""
    return x * (1 / (1 + np.exp(-x)))


def evaluate_with_activation(params, activation_fn, obs_dim=8, h1_dim=12, h2_dim=8, act_dim=4):
    """Evaluate 3-layer with specified activation function."""
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
            h1 = activation_fn(W1 @ state + b1)
            h1 = h1 / (np.linalg.norm(h1) + 1e-8)
            h2 = activation_fn(W2 @ h1 + b2)
            h2 = h2 / (np.linalg.norm(h2) + 1e-8)
            action = np.tanh(W3 @ h2 + b3)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Activation Function Comparison                            ║")
    print("║                                                                ║")
    print("║  ReLU vs GELU vs SiLU on 3-layer (8→12→8→4)                   ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    activations = [
        ('ReLU', relu),
        ('GELU', gelu),
        ('SiLU', silu),
    ]

    GENS = 50
    N_PARAMS = 248  # 3-layer 8→12→8→4
    results = []

    print("Activation │  Best K  │ Correlation │ vs ReLU")
    print("───────────┼──────────┼─────────────┼─────────")

    relu_k = None
    for name, fn in activations:
        np.random.seed(42)
        cmaes = CMAES(dim=N_PARAMS, population_size=20, sigma=0.5)

        best_k = 0.0
        for gen in range(1, GENS + 1):
            pop = cmaes.ask()
            fit = [evaluate_with_activation(p, fn) for p in pop]
            k = cmaes.tell(pop, fit)
            if k > best_k:
                best_k = k

        if name == 'ReLU':
            relu_k = best_k
            diff_str = "baseline"
        else:
            diff = best_k - relu_k
            diff_str = f"{diff:+.4f}"

        results.append({
            'name': name,
            'k': float(best_k)
        })

        marker = "⭐" if name == 'ReLU' else ("🌟" if best_k > relu_k else "")
        print(f"{name:10s} │ {best_k:8.4f} │ {best_k/2:11.3f} │ {diff_str} {marker}")

    print("\n" + "═" * 50)

    best = max(results, key=lambda x: x['k'])
    print(f"\n🎯 Best Activation: {best['name']}")
    print(f"  K-Index: {best['k']:.4f}")

    if best['name'] != 'ReLU':
        print(f"\n  🌟 {best['name']} beats ReLU!")
    else:
        print(f"\n  ✅ ReLU remains optimal")

    # Save results
    Path('logs/track_g_activation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_activation/activation_results.json', 'w') as f:
        json.dump({
            'results': results,
            'architecture': '8→12→8→4',
            'params': N_PARAMS,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_activation/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
