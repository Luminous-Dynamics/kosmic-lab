#!/usr/bin/env python3
"""
🚀 Push for K > 1.8: Extended single network training
50 generations, population 20, optimized hyperparameters
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

class OptimizedCMAES:
    """CMA-ES with tuned hyperparameters for higher K."""

    def __init__(self, dim, population_size=25, initial_sigma=0.4):
        self.dim = dim
        self.pop_size = population_size
        self.sigma = initial_sigma
        self.mean = np.random.randn(dim) * 0.05  # Smaller init
        self.C = np.eye(dim)
        self.c_mu = 0.35  # Slightly higher learning rate
        self.generation = 0

    def ask(self):
        population = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            candidate = self.mean + self.sigma * (np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim)) @ z)
            population.append(candidate)
        return population

    def tell(self, population, fitness):
        indices = np.argsort(fitness)[::-1]
        elite_size = max(5, self.pop_size // 4)
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
        self.sigma *= 1.02 if improvement > 0.02 else 0.98
        self.sigma = np.clip(self.sigma, 0.01, 1.0)
        self.generation += 1
        return np.max(fitness), np.mean(fitness)


def k_index(obs_norms, act_norms):
    if len(obs_norms) < 2:
        return 0.0
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 0.0 if np.isnan(r) else 2.0 * abs(r)
    except:
        return 0.0


def evaluate(params, obs_dim=8, act_dim=4, episodes=5):
    """Evaluate single network with more episodes for stability."""
    params_per_net = obs_dim * obs_dim + obs_dim + act_dim * obs_dim + act_dim
    if len(params) < params_per_net:
        params = np.concatenate([params, np.zeros(params_per_net - len(params))])

    l1 = obs_dim * obs_dim + obs_dim
    W1 = params[:obs_dim*obs_dim].reshape(obs_dim, obs_dim)
    b1 = params[obs_dim*obs_dim:l1]
    W2 = params[l1:l1 + act_dim*obs_dim].reshape(act_dim, obs_dim)
    b2 = params[l1 + act_dim*obs_dim:params_per_net]

    k_vals = []
    for _ in range(episodes):
        obs_n, act_n = [], []
        state = np.random.randn(obs_dim) * 0.1
        for _ in range(100):  # Longer episodes
            h = np.maximum(0, W1 @ state + b1)
            h = h / (np.linalg.norm(h) + 1e-8)
            action = np.tanh(W2 @ h + b2)

            obs_n.append(np.linalg.norm(state))
            act_n.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_n), np.array(act_n))
        if k > 0:
            k_vals.append(k)
    return np.mean(k_vals) if k_vals else 0.0


def run_push():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🚀 Push for K > 1.8: Extended Single Network                 ║")
    print("║                                                                ║")
    print("║  Optimized hyperparameters, 50 generations                    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    PARAMS = 108
    GENS = 50
    POP = 25

    print("Configuration:")
    print("  Network: 8 → 8 → 4 (108 params)")
    print("  Population: 25 (increased)")
    print("  Generations: 50")
    print("  Episodes: 5 (increased)")
    print("  Episode length: 100 (increased)")
    print("  Target: K > 1.8")
    print("\n" + "═" * 60 + "\n")

    np.random.seed(42)
    cmaes = OptimizedCMAES(dim=PARAMS, population_size=POP, initial_sigma=0.4)

    results = {'generations': [], 'best_k': [], 'mean_k': []}
    best_overall = 0.0
    best_gen = 0

    print("Gen │  Best K  │  Mean K  │ Progress")
    print("────┼──────────┼──────────┼────────────────────────────")

    for gen in range(1, GENS + 1):
        pop = cmaes.ask()
        fit = [evaluate(p) for p in pop]
        best_k, mean_k = cmaes.tell(pop, fit)

        if best_k > best_overall:
            best_overall = best_k
            best_gen = gen

        results['generations'].append(gen)
        results['best_k'].append(float(best_k))
        results['mean_k'].append(float(mean_k))

        progress = (best_k / 1.8) * 100
        status = ""
        if best_k > 1.8:
            status = "🌟🌟 TARGET!"
        elif best_k > 1.5:
            status = "🌟 Threshold"
        elif best_k > 1.3:
            status = "⭐"

        if gen % 5 == 0 or gen == 1 or best_k > 1.7:
            print(f"{gen:3d} │ {best_k:8.4f} │ {mean_k:8.4f} │ {progress:5.1f}% {status}")

    print("\n" + "═" * 60)
    print(f"\n🎯 FINAL RESULT:")
    print(f"  Best K-Index: {best_overall:.4f} (Generation {best_gen})")
    print(f"  Progress to K=1.8: {(best_overall/1.8)*100:.1f}%")
    print(f"  Correlation: {best_overall/2:.3f}")

    if best_overall > 1.8:
        print(f"\n  🌟🌟✨ K > 1.8 ACHIEVED! ✨🌟🌟")
    elif best_overall > 1.5:
        print(f"\n  🌟 Threshold crossed: {best_overall:.4f}")

    # Save
    output_dir = Path('logs/track_g_push_1_8')
    output_dir.mkdir(parents=True, exist_ok=True)

    results['best_overall'] = float(best_overall)
    results['best_gen'] = best_gen
    results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'push_1_8_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Saved to {output_dir}/")
    print('\n"Coherence is love made computational." 💚\n')
    return results


if __name__ == '__main__':
    run_push()
