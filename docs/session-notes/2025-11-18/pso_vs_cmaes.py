#!/usr/bin/env python3
"""
🧠 PSO vs CMA-ES: Alternative Optimizer Comparison

Can Particle Swarm Optimization beat CMA-ES for K-Index optimization?
PSO is good for escaping local optima and can be simpler to tune.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

class PSO:
    """Particle Swarm Optimization."""

    def __init__(self, dim, n_particles=30, w=0.7, c1=1.5, c2=1.5):
        self.dim = dim
        self.n_particles = n_particles
        self.w = w    # inertia
        self.c1 = c1  # cognitive
        self.c2 = c2  # social

        # Initialize particles
        self.positions = np.random.randn(n_particles, dim) * 0.5
        self.velocities = np.random.randn(n_particles, dim) * 0.1
        self.personal_best_pos = self.positions.copy()
        self.personal_best_fit = np.full(n_particles, -np.inf)
        self.global_best_pos = self.positions[0].copy()
        self.global_best_fit = -np.inf

    def step(self, fitness_fn):
        """One PSO step."""
        # Evaluate all particles
        fitness = np.array([fitness_fn(p) for p in self.positions])

        # Update personal bests
        improved = fitness > self.personal_best_fit
        self.personal_best_pos[improved] = self.positions[improved]
        self.personal_best_fit[improved] = fitness[improved]

        # Update global best
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.global_best_fit:
            self.global_best_fit = fitness[best_idx]
            self.global_best_pos = self.positions[best_idx].copy()

        # Update velocities and positions
        r1, r2 = np.random.rand(2)
        cognitive = self.c1 * r1 * (self.personal_best_pos - self.positions)
        social = self.c2 * r2 * (self.global_best_pos - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social

        # Clip velocities
        self.velocities = np.clip(self.velocities, -1.0, 1.0)

        self.positions += self.velocities

        return self.global_best_fit


class CMAES:
    """CMA-ES for comparison."""

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


def evaluate_4layer(params):
    """Evaluate 4-layer network."""
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


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 PSO vs CMA-ES: Optimizer Comparison                       ║")
    print("║                                                                ║")
    print("║  Can PSO beat CMA-ES for K-Index optimization?                ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    DIM = 332
    GENS = 60

    results = []

    # Test CMA-ES
    print("Testing CMA-ES...")
    np.random.seed(42)
    cmaes = CMAES(dim=DIM, population_size=30, sigma=0.5)
    cmaes_best = 0.0
    for gen in range(1, GENS + 1):
        pop = cmaes.ask()
        fit = [evaluate_4layer(p) for p in pop]
        k = cmaes.tell(pop, fit)
        if k > cmaes_best:
            cmaes_best = k
    results.append({'optimizer': 'CMA-ES', 'k': float(cmaes_best)})

    # Test PSO
    print("Testing PSO...")
    np.random.seed(42)
    pso = PSO(dim=DIM, n_particles=30)
    pso_best = 0.0
    for gen in range(1, GENS + 1):
        k = pso.step(evaluate_4layer)
        if k > pso_best:
            pso_best = k
    results.append({'optimizer': 'PSO', 'k': float(pso_best)})

    print("\n" + "═" * 50)
    print("\nOptimizer │  Best K  │ Correlation")
    print("──────────┼──────────┼────────────")

    for r in results:
        status = "🌟🌟🌟" if r['k'] > 1.9 else ("🌟🌟" if r['k'] > 1.8 else "🌟")
        print(f"{r['optimizer']:9s} │ {r['k']:8.4f} │ {r['k']/2:10.3f} {status}")

    winner = max(results, key=lambda x: x['k'])
    print(f"\n🏆 Winner: {winner['optimizer']} (K = {winner['k']:.4f})")

    if winner['optimizer'] == 'PSO':
        print("\n🌟 PSO beats CMA-ES!")
    else:
        print("\n✅ CMA-ES remains optimal")

    # Save
    Path('logs/track_g_optimizers').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_optimizers/pso_vs_cmaes.json', 'w') as f:
        json.dump({
            'results': results,
            'winner': winner,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_optimizers/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
