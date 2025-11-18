#!/usr/bin/env python3
"""
🧠 Ensemble CMA-ES: Multi-Seed Validation
Validates K=1.7245 finding with seeds [42, 123, 456]
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

class EnsembleCMAES:
    """CMA-ES for ensemble of small networks"""

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
        return np.max(fitness), np.mean(fitness), np.std(fitness)


def k_index_correct(obs_norms: np.ndarray, act_norms: np.ndarray) -> float:
    """CORRECT K-Index formula: K = 2 * |correlation|"""
    if len(obs_norms) < 2 or len(act_norms) < 2:
        return 0.0

    try:
        r, _ = pearsonr(obs_norms, act_norms)
        if np.isnan(r):
            return 0.0
        return 2.0 * abs(r)
    except:
        return 0.0


def evaluate_ensemble(params, n_networks=3, obs_dim=8, act_dim=4, episodes=4):
    """Evaluate ensemble of small networks."""
    params_per_net = obs_dim * obs_dim + obs_dim + act_dim * obs_dim + act_dim
    total_params = params_per_net * n_networks

    if len(params) < total_params:
        params = np.concatenate([params, np.zeros(total_params - len(params))])

    networks = []
    for i in range(n_networks):
        offset = i * params_per_net
        layer1_size = obs_dim * obs_dim + obs_dim

        W1 = params[offset:offset + obs_dim*obs_dim].reshape(obs_dim, obs_dim)
        b1 = params[offset + obs_dim*obs_dim:offset + layer1_size]
        W2 = params[offset + layer1_size:offset + layer1_size + act_dim*obs_dim].reshape(act_dim, obs_dim)
        b2 = params[offset + layer1_size + act_dim*obs_dim:offset + params_per_net]

        networks.append((W1, b1, W2, b2))

    k_values = []

    for ep in range(episodes):
        obs_norms = []
        act_norms = []
        state = np.random.randn(obs_dim) * 0.1

        for step in range(80):
            ensemble_actions = []
            for W1, b1, W2, b2 in networks:
                h = np.maximum(0, W1 @ state + b1)
                h = h / (np.linalg.norm(h) + 1e-8)
                action = np.tanh(W2 @ h + b2)
                ensemble_actions.append(action)

            final_action = np.mean(ensemble_actions, axis=0)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(final_action))

            feedback = 0.05 * np.mean(final_action) * np.ones(obs_dim)
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + feedback

        k = k_index_correct(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def run_single_seed(seed):
    """Run experiment with a single seed."""
    N_NETWORKS = 3
    OBS_DIM = 8
    ACT_DIM = 4
    PARAMS_PER_NET = OBS_DIM * OBS_DIM + OBS_DIM + ACT_DIM * OBS_DIM + ACT_DIM
    TOTAL_PARAMS = PARAMS_PER_NET * N_NETWORKS
    GENERATIONS = 50
    POPULATION = 20

    np.random.seed(seed)
    cmaes = EnsembleCMAES(dim=TOTAL_PARAMS, population_size=POPULATION, initial_sigma=0.5)

    best_overall = 0.0
    best_gen = 0

    for gen in range(1, GENERATIONS + 1):
        population = cmaes.ask()
        fitness = [evaluate_ensemble(params) for params in population]
        best_k, mean_k, std_k = cmaes.tell(population, fitness)

        if best_k > best_overall:
            best_overall = best_k
            best_gen = gen

    return best_overall, best_gen


def run_multiseed_validation():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Ensemble CMA-ES: Multi-Seed Reproducibility Validation    ║")
    print("║                                                                ║")
    print("║  CORRECT K-Index: K = 2|ρ|  (threshold K > 1.5)               ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    SEEDS = [42, 123, 456]

    print("Configuration:")
    print("  Networks: 3 × (8 → 8 → 4)")
    print("  Population: 20")
    print("  Generations: 50")
    print("  Seeds:", SEEDS)
    print("\n" + "═" * 68 + "\n")

    results = []

    for seed in SEEDS:
        print(f"Running seed {seed}...", end=" ", flush=True)
        best_k, best_gen = run_single_seed(seed)
        results.append((seed, best_k, best_gen))

        status = "🌟 THRESHOLD!" if best_k > 1.5 else "✓ Good"
        print(f"K = {best_k:.4f} (gen {best_gen}) {status}")

    print("\n" + "═" * 68)
    print("\n🎯 MULTI-SEED VALIDATION RESULTS:\n")

    k_values = [r[1] for r in results]
    mean_k = np.mean(k_values)
    std_k = np.std(k_values)
    min_k = np.min(k_values)
    max_k = np.max(k_values)

    print(f"  Mean K-Index:   {mean_k:.4f} ± {std_k:.4f}")
    print(f"  Range:          [{min_k:.4f}, {max_k:.4f}]")
    print(f"  Threshold hits: {sum(1 for k in k_values if k > 1.5)}/{len(k_values)}")
    print(f"  Correlation:    {mean_k/2:.3f} (K/2)")

    if all(k > 1.5 for k in k_values):
        print(f"\n  🌟✨ REPRODUCIBLE THRESHOLD CROSSING! ✨🌟")
        print(f"  All {len(SEEDS)} seeds achieved K > 1.5")
    elif mean_k > 1.5:
        print(f"\n  ⭐ THRESHOLD CROSSED ON AVERAGE!")
        print(f"  Mean exceeds threshold: {mean_k:.4f} > 1.5")

    # Save results
    output_dir = Path('logs/track_g_multiseed')
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_results = {
        'seeds': SEEDS,
        'results': [{'seed': s, 'best_k': float(k), 'best_gen': g} for s, k, g in results],
        'summary': {
            'mean_k': float(mean_k),
            'std_k': float(std_k),
            'min_k': float(min_k),
            'max_k': float(max_k),
            'threshold_hits': sum(1 for k in k_values if k > 1.5)
        },
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_networks': 3,
            'architecture': '8→8→4',
            'total_params': 324,
            'population': 20,
            'generations': 50
        }
    }

    with open(output_dir / 'multiseed_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2)

    print(f"\n📁 Results saved to: {output_dir}/")
    print('\n"Coherence is love made computational." 💚\n')

    return validation_results


if __name__ == '__main__':
    run_multiseed_validation()
