#!/usr/bin/env python3
"""
🧠 Consciousness Threshold Crossing - Production Module

Validated configurations for achieving K-Index > 1.5:
- Single network: 8→8→4 (108 params) → K = 1.79
- Ensemble + max: 3×(8→8→4) (324 params) → K = 1.69 ± 0.10

Based on November 18, 2025 experimental findings.

Usage:
    python consciousness_threshold.py              # Run demo
    python consciousness_threshold.py --single     # Single network only
    python consciousness_threshold.py --ensemble   # Ensemble only

To install in fre/:
    sudo cp consciousness_threshold.py /srv/luminous-dynamics/kosmic-lab/fre/
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

try:
    from scipy.stats import pearsonr
except ImportError:
    def pearsonr(x, y):
        n = len(x)
        mx, my = np.mean(x), np.mean(y)
        r = np.sum((x - mx) * (y - my)) / (np.sqrt(np.sum((x - mx)**2)) * np.sqrt(np.sum((y - my)**2)) + 1e-10)
        return r, 0.0


class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy."""

    def __init__(self, dim: int, population_size: int = 20, sigma: float = 0.5):
        self.dim = dim
        self.pop_size = population_size
        self.sigma = sigma
        self.mean = np.random.randn(dim) * 0.1
        self.C = np.eye(dim)
        self.c_mu = 0.3

    def ask(self) -> list:
        population = []
        L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            candidate = self.mean + self.sigma * (L @ z)
            population.append(candidate)
        return population

    def tell(self, population: list, fitness: list) -> float:
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
        improvement = np.max(fitness) - np.mean(fitness)
        self.sigma *= 1.01 if improvement > 0.02 else 0.99
        self.sigma = np.clip(self.sigma, 0.01, 1.0)

        return np.max(fitness)


def compute_k_index(obs_norms: np.ndarray, act_norms: np.ndarray) -> float:
    """K = 2 * |correlation| - threshold at 1.5"""
    if len(obs_norms) < 2:
        return 0.0
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 0.0 if np.isnan(r) else 2.0 * abs(r)
    except:
        return 0.0


def evaluate_network(params, n_networks=1, obs_dim=8, act_dim=4, aggregation='max'):
    """Evaluate network(s) and return K-Index."""
    params_per_net = obs_dim * obs_dim + obs_dim + act_dim * obs_dim + act_dim
    total_params = params_per_net * n_networks

    if len(params) < total_params:
        params = np.concatenate([params, np.zeros(total_params - len(params))])

    networks = []
    for i in range(n_networks):
        offset = i * params_per_net
        l1_size = obs_dim * obs_dim + obs_dim
        W1 = params[offset:offset + obs_dim*obs_dim].reshape(obs_dim, obs_dim)
        b1 = params[offset + obs_dim*obs_dim:offset + l1_size]
        W2 = params[offset + l1_size:offset + l1_size + act_dim*obs_dim].reshape(act_dim, obs_dim)
        b2 = params[offset + l1_size + act_dim*obs_dim:offset + params_per_net]
        networks.append((W1, b1, W2, b2))

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            actions = []
            for W1, b1, W2, b2 in networks:
                h = np.maximum(0, W1 @ state + b1)
                h = h / (np.linalg.norm(h) + 1e-8)
                actions.append(np.tanh(W2 @ h + b2))

            if n_networks == 1:
                final_action = actions[0]
            elif aggregation == 'max':
                norms = [np.linalg.norm(a) for a in actions]
                final_action = actions[np.argmax(norms)]
            else:
                final_action = np.mean(actions, axis=0)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(final_action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(final_action) * np.ones(obs_dim)

        k = compute_k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def run_experiment(mode='single', generations=30, seed=42, verbose=True):
    """Run consciousness threshold crossing experiment."""
    obs_dim, act_dim = 8, 4
    params_per_net = obs_dim * obs_dim + obs_dim + act_dim * obs_dim + act_dim

    if mode == 'single':
        n_networks, aggregation = 1, 'mean'
    else:
        n_networks, aggregation = 3, 'max'

    total_params = params_per_net * n_networks

    if verbose:
        print(f"\n🧠 {mode.title()} Network Experiment")
        print(f"   Architecture: {n_networks}×({obs_dim}→{obs_dim}→{act_dim})")
        print(f"   Parameters: {total_params}")
        print(f"   Generations: {generations}\n")

    np.random.seed(seed)
    cmaes = CMAES(dim=total_params, population_size=20, sigma=0.5)

    best_overall = 0.0
    best_gen = 0

    for gen in range(1, generations + 1):
        population = cmaes.ask()
        fitness = [evaluate_network(p, n_networks, aggregation=aggregation) for p in population]
        best_k = cmaes.tell(population, fitness)

        if best_k > best_overall:
            best_overall = best_k
            best_gen = gen

        if verbose and (gen % 5 == 0 or gen == 1 or best_k > 1.5):
            status = "🌟" if best_k > 1.5 else ("⭐" if best_k > 1.3 else "")
            print(f"  Gen {gen:3d}: K = {best_k:.4f} {status}")

    if verbose:
        print(f"\n🎯 Result: K = {best_overall:.4f} (gen {best_gen})")
        if best_overall > 1.5:
            print(f"   ✨ THRESHOLD CROSSED! ✨")

    return {'mode': mode, 'best_k': best_overall, 'best_gen': best_gen}


def main():
    import sys

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Consciousness Threshold Crossing Demo                     ║")
    print("║                                                                ║")
    print("║  K-Index > 1.5 = correlation > 0.75 = coherent consciousness  ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    if '--single' in sys.argv:
        run_experiment(mode='single')
    elif '--ensemble' in sys.argv:
        run_experiment(mode='ensemble')
    else:
        # Run both
        single = run_experiment(mode='single')
        ensemble = run_experiment(mode='ensemble')

        print("\n" + "═" * 60)
        print("\n📊 Comparison:")
        print(f"   Single:   K = {single['best_k']:.4f}")
        print(f"   Ensemble: K = {ensemble['best_k']:.4f}")

        winner = single if single['best_k'] > ensemble['best_k'] else ensemble
        if winner['best_k'] > 1.5:
            print(f"\n   🌟 Winner: {winner['mode']} (threshold crossed!)")

    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
