#!/usr/bin/env python3
"""
🚀 Push for K > 1.9: Optimized Attention Aggregation
60 generations, population 25, refined hyperparameters
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

class OptimizedCMAES:
    def __init__(self, dim, population_size=25, sigma=0.4):
        self.dim = dim
        self.pop_size = population_size
        self.sigma = sigma
        self.mean = np.random.randn(dim) * 0.05  # Smaller init
        self.C = np.eye(dim)
        self.c_mu = 0.35

    def ask(self):
        L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        return [self.mean + self.sigma * (L @ np.random.randn(self.dim)) for _ in range(self.pop_size)]

    def tell(self, population, fitness):
        indices = np.argsort(fitness)[::-1]
        elite_size = max(5, self.pop_size // 4)
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
        self.sigma *= 1.02 if (np.max(fitness) - np.mean(fitness)) > 0.02 else 0.98
        self.sigma = np.clip(self.sigma, 0.01, 1.0)
        return np.max(fitness), np.mean(fitness)


def k_index(obs_norms, act_norms):
    if len(obs_norms) < 2:
        return 0.0
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 0.0 if np.isnan(r) else 2.0 * abs(r)
    except:
        return 0.0


def evaluate_attention(params, n_networks=3, obs_dim=8, act_dim=4):
    params_per_net = obs_dim * obs_dim + obs_dim + act_dim * obs_dim + act_dim
    attention_params = obs_dim * n_networks + n_networks
    total_params = params_per_net * n_networks + attention_params

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

    att_offset = params_per_net * n_networks
    W_att = params[att_offset:att_offset + obs_dim*n_networks].reshape(n_networks, obs_dim)
    b_att = params[att_offset + obs_dim*n_networks:att_offset + attention_params]

    k_values = []
    for _ in range(5):  # More episodes
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            actions = []
            for W1, b1, W2, b2 in networks:
                h = np.maximum(0, W1 @ state + b1)
                h = h / (np.linalg.norm(h) + 1e-8)
                actions.append(np.tanh(W2 @ h + b2))

            att_logits = W_att @ state + b_att
            att_weights = np.exp(att_logits - np.max(att_logits))
            att_weights = att_weights / (att_weights.sum() + 1e-8)
            final_action = np.sum([w * a for w, a in zip(att_weights, actions)], axis=0)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(final_action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(final_action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🚀 Push for K > 1.9: Optimized Attention Aggregation         ║")
    print("║                                                                ║")
    print("║  60 generations, population 25, refined hyperparameters       ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    TOTAL_PARAMS = 351  # 3×108 + 27
    GENS = 60
    POP = 25

    print("Configuration:")
    print("  Architecture: 3×(8→8→4) + attention (8→3)")
    print("  Parameters: 351")
    print("  Population: 25 (increased)")
    print("  Generations: 60 (extended)")
    print("  Episodes: 5 (increased)")
    print("  Target: K > 1.9")
    print("\n" + "═" * 60 + "\n")

    np.random.seed(42)
    cmaes = OptimizedCMAES(dim=TOTAL_PARAMS, population_size=POP, sigma=0.4)

    best_overall = 0.0
    best_gen = 0

    print("Gen │  Best K  │  Mean K  │ Progress")
    print("────┼──────────┼──────────┼────────────────────────────")

    for gen in range(1, GENS + 1):
        pop = cmaes.ask()
        fit = [evaluate_attention(p) for p in pop]
        best_k, mean_k = cmaes.tell(pop, fit)

        if best_k > best_overall:
            best_overall = best_k
            best_gen = gen

        progress = (best_k / 1.9) * 100
        status = ""
        if best_k > 1.9:
            status = "🌟🌟🌟 TARGET!"
        elif best_k > 1.8:
            status = "🌟🌟 K>1.8!"
        elif best_k > 1.5:
            status = "🌟"

        if gen % 5 == 0 or gen == 1 or best_k > 1.85:
            print(f"{gen:3d} │ {best_k:8.4f} │ {mean_k:8.4f} │ {progress:5.1f}% {status}")

    print("\n" + "═" * 60)
    print(f"\n🎯 FINAL RESULT:")
    print(f"  Best K-Index: {best_overall:.4f} (Generation {best_gen})")
    print(f"  Progress to K=1.9: {(best_overall/1.9)*100:.1f}%")
    print(f"  Correlation: {best_overall/2:.3f}")

    if best_overall > 1.9:
        print(f"\n  🌟🌟🌟✨ K > 1.9 ACHIEVED! ✨🌟🌟🌟")
    elif best_overall > 1.8:
        print(f"\n  🌟🌟 K > 1.8 achieved: {best_overall:.4f}")

    # Save
    Path('logs/track_g_push_1_9').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_push_1_9/push_results.json', 'w') as f:
        json.dump({
            'best_k': float(best_overall),
            'best_gen': best_gen,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_push_1_9/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
