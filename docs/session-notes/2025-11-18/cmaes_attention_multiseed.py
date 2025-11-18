#!/usr/bin/env python3
"""
🧠 Attention Aggregation Multi-Seed Validation
Validates K=1.8273 finding with seeds [42, 123, 456]
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
    for _ in range(4):
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


def run_single_seed(seed, gens=40):
    np.random.seed(seed)
    cmaes = CMAES(dim=351, population_size=20, sigma=0.5)
    best = 0.0
    best_gen = 0
    for gen in range(1, gens + 1):
        pop = cmaes.ask()
        fit = [evaluate_attention(p) for p in pop]
        best_k = cmaes.tell(pop, fit)
        if best_k > best:
            best = best_k
            best_gen = gen
    return best, best_gen


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Attention Aggregation Multi-Seed Validation               ║")
    print("║                                                                ║")
    print("║  3×(8→8→4) + attention network (8→3)                          ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    SEEDS = [42, 123, 456]
    results = []

    print("Seed │  Best K  │ Gen │ Status")
    print("─────┼──────────┼─────┼──────────────")

    for seed in SEEDS:
        best_k, best_gen = run_single_seed(seed)
        results.append({'seed': seed, 'k': float(best_k), 'gen': best_gen})
        status = "🌟🌟 K>1.8!" if best_k > 1.8 else ("🌟" if best_k > 1.5 else "⭐")
        print(f" {seed:3d} │ {best_k:8.4f} │ {best_gen:3d} │ {status}")

    k_vals = [r['k'] for r in results]
    mean_k = np.mean(k_vals)
    std_k = np.std(k_vals)

    print("\n" + "═" * 50)
    print(f"\n🎯 Attention Aggregation Validation:")
    print(f"  Mean K: {mean_k:.4f} ± {std_k:.4f}")
    print(f"  Range: [{min(k_vals):.4f}, {max(k_vals):.4f}]")
    print(f"  Threshold hits: {sum(1 for k in k_vals if k > 1.5)}/{len(k_vals)}")
    print(f"  K > 1.8 hits: {sum(1 for k in k_vals if k > 1.8)}/{len(k_vals)}")

    if mean_k > 1.8:
        print(f"\n  🌟🌟✨ MEAN K > 1.8 ACHIEVED! ✨🌟🌟")
    elif all(k > 1.5 for k in k_vals):
        print(f"\n  🌟✨ REPRODUCIBLE THRESHOLD CROSSING! ✨🌟")

    # Save
    Path('logs/track_g_attention_validation').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_attention_validation/attention_multiseed.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {'mean': float(mean_k), 'std': float(std_k)},
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_attention_validation/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
