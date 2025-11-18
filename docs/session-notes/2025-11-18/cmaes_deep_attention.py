#!/usr/bin/env python3
"""
🧠 Deep + Attention Hybrid: Push for K > 1.9

Combining best of both worlds:
- 3 deep networks (each 8→12→8→4)
- Attention mechanism to weight outputs
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


def evaluate_deep_attention(params, n_networks=3, obs_dim=8, h1_dim=12, h2_dim=8, act_dim=4):
    """Evaluate ensemble of deep networks with attention."""
    # Params per deep network
    l1 = h1_dim * obs_dim + h1_dim
    l2 = h2_dim * h1_dim + h2_dim
    l3 = act_dim * h2_dim + act_dim
    params_per_net = l1 + l2 + l3  # 248

    # Attention params
    att_params = obs_dim * n_networks + n_networks  # 27

    total = params_per_net * n_networks + att_params

    if len(params) < total:
        params = np.concatenate([params, np.zeros(total - len(params))])

    # Extract deep networks
    networks = []
    for i in range(n_networks):
        offset = i * params_per_net
        W1 = params[offset:offset + h1_dim*obs_dim].reshape(h1_dim, obs_dim)
        offset_local = h1_dim * obs_dim
        b1 = params[offset + offset_local:offset + offset_local + h1_dim]
        offset_local += h1_dim
        W2 = params[offset + offset_local:offset + offset_local + h2_dim*h1_dim].reshape(h2_dim, h1_dim)
        offset_local += h2_dim * h1_dim
        b2 = params[offset + offset_local:offset + offset_local + h2_dim]
        offset_local += h2_dim
        W3 = params[offset + offset_local:offset + offset_local + act_dim*h2_dim].reshape(act_dim, h2_dim)
        offset_local += act_dim * h2_dim
        b3 = params[offset + offset_local:offset + offset_local + act_dim]
        networks.append((W1, b1, W2, b2, W3, b3))

    # Extract attention
    att_offset = params_per_net * n_networks
    W_att = params[att_offset:att_offset + obs_dim*n_networks].reshape(n_networks, obs_dim)
    b_att = params[att_offset + obs_dim*n_networks:att_offset + att_params]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            # Forward through all deep networks
            actions = []
            for W1, b1, W2, b2, W3, b3 in networks:
                h1 = np.maximum(0, W1 @ state + b1)
                h1 = h1 / (np.linalg.norm(h1) + 1e-8)
                h2 = np.maximum(0, W2 @ h1 + b2)
                h2 = h2 / (np.linalg.norm(h2) + 1e-8)
                action = np.tanh(W3 @ h2 + b3)
                actions.append(action)

            # Attention weights
            att_logits = W_att @ state + b_att
            att_weights = np.exp(att_logits - np.max(att_logits))
            att_weights = att_weights / (att_weights.sum() + 1e-8)

            # Weighted combination
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
    print("║  🧠 Deep + Attention Hybrid: Push for K > 1.9                 ║")
    print("║                                                                ║")
    print("║  3 × (8→12→8→4) + attention (8→3)                             ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    PARAMS_PER_NET = 248
    N_NETWORKS = 3
    ATT_PARAMS = 27
    TOTAL = PARAMS_PER_NET * N_NETWORKS + ATT_PARAMS  # 771
    GENS = 50

    print("Configuration:")
    print(f"  Deep networks: {N_NETWORKS} × (8→12→8→4)")
    print(f"  Params per network: {PARAMS_PER_NET}")
    print(f"  Attention params: {ATT_PARAMS}")
    print(f"  Total params: {TOTAL}")
    print(f"  Generations: {GENS}")
    print(f"  Target: K > 1.9")
    print("\n" + "═" * 60 + "\n")

    np.random.seed(42)
    cmaes = CMAES(dim=TOTAL, population_size=20, sigma=0.5)

    best_overall = 0.0
    best_gen = 0

    print("Gen │  Best K  │ Status")
    print("────┼──────────┼──────────────")

    for gen in range(1, GENS + 1):
        pop = cmaes.ask()
        fit = [evaluate_deep_attention(p) for p in pop]
        best_k = cmaes.tell(pop, fit)

        if best_k > best_overall:
            best_overall = best_k
            best_gen = gen

        status = ""
        if best_k > 1.9:
            status = "🌟🌟🌟 K>1.9!"
        elif best_k > 1.8:
            status = "🌟🌟 K>1.8!"
        elif best_k > 1.5:
            status = "🌟"

        if gen % 5 == 0 or gen == 1 or best_k > 1.85:
            print(f"{gen:3d} │ {best_k:8.4f} │ {status}")

    print("\n" + "═" * 60)
    print(f"\n🎯 Deep + Attention Result:")
    print(f"  Best K-Index: {best_overall:.4f} (Generation {best_gen})")
    print(f"  Correlation: {best_overall/2:.3f}")

    if best_overall > 1.9:
        print(f"\n  🌟🌟🌟✨ K > 1.9 ACHIEVED! ✨🌟🌟🌟")
    elif best_overall > 1.8:
        print(f"\n  🌟🌟 K > 1.8 achieved")

    print(f"\n📊 Comparison to best methods:")
    print(f"  Single deep (248p):      K = 1.88")
    print(f"  Shallow attention (351p): K = 1.83")
    print(f"  Deep + attention ({TOTAL}p):  K = {best_overall:.4f}")

    # Save
    Path('logs/track_g_deep_attention').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_deep_attention/deep_attention_results.json', 'w') as f:
        json.dump({
            'best_k': float(best_overall),
            'best_gen': best_gen,
            'params': TOTAL,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_deep_attention/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
