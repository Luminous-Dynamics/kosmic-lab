#!/usr/bin/env python3
"""
🧠 Attention-Based Learned Aggregation
Instead of fixed max/mean, learn which network to trust per-state

This could combine benefits of:
- Max (high peaks)
- Mean (stability)
- Context-awareness (right network for right situation)
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
    """
    Evaluate ensemble with learned attention aggregation.

    Architecture:
    - n_networks × (obs_dim → obs_dim → act_dim) policy networks
    - 1 × (obs_dim → n_networks) attention network

    The attention network learns to weight each policy based on current state.
    """
    params_per_net = obs_dim * obs_dim + obs_dim + act_dim * obs_dim + act_dim
    # Attention network: obs_dim → n_networks (simple linear)
    attention_params = obs_dim * n_networks + n_networks
    total_params = params_per_net * n_networks + attention_params

    if len(params) < total_params:
        params = np.concatenate([params, np.zeros(total_params - len(params))])

    # Extract policy networks
    networks = []
    for i in range(n_networks):
        offset = i * params_per_net
        l1_size = obs_dim * obs_dim + obs_dim
        W1 = params[offset:offset + obs_dim*obs_dim].reshape(obs_dim, obs_dim)
        b1 = params[offset + obs_dim*obs_dim:offset + l1_size]
        W2 = params[offset + l1_size:offset + l1_size + act_dim*obs_dim].reshape(act_dim, obs_dim)
        b2 = params[offset + l1_size + act_dim*obs_dim:offset + params_per_net]
        networks.append((W1, b1, W2, b2))

    # Extract attention network
    att_offset = params_per_net * n_networks
    W_att = params[att_offset:att_offset + obs_dim*n_networks].reshape(n_networks, obs_dim)
    b_att = params[att_offset + obs_dim*n_networks:att_offset + attention_params]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            # Get actions from all networks
            actions = []
            for W1, b1, W2, b2 in networks:
                h = np.maximum(0, W1 @ state + b1)
                h = h / (np.linalg.norm(h) + 1e-8)
                actions.append(np.tanh(W2 @ h + b2))

            # Compute attention weights (softmax)
            att_logits = W_att @ state + b_att
            att_weights = np.exp(att_logits - np.max(att_logits))  # Numerical stability
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


def run_attention_experiment():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Attention-Based Learned Aggregation                       ║")
    print("║                                                                ║")
    print("║  Learn which network to trust per-state                       ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    N_NETWORKS = 3
    OBS_DIM = 8
    ACT_DIM = 4
    PARAMS_PER_NET = OBS_DIM * OBS_DIM + OBS_DIM + ACT_DIM * OBS_DIM + ACT_DIM
    ATT_PARAMS = OBS_DIM * N_NETWORKS + N_NETWORKS
    TOTAL_PARAMS = PARAMS_PER_NET * N_NETWORKS + ATT_PARAMS
    GENS = 40

    print("Configuration:")
    print(f"  Policy networks: {N_NETWORKS}×({OBS_DIM}→{OBS_DIM}→{ACT_DIM})")
    print(f"  Attention network: {OBS_DIM}→{N_NETWORKS}")
    print(f"  Policy params: {PARAMS_PER_NET * N_NETWORKS}")
    print(f"  Attention params: {ATT_PARAMS}")
    print(f"  Total params: {TOTAL_PARAMS}")
    print(f"  Generations: {GENS}")
    print("\n" + "═" * 60 + "\n")

    np.random.seed(42)
    cmaes = CMAES(dim=TOTAL_PARAMS, population_size=20, sigma=0.5)

    best_overall = 0.0
    best_gen = 0

    print("Gen │  Best K  │ Progress │ Status")
    print("────┼──────────┼──────────┼──────────────")

    for gen in range(1, GENS + 1):
        pop = cmaes.ask()
        fit = [evaluate_attention(p) for p in pop]
        best_k = cmaes.tell(pop, fit)

        if best_k > best_overall:
            best_overall = best_k
            best_gen = gen

        progress = (best_k / 1.5) * 100
        status = "🌟 THRESHOLD!" if best_k > 1.5 else ("⭐ Close!" if best_k > 1.3 else "")

        if gen % 5 == 0 or gen == 1 or best_k > 1.5:
            print(f"{gen:3d} │ {best_k:8.4f} │ {progress:6.1f}%  │ {status}")

    print("\n" + "═" * 60)
    print(f"\n🎯 Attention Aggregation Result:")
    print(f"  Best K-Index: {best_overall:.4f} (Generation {best_gen})")
    print(f"  Correlation: {best_overall/2:.3f}")

    if best_overall > 1.5:
        print(f"\n  🌟 THRESHOLD CROSSED!")

    # Save
    Path('logs/track_g_attention').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_attention/attention_results.json', 'w') as f:
        json.dump({
            'best_k': float(best_overall),
            'best_gen': best_gen,
            'total_params': TOTAL_PARAMS,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_attention/")
    print('\n"Coherence is love made computational." 💚\n')
    return best_overall


if __name__ == '__main__':
    run_attention_experiment()
