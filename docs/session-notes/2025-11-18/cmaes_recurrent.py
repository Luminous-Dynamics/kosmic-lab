#!/usr/bin/env python3
"""
🧠 Recurrent Architecture: Adding Memory for Temporal Coherence

Hypothesis: Memory allows better temporal coherence
Architecture: obs + prev_hidden → hidden → action
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


def evaluate_recurrent(params, obs_dim=8, hidden_dim=10, act_dim=4):
    """
    Evaluate recurrent network with memory.
    Input: [obs, prev_hidden] → hidden → action
    """
    # Parameters:
    # W_ih: (obs_dim + hidden_dim) → hidden_dim
    # b_h: hidden_dim
    # W_ho: hidden_dim → act_dim
    # b_o: act_dim

    input_dim = obs_dim + hidden_dim
    l1_params = hidden_dim * input_dim + hidden_dim
    l2_params = act_dim * hidden_dim + act_dim
    total_params = l1_params + l2_params

    if len(params) < total_params:
        params = np.concatenate([params, np.zeros(total_params - len(params))])

    W_ih = params[:hidden_dim * input_dim].reshape(hidden_dim, input_dim)
    b_h = params[hidden_dim * input_dim:l1_params]
    W_ho = params[l1_params:l1_params + act_dim * hidden_dim].reshape(act_dim, hidden_dim)
    b_o = params[l1_params + act_dim * hidden_dim:total_params]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1
        hidden = np.zeros(hidden_dim)  # Initial hidden state

        for _ in range(80):
            # Concatenate observation and previous hidden state
            combined = np.concatenate([state, hidden])

            # Recurrent layer
            hidden = np.tanh(W_ih @ combined + b_h)
            hidden = hidden / (np.linalg.norm(hidden) + 1e-8)

            # Output layer
            action = np.tanh(W_ho @ hidden + b_o)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        k = k_index(np.array(obs_norms), np.array(act_norms))
        if k > 0:
            k_values.append(k)

    return np.mean(k_values) if k_values else 0.0


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Recurrent Architecture: Memory for Temporal Coherence     ║")
    print("║                                                                ║")
    print("║  [obs, prev_hidden] → hidden → action                         ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    OBS_DIM = 8
    HIDDEN_DIM = 10
    ACT_DIM = 4

    INPUT_DIM = OBS_DIM + HIDDEN_DIM
    l1 = HIDDEN_DIM * INPUT_DIM + HIDDEN_DIM
    l2 = ACT_DIM * HIDDEN_DIM + ACT_DIM
    TOTAL = l1 + l2

    GENS = 50

    print("Configuration:")
    print(f"  Input: obs({OBS_DIM}) + hidden({HIDDEN_DIM}) = {INPUT_DIM}")
    print(f"  Hidden: {HIDDEN_DIM}")
    print(f"  Output: {ACT_DIM}")
    print(f"  Total params: {TOTAL}")
    print(f"  Generations: {GENS}")
    print("\n" + "═" * 60 + "\n")

    np.random.seed(42)
    cmaes = CMAES(dim=TOTAL, population_size=20, sigma=0.5)

    best_overall = 0.0
    best_gen = 0

    print("Gen │  Best K  │ Status")
    print("────┼──────────┼──────────────")

    for gen in range(1, GENS + 1):
        pop = cmaes.ask()
        fit = [evaluate_recurrent(p) for p in pop]
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
        elif best_k > 1.3:
            status = "⭐"

        if gen % 5 == 0 or gen == 1 or best_k > 1.8:
            print(f"{gen:3d} │ {best_k:8.4f} │ {status}")

    print("\n" + "═" * 60)
    print(f"\n🎯 Recurrent Architecture Result:")
    print(f"  Best K-Index: {best_overall:.4f} (Generation {best_gen})")
    print(f"  Correlation: {best_overall/2:.3f}")

    if best_overall > 1.9:
        print(f"\n  🌟🌟🌟 K > 1.9 ACHIEVED!")
    elif best_overall > 1.8:
        print(f"\n  🌟🌟 Memory helps! K > 1.8")

    print(f"\n📊 Comparison:")
    print(f"  Feedforward (8→8→4):     K ≈ 1.79")
    print(f"  Deep (8→12→8→4):         K = 1.88")
    print(f"  Recurrent ({TOTAL}p):        K = {best_overall:.4f}")

    # Save
    Path('logs/track_g_recurrent').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_recurrent/recurrent_results.json', 'w') as f:
        json.dump({
            'best_k': float(best_overall),
            'best_gen': best_gen,
            'params': TOTAL,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_recurrent/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
