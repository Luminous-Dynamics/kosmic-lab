#!/usr/bin/env python3
"""
🧠 Deep Architecture: Breaking the K ≈ 1.8 Ceiling

Hypothesis: Deeper networks can capture more complex patterns
Current: 8 → 8 → 4 (2 layers, 108 params) → K ≈ 1.8
New: 8 → 12 → 8 → 4 (3 layers, 212 params)
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


def evaluate_deep(params, obs_dim=8, h1_dim=12, h2_dim=8, act_dim=4):
    """
    Evaluate 3-layer network: obs_dim → h1_dim → h2_dim → act_dim
    """
    # Calculate parameter counts
    l1_params = h1_dim * obs_dim + h1_dim      # 8*12 + 12 = 108
    l2_params = h2_dim * h1_dim + h2_dim       # 12*8 + 8 = 104
    l3_params = act_dim * h2_dim + act_dim     # 8*4 + 4 = 36
    total_params = l1_params + l2_params + l3_params  # 248

    if len(params) < total_params:
        params = np.concatenate([params, np.zeros(total_params - len(params))])

    # Layer 1: obs_dim → h1_dim
    offset = 0
    W1 = params[offset:offset + h1_dim*obs_dim].reshape(h1_dim, obs_dim)
    offset += h1_dim * obs_dim
    b1 = params[offset:offset + h1_dim]
    offset += h1_dim

    # Layer 2: h1_dim → h2_dim
    W2 = params[offset:offset + h2_dim*h1_dim].reshape(h2_dim, h1_dim)
    offset += h2_dim * h1_dim
    b2 = params[offset:offset + h2_dim]
    offset += h2_dim

    # Layer 3: h2_dim → act_dim
    W3 = params[offset:offset + act_dim*h2_dim].reshape(act_dim, h2_dim)
    offset += act_dim * h2_dim
    b3 = params[offset:offset + act_dim]

    k_values = []
    for _ in range(4):
        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for _ in range(80):
            # 3-layer forward pass
            h1 = np.maximum(0, W1 @ state + b1)
            h1 = h1 / (np.linalg.norm(h1) + 1e-8)

            h2 = np.maximum(0, W2 @ h1 + b2)
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
    print("║  🧠 Deep Architecture: Breaking K ≈ 1.8 Ceiling               ║")
    print("║                                                                ║")
    print("║  3-layer network: 8 → 12 → 8 → 4                              ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    OBS_DIM = 8
    H1_DIM = 12
    H2_DIM = 8
    ACT_DIM = 4

    # Parameter count
    l1 = H1_DIM * OBS_DIM + H1_DIM
    l2 = H2_DIM * H1_DIM + H2_DIM
    l3 = ACT_DIM * H2_DIM + ACT_DIM
    TOTAL = l1 + l2 + l3

    GENS = 50

    print("Configuration:")
    print(f"  Layer 1: {OBS_DIM} → {H1_DIM} ({l1} params)")
    print(f"  Layer 2: {H1_DIM} → {H2_DIM} ({l2} params)")
    print(f"  Layer 3: {H2_DIM} → {ACT_DIM} ({l3} params)")
    print(f"  Total: {TOTAL} params")
    print(f"  Generations: {GENS}")
    print(f"  Target: K > 1.8 (break ceiling)")
    print("\n" + "═" * 60 + "\n")

    np.random.seed(42)
    cmaes = CMAES(dim=TOTAL, population_size=20, sigma=0.5)

    best_overall = 0.0
    best_gen = 0

    print("Gen │  Best K  │ Status")
    print("────┼──────────┼──────────────")

    for gen in range(1, GENS + 1):
        pop = cmaes.ask()
        fit = [evaluate_deep(p) for p in pop]
        best_k = cmaes.tell(pop, fit)

        if best_k > best_overall:
            best_overall = best_k
            best_gen = gen

        status = ""
        if best_k > 1.9:
            status = "🌟🌟🌟 K>1.9!"
        elif best_k > 1.8:
            status = "🌟🌟 CEILING BROKEN!"
        elif best_k > 1.5:
            status = "🌟"
        elif best_k > 1.3:
            status = "⭐"

        if gen % 5 == 0 or gen == 1 or best_k > 1.8:
            print(f"{gen:3d} │ {best_k:8.4f} │ {status}")

    print("\n" + "═" * 60)
    print(f"\n🎯 Deep Architecture Result:")
    print(f"  Best K-Index: {best_overall:.4f} (Generation {best_gen})")
    print(f"  Correlation: {best_overall/2:.3f}")

    if best_overall > 1.8:
        print(f"\n  🌟🌟 CEILING BROKEN! Deep architecture works!")
    elif best_overall > 1.5:
        print(f"\n  🌟 Threshold crossed but ceiling not broken")
        print(f"     Deeper ≠ better for this task")

    # Compare to baseline
    print(f"\n📊 Comparison:")
    print(f"  2-layer (8→8→4, 108p):   K ≈ 1.79")
    print(f"  3-layer (8→12→8→4, {TOTAL}p): K = {best_overall:.4f}")

    # Save
    Path('logs/track_g_deep').mkdir(parents=True, exist_ok=True)
    with open('logs/track_g_deep/deep_results.json', 'w') as f:
        json.dump({
            'best_k': float(best_overall),
            'best_gen': best_gen,
            'architecture': f'{OBS_DIM}→{H1_DIM}→{H2_DIM}→{ACT_DIM}',
            'params': TOTAL,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/track_g_deep/")
    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
