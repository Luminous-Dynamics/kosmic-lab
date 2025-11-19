#!/usr/bin/env python3
"""
🧠 Full 7-Harmony K-Index Computation

Compare simplified K (correlation-only) vs full 7-Harmony K-Index.
Based on K_INDEX_MATHEMATICAL_FORMALISM.md

Hypothesis: Full K-Index captures more meaningful coherence than correlation alone.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from collections import Counter


class CMAES:
    """CMA-ES optimizer."""
    def __init__(self, dim, population_size=30, sigma=0.5):
        self.dim = dim
        self.pop_size = population_size
        self.sigma = sigma
        self.mean = np.random.randn(dim) * 0.1
        self.C = np.eye(dim)
        self.c_mu = 0.3

    def ask(self):
        L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        return [self.mean + self.sigma * (L @ np.random.randn(self.dim))
                for _ in range(self.pop_size)]

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


def compute_h1_resonant_coherence(hidden_states, phi_baseline=1.0):
    """H1: Normalized integrated information (simplified Φ approximation).

    Uses average correlation across hidden layer dimensions as Φ proxy.
    """
    if len(hidden_states) < 2:
        return 0.0

    states = np.array(hidden_states)
    # Simplified Φ: average absolute correlation between dimensions
    n_dims = states.shape[1]
    if n_dims < 2:
        return 0.0

    correlations = []
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            try:
                r, _ = pearsonr(states[:, i], states[:, j])
                if not np.isnan(r):
                    correlations.append(abs(r))
            except:
                pass

    if not correlations:
        return 0.0

    phi = np.mean(correlations)
    return (phi - phi_baseline * 0.1) / (phi_baseline * 0.9)  # Normalize


def compute_h2_diversity(agent_types):
    """H2: Shannon entropy of agent-type distribution."""
    if not agent_types:
        return 0.0

    type_counts = Counter(agent_types)
    n_types = len(type_counts)
    if n_types <= 1:
        return 1.0  # Perfect homogeneity

    total = len(agent_types)
    probs = np.array([count / total for count in type_counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    max_h = np.log(n_types)

    return h / max_h if max_h > 0 else 1.0


def compute_h3_prediction_accuracy(prediction_errors, baseline_fe=10.0):
    """H3: Inverse of mean prediction error."""
    if not prediction_errors:
        return 0.0

    mean_error = np.mean(prediction_errors)
    return max(0.0, 1.0 - (mean_error / baseline_fe))


def compute_h4_behavioral_entropy(action_history, n_bins=10):
    """H4: Shannon entropy of discretized action distribution."""
    if len(action_history) < 2:
        return 0.0

    all_actions = np.array(action_history)
    if all_actions.ndim == 1:
        all_actions = all_actions.reshape(-1, 1)

    n_bins = min(10, len(all_actions) // 10)
    if n_bins < 2:
        return 0.5  # Default moderate entropy

    entropies = []
    for dim in range(all_actions.shape[1]):
        hist, _ = np.histogram(all_actions[:, dim], bins=n_bins)
        hist = hist + 1e-10
        probs = hist / hist.sum()
        entropies.append(-np.sum(probs * np.log(probs)))

    max_entropy = np.log(n_bins)
    return np.mean(entropies) / max_entropy if max_entropy > 0 else 0.0


def compute_h5_mutual_te(time_series):
    """H5: Proportion of bidirectional information flow (simplified).

    Uses cross-correlation as proxy for transfer entropy.
    """
    if len(time_series) < 10:
        return 0.5

    series = np.array(time_series)
    if series.ndim == 1:
        return 0.5

    n_dims = series.shape[1]
    if n_dims < 2:
        return 0.5

    # Compute time-lagged correlations as TE proxy
    mutual_influences = []
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            # Correlation at lag 1
            if len(series) > 1:
                corr_ij = abs(np.corrcoef(series[:-1, i], series[1:, j])[0, 1])
                corr_ji = abs(np.corrcoef(series[:-1, j], series[1:, i])[0, 1])
                if not (np.isnan(corr_ij) or np.isnan(corr_ji)):
                    mutual = min(corr_ij, corr_ji)
                    total = corr_ij + corr_ji + 1e-10
                    mutual_influences.append(2 * mutual / total)

    return np.mean(mutual_influences) if mutual_influences else 0.5


def compute_h6_flow_symmetry(outgoing, incoming):
    """H6: 1 - Jensen-Shannon divergence of flows."""
    if len(outgoing) == 0 or len(incoming) == 0:
        return 0.5

    # Normalize to distributions
    out_norm = np.abs(outgoing) + 1e-10
    in_norm = np.abs(incoming) + 1e-10
    out_prob = out_norm / out_norm.sum()
    in_prob = in_norm / in_norm.sum()

    # JS divergence
    m = 0.5 * (out_prob + in_prob)
    kl_out = np.sum(out_prob * np.log(out_prob / m + 1e-10))
    kl_in = np.sum(in_prob * np.log(in_prob / m + 1e-10))
    js_div = 0.5 * (kl_out + kl_in)

    return 1.0 - np.sqrt(js_div)  # Bounded [0, 1]


def compute_h7_phi_growth(phi_history):
    """H7: Rate of Φ increase over sliding window."""
    if len(phi_history) < 10:
        return 0.0

    recent = phi_history[-100:] if len(phi_history) > 100 else phi_history
    times = np.arange(len(recent))
    phis = np.array(recent)

    # Linear regression slope
    if len(times) > 1:
        times_norm = times / (times.max() + 1e-10)
        slope = np.polyfit(times_norm, phis, deg=1)[0]
        std_phi = np.std(phis) + 1e-10
        return np.tanh(slope / std_phi)

    return 0.0


def compute_full_k_index(metrics):
    """Compute full 7-Harmony K-Index.

    K = Σ(i=1 to 7) w_i × H_i with equal weights.
    """
    weights = np.ones(7) / 7.0

    harmonies = np.array([
        metrics['h1'],  # Resonant Coherence
        metrics['h2'],  # Diversity
        metrics['h3'],  # Prediction Accuracy
        metrics['h4'],  # Behavioral Entropy
        metrics['h5'],  # Mutual TE
        metrics['h6'],  # Flow Symmetry
        metrics['h7'],  # Φ Growth
    ])

    return np.dot(weights, harmonies)


def evaluate_full_7harmony(params):
    """Evaluate 4-layer network with full 7-Harmony K-Index."""
    obs_dim, h1, h2, h3, act_dim = 8, 12, 10, 6, 4
    total = 332

    if len(params) < total:
        params = np.concatenate([params, np.zeros(total - len(params))])

    # Extract weights
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
    simple_k_values = []

    for episode in range(4):
        # Trajectory storage
        hidden_states = []  # For H1
        action_history = []  # For H4
        prediction_errors = []  # For H3
        phi_history = []  # For H7

        obs_norms, act_norms = [], []
        state = np.random.randn(obs_dim) * 0.1

        for t in range(80):
            # Forward pass
            h1_out = np.maximum(0, W1 @ state + b1)
            h1_out = h1_out / (np.linalg.norm(h1_out) + 1e-8)
            h2_out = np.maximum(0, W2 @ h1_out + b2)
            h2_out = h2_out / (np.linalg.norm(h2_out) + 1e-8)
            h3_out = np.maximum(0, W3 @ h2_out + b3)
            h3_out = h3_out / (np.linalg.norm(h3_out) + 1e-8)
            action = np.tanh(W4 @ h3_out + b4)

            # Store for metrics
            hidden_states.append(h3_out.copy())
            action_history.append(action.copy())

            # Prediction error (state prediction as proxy)
            if t > 0:
                pred_error = np.linalg.norm(state - prev_state * 0.85)
                prediction_errors.append(pred_error)

            # Φ proxy (layer correlation)
            if t > 5:
                phi_t = np.mean([abs(np.corrcoef(hidden_states[-1], hidden_states[-2])[0,1]
                                    if len(hidden_states) > 1 else 0)])
                phi_history.append(phi_t if not np.isnan(phi_t) else 0.5)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))

            prev_state = state.copy()
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        # Compute all harmonies
        metrics = {
            'h1': compute_h1_resonant_coherence(hidden_states),
            'h2': compute_h2_diversity([0, 1, 0, 1]),  # Binary agent types
            'h3': compute_h3_prediction_accuracy(prediction_errors),
            'h4': compute_h4_behavioral_entropy(action_history),
            'h5': compute_h5_mutual_te(hidden_states),
            'h6': compute_h6_flow_symmetry(
                np.array([np.linalg.norm(a) for a in action_history[:40]]),
                np.array([np.linalg.norm(a) for a in action_history[40:]])
            ),
            'h7': compute_h7_phi_growth(phi_history) if phi_history else 0.0,
        }

        # Full K-Index
        k_full = compute_full_k_index(metrics)

        # Simple K-Index (correlation)
        try:
            r, _ = pearsonr(obs_norms, act_norms)
            k_simple = 2.0 * abs(r) if not np.isnan(r) else 0.0
        except:
            k_simple = 0.0

        if k_full > 0:
            k_values.append(k_full)
        if k_simple > 0:
            simple_k_values.append(k_simple)

    return (
        np.mean(k_values) if k_values else 0.0,
        np.mean(simple_k_values) if simple_k_values else 0.0
    )


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Full 7-Harmony K-Index vs Simplified                      ║")
    print("║                                                                ║")
    print("║  Comparing correlation-only vs full 7-harmony computation     ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)

    GENS = 40

    # CMA-ES optimizing for full K-Index
    cmaes = CMAES(dim=332, population_size=30, sigma=0.5)

    history = []
    best_full = 0.0
    best_simple = 0.0

    print("Gen │  Full K  │ Simple K │ Ratio")
    print("────┼──────────┼──────────┼──────")

    for gen in range(1, GENS + 1):
        pop = cmaes.ask()

        # Evaluate with full K-Index
        results = [evaluate_full_7harmony(p) for p in pop]
        full_k = [r[0] for r in results]
        simple_k = [r[1] for r in results]

        # Tell CMA-ES to optimize full K
        cmaes.tell(pop, full_k)

        gen_best_full = max(full_k)
        gen_best_simple = max(simple_k)

        if gen_best_full > best_full:
            best_full = gen_best_full
        if gen_best_simple > best_simple:
            best_simple = gen_best_simple

        history.append({
            'gen': gen,
            'full_k': float(gen_best_full),
            'simple_k': float(gen_best_simple)
        })

        if gen % 5 == 0 or gen == 1:
            ratio = gen_best_full / (gen_best_simple + 1e-10)
            print(f"{gen:3d} │ {gen_best_full:8.4f} │ {gen_best_simple:8.4f} │ {ratio:.3f}")

    print("\n" + "═" * 50)

    # Correlation between full and simple K
    full_values = [h['full_k'] for h in history]
    simple_values = [h['simple_k'] for h in history]
    corr, _ = pearsonr(full_values, simple_values)

    print(f"\n📊 Results:")
    print(f"  Best Full K-Index: {best_full:.4f}")
    print(f"  Best Simple K-Index: {best_simple:.4f}")
    print(f"  Correlation (full vs simple): {corr:.4f}")

    if corr > 0.8:
        print(f"\n  ✅ HIGH correlation - simple K is good proxy")
    elif corr > 0.5:
        print(f"\n  🔶 MODERATE correlation - simple K captures some signal")
    else:
        print(f"\n  ❌ LOW correlation - full K measures different things")

    # Save
    Path('logs/7harmony_comparison').mkdir(parents=True, exist_ok=True)
    with open('logs/7harmony_comparison/results.json', 'w') as f:
        json.dump({
            'history': history,
            'best_full': float(best_full),
            'best_simple': float(best_simple),
            'correlation': float(corr),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/7harmony_comparison/")
    print('\n"The full picture reveals what correlation alone cannot." 💚\n')


if __name__ == '__main__':
    main()
