#!/usr/bin/env python3
"""
🧠 Full 7-Harmony K-Index vs Task Performance Validation

Implements the COMPLETE 7-Harmony K-Index from the mathematical formalism
and tests if it correlates with actual CartPole task performance.

The 7 Harmonies:
H1: Resonant Coherence (Φ-based integration)
H2: Pan-Sentient Flourishing (Diversity)
H3: Integral Wisdom (Prediction Accuracy)
H4: Infinite Play (Behavioral Entropy)
H5: Universal Interconnectedness (Mutual Transfer Entropy)
H6: Sacred Reciprocity (Flow Symmetry)
H7: Evolutionary Progression (Φ Growth Rate)

Question: Does Full K predict task performance better than Simple K?
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from collections import Counter


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
        return [self.mean + self.sigma * (L @ np.random.randn(self.dim))
                for _ in range(self.pop_size)]

    def tell(self, population, fitness):
        indices = np.argsort(fitness)[::-1]
        elite_size = max(3, self.pop_size // 4)
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


class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = 1.1
        self.length = 0.5
        self.polemass_length = 0.05
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 0.2095
        self.x_threshold = 2.4
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4/3 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        self.state = np.array([
            x + self.tau * x_dot,
            x_dot + self.tau * xacc,
            theta + self.tau * theta_dot,
            theta_dot + self.tau * thetaacc
        ])
        done = abs(self.state[0]) > self.x_threshold or abs(self.state[2]) > self.theta_threshold
        return self.state.copy(), 0 if done else 1, done


class Policy:
    def __init__(self):
        # 4→16→8→2 network with hidden state tracking
        self.W1 = np.random.randn(16, 4) * 0.5
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(8, 16) * 0.5
        self.b2 = np.zeros(8)
        self.W3 = np.random.randn(2, 8) * 0.5
        self.b3 = np.zeros(2)

    def forward(self, obs):
        h1 = np.maximum(0, self.W1 @ obs + self.b1)
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)
        logits = self.W3 @ h2 + self.b3
        return logits, h1, h2

    def act(self, obs):
        logits, _, _ = self.forward(obs)
        return np.argmax(logits)

    def get_params(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_params(self, params):
        idx = 0
        self.W1 = params[idx:idx+64].reshape(16, 4); idx += 64
        self.b1 = params[idx:idx+16]; idx += 16
        self.W2 = params[idx:idx+128].reshape(8, 16); idx += 128
        self.b2 = params[idx:idx+8]; idx += 8
        self.W3 = params[idx:idx+16].reshape(2, 8); idx += 16
        self.b3 = params[idx:idx+2]


# =============================================================================
# FULL 7-HARMONY IMPLEMENTATION
# =============================================================================

def compute_h1_resonant_coherence(hidden_states, phi_baseline=1.0):
    """H1: Normalized integrated information (Φ approximation).

    Uses average correlation across hidden dimensions as Φ proxy.
    High H1 = system cannot be decomposed into independent parts.
    """
    if len(hidden_states) < 5:
        return 0.5

    states = np.array(hidden_states)
    n_dims = states.shape[1]
    if n_dims < 2:
        return 0.5

    # Compute average absolute correlation between all dimension pairs
    correlations = []
    for i in range(min(n_dims, 8)):  # Limit to first 8 dims for speed
        for j in range(i+1, min(n_dims, 8)):
            if np.std(states[:, i]) > 1e-10 and np.std(states[:, j]) > 1e-10:
                r = np.corrcoef(states[:, i], states[:, j])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))

    if not correlations:
        return 0.5

    phi = np.mean(correlations)
    # Normalize: phi=0 → H1=0, phi=phi_baseline → H1=1
    h1 = phi / phi_baseline
    return float(np.clip(h1, 0, 2))


def compute_h2_diversity(action_history):
    """H2: Shannon entropy of action distribution.

    High H2 = diverse actions (not stuck on one).
    """
    if len(action_history) < 5:
        return 0.5

    type_counts = Counter(action_history)
    n_types = len(type_counts)
    if n_types <= 1:
        return 0.0  # No diversity

    total = len(action_history)
    probs = np.array([c / total for c in type_counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    max_h = np.log(n_types)

    return float(h / max_h) if max_h > 0 else 0.5


def compute_h3_prediction_accuracy(observations, predictions, baseline_fe=1.0):
    """H3: Inverse of mean prediction error.

    High H3 = accurate predictions of next state.
    """
    if len(observations) < 5 or len(predictions) < 5:
        return 0.5

    # Compare predictions to actual next observations
    min_len = min(len(observations) - 1, len(predictions))
    if min_len < 3:
        return 0.5

    errors = []
    for i in range(min_len):
        error = np.linalg.norm(observations[i+1] - predictions[i])
        errors.append(error)

    mean_error = np.mean(errors)
    h3 = max(0.0, 1.0 - (mean_error / baseline_fe))
    return float(np.clip(h3, 0, 1))


def compute_h4_behavioral_entropy(action_logits, n_bins=5):
    """H4: Shannon entropy of discretized action distribution.

    High H4 = diverse, exploratory actions.
    """
    if len(action_logits) < 10:
        return 0.5

    all_actions = np.array(action_logits)
    if all_actions.ndim == 1:
        all_actions = all_actions.reshape(-1, 1)

    n_bins = min(5, len(all_actions) // 5)
    if n_bins < 2:
        return 0.5

    entropies = []
    for dim in range(all_actions.shape[1]):
        col = all_actions[:, dim]
        if np.std(col) < 1e-10:
            continue
        hist, _ = np.histogram(col, bins=n_bins)
        hist = hist + 1e-10
        probs = hist / hist.sum()
        entropies.append(-np.sum(probs * np.log(probs)))

    if not entropies:
        return 0.5

    max_entropy = np.log(n_bins)
    return float(np.mean(entropies) / max_entropy) if max_entropy > 0 else 0.5


def compute_h5_mutual_te(hidden_states):
    """H5: Proportion of bidirectional information flow.

    Uses time-lagged correlation as transfer entropy proxy.
    High H5 = genuine mutual influence between dimensions.
    """
    if len(hidden_states) < 10:
        return 0.5

    series = np.array(hidden_states)
    n_dims = series.shape[1]
    if n_dims < 2:
        return 0.5

    # Compute time-lagged correlations as TE proxy
    mutual_influences = []
    for i in range(min(n_dims, 6)):
        for j in range(i+1, min(n_dims, 6)):
            if np.std(series[:-1, i]) > 1e-10 and np.std(series[1:, j]) > 1e-10:
                corr_ij = abs(np.corrcoef(series[:-1, i], series[1:, j])[0, 1])
                corr_ji = abs(np.corrcoef(series[:-1, j], series[1:, i])[0, 1])
                if not (np.isnan(corr_ij) or np.isnan(corr_ji)):
                    mutual = min(corr_ij, corr_ji)
                    total = corr_ij + corr_ji + 1e-10
                    mutual_influences.append(2 * mutual / total)

    return float(np.mean(mutual_influences)) if mutual_influences else 0.5


def compute_h6_flow_symmetry(outgoing, incoming):
    """H6: 1 - Jensen-Shannon divergence of flows.

    High H6 = balanced giving/receiving (reciprocal exchange).
    """
    if len(outgoing) < 3 or len(incoming) < 3:
        return 0.5

    # Ensure same length
    min_len = min(len(outgoing), len(incoming))
    outgoing = outgoing[:min_len]
    incoming = incoming[:min_len]

    # Normalize to probability distributions
    out_norm = np.abs(outgoing) + 1e-10
    in_norm = np.abs(incoming) + 1e-10
    out_prob = out_norm / out_norm.sum()
    in_prob = in_norm / in_norm.sum()

    # JS divergence
    m = 0.5 * (out_prob + in_prob)
    kl_out = np.sum(out_prob * np.log(out_prob / (m + 1e-10) + 1e-10))
    kl_in = np.sum(in_prob * np.log(in_prob / (m + 1e-10) + 1e-10))
    js_div = 0.5 * (kl_out + kl_in)

    return float(1.0 - np.sqrt(np.clip(js_div, 0, 1)))


def compute_h7_phi_growth(phi_history):
    """H7: Rate of Φ increase over time.

    Positive H7 = system becoming more integrated.
    """
    if len(phi_history) < 10:
        return 0.0

    recent = phi_history[-50:] if len(phi_history) > 50 else phi_history
    times = np.arange(len(recent))
    phis = np.array(recent)

    if np.std(phis) < 1e-10:
        return 0.0

    # Linear regression slope
    times_norm = times / (times.max() + 1e-10)
    slope = np.polyfit(times_norm, phis, deg=1)[0]
    std_phi = np.std(phis) + 1e-10

    return float(np.tanh(slope / std_phi))


def compute_full_k_index(h1, h2, h3, h4, h5, h6, h7):
    """Aggregate all 7 harmonies into Full K-Index.

    K = Σ w_i × H_i with equal weights.
    """
    harmonies = np.array([h1, h2, h3, h4, h5, h6, h7])
    weights = np.ones(7) / 7.0
    return float(np.dot(weights, harmonies))


def compute_simple_k_index(obs_norms, act_norms):
    """Simple K-Index: just correlation."""
    if len(obs_norms) < 3 or len(act_norms) < 3:
        return 0.0
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_policy(policy, env, n_episodes=5):
    """Evaluate actual task performance."""
    total = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(500):
            obs, r, done = env.step(policy.act(obs))
            total += r
            if done:
                break
    return total / n_episodes


def compute_all_k_indices(policy, env):
    """Compute both Full 7-Harmony K and Simple K."""

    # Trajectory storage
    observations = []
    actions = []
    action_logits = []
    hidden_states_h1 = []  # First hidden layer
    hidden_states_h2 = []  # Second hidden layer
    obs_norms = []
    act_norms = []
    phi_history = []

    for _ in range(3):  # 3 episodes
        obs = env.reset()
        predictions = []

        for t in range(150):
            logits, h1, h2 = policy.forward(obs)
            action = np.argmax(logits)

            # Store data
            observations.append(obs.copy())
            actions.append(action)
            action_logits.append(logits.copy())
            hidden_states_h1.append(h1.copy())
            hidden_states_h2.append(h2.copy())
            obs_norms.append(np.linalg.norm(obs))
            act_norms.append(np.linalg.norm(logits))

            # Simple prediction: next state ≈ current state
            predictions.append(obs.copy())

            # Phi proxy for H7
            if len(hidden_states_h2) > 5:
                recent = np.array(hidden_states_h2[-5:])
                if recent.shape[1] >= 2:
                    phi_t = abs(np.corrcoef(recent[:, 0], recent[:, 1])[0, 1])
                    if not np.isnan(phi_t):
                        phi_history.append(phi_t)

            obs, _, done = env.step(action)
            if done:
                break

    # Compute all 7 harmonies
    h1 = compute_h1_resonant_coherence(hidden_states_h2)
    h2 = compute_h2_diversity(actions)
    h3 = compute_h3_prediction_accuracy(observations, predictions)
    h4 = compute_h4_behavioral_entropy(action_logits)
    h5 = compute_h5_mutual_te(hidden_states_h2)

    # H6: Flow symmetry between first and second half
    mid = len(obs_norms) // 2
    h6 = compute_h6_flow_symmetry(
        np.array(obs_norms[:mid]),
        np.array(obs_norms[mid:])
    )

    h7 = compute_h7_phi_growth(phi_history)

    # Full K-Index
    full_k = compute_full_k_index(h1, h2, h3, h4, h5, h6, h7)

    # Simple K-Index
    simple_k = compute_simple_k_index(obs_norms, act_norms)

    return {
        'full_k': full_k,
        'simple_k': simple_k,
        'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5, 'h6': h6, 'h7': h7
    }


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Full 7-Harmony K-Index vs Task Performance                ║")
    print("║                                                                ║")
    print("║  Testing if Full K predicts performance better than Simple K  ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)
    env = CartPoleEnv()
    policy = Policy()

    # Optimize for PERFORMANCE
    cmaes = CMAES(dim=234, population_size=20, sigma=0.5)

    all_data = []

    print("Training policies optimized for TASK PERFORMANCE...\n")
    print("Gen │  Perf  │ Full K │ Simple K │  H1   │  H2   │  H4   │  H7")
    print("────┼────────┼────────┼──────────┼───────┼───────┼───────┼───────")

    for gen in range(1, 31):
        pop = cmaes.ask()
        performances = []
        all_metrics = []

        for params in pop:
            policy.set_params(params)
            perf = evaluate_policy(policy, env)
            metrics = compute_all_k_indices(policy, env)
            metrics['performance'] = perf
            performances.append(perf)
            all_metrics.append(metrics)

        cmaes.tell(pop, performances)

        best_idx = np.argmax(performances)
        best = all_metrics[best_idx]

        all_data.append(best)

        if gen % 5 == 0 or gen == 1:
            print(f"{gen:3d} │ {best['performance']:6.1f} │ {best['full_k']:6.3f} │ "
                  f"{best['simple_k']:8.4f} │ {best['h1']:5.3f} │ {best['h2']:5.3f} │ "
                  f"{best['h4']:5.3f} │ {best['h7']:+5.3f}")

    # Compute correlations
    print("\n" + "═" * 70)
    print("\n📊 Correlation with Task Performance:\n")

    perfs = [d['performance'] for d in all_data]

    metrics_to_test = ['full_k', 'simple_k', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']
    correlations = {}

    for metric in metrics_to_test:
        values = [d[metric] for d in all_data]
        try:
            r, p = pearsonr(perfs, values)
            correlations[metric] = (r, p)
        except:
            correlations[metric] = (0.0, 1.0)

    # Sort by absolute correlation
    sorted_metrics = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)

    print("  Metric      │  r      │ p-value │ Status")
    print("  ────────────┼─────────┼─────────┼────────")

    for metric, (r, p) in sorted_metrics:
        sig = "✅" if p < 0.05 else "  "
        direction = "+" if r > 0 else "-"
        print(f"  {metric:11s} │ {direction}{abs(r):.4f} │ {p:.4f}  │ {sig}")

    # Key comparison
    full_r, full_p = correlations['full_k']
    simple_r, simple_p = correlations['simple_k']

    print("\n" + "═" * 70)
    print("\n🏆 KEY COMPARISON:\n")
    print(f"  Full 7-Harmony K:  r = {full_r:+.4f} (p = {full_p:.4f})")
    print(f"  Simple K-Index:    r = {simple_r:+.4f} (p = {simple_p:.4f})")

    if abs(full_r) > abs(simple_r):
        improvement = abs(full_r) / (abs(simple_r) + 0.001)
        print(f"\n  ✅ Full K is {improvement:.1f}x BETTER predictor!")
        winner = "FULL_K"
    else:
        print(f"\n  ❌ Simple K is still better (or both are poor)")
        winner = "SIMPLE_K"

    if full_r > 0:
        print(f"\n  📈 Full K POSITIVELY correlates with performance!")
        print("     → Higher Full K = Better performance")
        print("     → The 7-Harmony formalism may be valid!")
    else:
        print(f"\n  📉 Full K still NEGATIVELY correlates with performance")
        print("     → The full formalism has the same problem")

    # Save results
    Path('logs/full_7harmony_validation').mkdir(parents=True, exist_ok=True)
    with open('logs/full_7harmony_validation/results.json', 'w') as f:
        json.dump({
            'history': all_data,
            'correlations': {k: {'r': float(v[0]), 'p': float(v[1])}
                           for k, v in correlations.items()},
            'winner': winner,
            'full_k_correlation': float(full_r),
            'simple_k_correlation': float(simple_r),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/full_7harmony_validation/")
    print('\n"Test the full formalism before abandoning the concept." 💚\n')


if __name__ == '__main__':
    main()
