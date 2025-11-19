#!/usr/bin/env python3
"""
🧠 Full 7-Harmony K-Index Optimization

Now that we know simple K ≠ coherence, let's optimize for the REAL thing.
Then test if full K predicts task performance better than simple K.

Hypothesis: Full K-Index will correlate better with RL performance.
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


# ============== 7-Harmony Computations ==============

def compute_h1_integration(hidden_states):
    """H1: Integrated information proxy (cross-dimensional correlation)."""
    if len(hidden_states) < 10:
        return 0.0

    states = np.array(hidden_states)
    n_dims = states.shape[1]
    if n_dims < 2:
        return 0.0

    # Average absolute correlation across all dimension pairs
    correlations = []
    for i in range(min(n_dims, 6)):  # Limit for speed
        for j in range(i+1, min(n_dims, 6)):
            try:
                r = np.corrcoef(states[:, i], states[:, j])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))
            except:
                pass

    return np.mean(correlations) if correlations else 0.0


def compute_h2_diversity(action_history):
    """H2: Diversity of action types (discretized)."""
    if len(action_history) < 5:
        return 0.5

    actions = np.array(action_history)
    # Discretize actions into bins
    n_bins = 5
    discretized = []
    for a in actions:
        bins = tuple(int(min(n_bins-1, max(0, (x + 1) / 2 * n_bins))) for x in a)
        discretized.append(bins)

    type_counts = Counter(discretized)
    n_types = len(type_counts)
    if n_types <= 1:
        return 0.0

    total = len(discretized)
    probs = np.array([count / total for count in type_counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    max_h = np.log(n_types)

    return h / max_h if max_h > 0 else 0.0


def compute_h3_prediction(states, actions):
    """H3: How well actions predict next state (inverse error)."""
    if len(states) < 10:
        return 0.5

    errors = []
    for t in range(1, len(states)):
        # Simple prediction: next state ≈ 0.85 * current + action effect
        predicted = 0.85 * states[t-1] + 0.05 * np.mean(actions[t-1])
        error = np.linalg.norm(states[t] - predicted)
        errors.append(error)

    mean_error = np.mean(errors)
    # Normalize: 0 error → 1.0, high error → 0.0
    return max(0.0, 1.0 - mean_error / 2.0)


def compute_h4_behavioral_entropy(action_history):
    """H4: Entropy of action distribution (exploration)."""
    if len(action_history) < 10:
        return 0.5

    actions = np.array(action_history)
    n_bins = min(10, len(actions) // 5)
    if n_bins < 2:
        return 0.5

    entropies = []
    for dim in range(actions.shape[1]):
        hist, _ = np.histogram(actions[:, dim], bins=n_bins)
        hist = hist + 1e-10
        probs = hist / hist.sum()
        entropies.append(-np.sum(probs * np.log(probs)))

    max_entropy = np.log(n_bins)
    return np.mean(entropies) / max_entropy if max_entropy > 0 else 0.0


def compute_h5_temporal_coherence(hidden_states):
    """H5: Temporal consistency (autocorrelation)."""
    if len(hidden_states) < 10:
        return 0.5

    states = np.array(hidden_states)
    # Lag-1 autocorrelation per dimension
    autocorrs = []
    for dim in range(min(states.shape[1], 6)):
        try:
            r = np.corrcoef(states[:-1, dim], states[1:, dim])[0, 1]
            if not np.isnan(r):
                autocorrs.append(abs(r))
        except:
            pass

    return np.mean(autocorrs) if autocorrs else 0.5


def compute_h6_balance(action_history):
    """H6: Balance/symmetry of actions (not one-sided)."""
    if len(action_history) < 10:
        return 0.5

    actions = np.array(action_history)
    # Check if actions are balanced around zero
    mean_action = np.mean(actions, axis=0)
    # Perfect balance: mean = 0
    imbalance = np.linalg.norm(mean_action)
    # Normalize: 0 imbalance → 1.0, high imbalance → 0.0
    return max(0.0, 1.0 - imbalance)


def compute_h7_growth(metric_history):
    """H7: Improvement over time (positive slope)."""
    if len(metric_history) < 10:
        return 0.5

    recent = metric_history[-50:] if len(metric_history) > 50 else metric_history
    times = np.arange(len(recent))
    values = np.array(recent)

    if len(times) > 1 and np.std(values) > 1e-10:
        slope = np.polyfit(times, values, deg=1)[0]
        # Normalize slope
        return 0.5 + 0.5 * np.tanh(slope * 10)

    return 0.5


def compute_full_k(metrics):
    """Compute full 7-Harmony K-Index with equal weights."""
    harmonies = np.array([
        metrics['h1'],  # Integration
        metrics['h2'],  # Diversity
        metrics['h3'],  # Prediction
        metrics['h4'],  # Behavioral entropy
        metrics['h5'],  # Temporal coherence
        metrics['h6'],  # Balance
        metrics['h7'],  # Growth
    ])

    # Scale to [0, 2] range like simple K
    return 2.0 * np.mean(harmonies)


# ============== Network Evaluation ==============

def evaluate_full_k(params):
    """Evaluate 4-layer network with full 7-Harmony K-Index."""
    obs_dim, h1_size, h2_size, h3_size, act_dim = 8, 12, 10, 6, 4
    total = 332

    if len(params) < total:
        params = np.concatenate([params, np.zeros(total - len(params))])

    # Extract weights
    offset = 0
    W1 = params[offset:offset + h1_size*obs_dim].reshape(h1_size, obs_dim)
    offset += h1_size * obs_dim
    b1 = params[offset:offset + h1_size]
    offset += h1_size
    W2 = params[offset:offset + h2_size*h1_size].reshape(h2_size, h1_size)
    offset += h2_size * h1_size
    b2 = params[offset:offset + h2_size]
    offset += h2_size
    W3 = params[offset:offset + h3_size*h2_size].reshape(h3_size, h2_size)
    offset += h3_size * h2_size
    b3 = params[offset:offset + h3_size]
    offset += h3_size
    W4 = params[offset:offset + act_dim*h3_size].reshape(act_dim, h3_size)
    offset += act_dim * h3_size
    b4 = params[offset:offset + act_dim]

    k_values = []
    simple_k_values = []

    for _ in range(4):
        # Trajectory storage
        hidden_states = []
        action_history = []
        state_history = []
        h1_history = []  # For growth tracking

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

            # Store
            hidden_states.append(h3_out.copy())
            action_history.append(action.copy())
            state_history.append(state.copy())

            # Track H1 for growth
            if t > 5:
                h1_t = compute_h1_integration(hidden_states[-10:])
                h1_history.append(h1_t)

            obs_norms.append(np.linalg.norm(state))
            act_norms.append(np.linalg.norm(action))

            # Environment dynamics
            state = 0.85 * state + 0.1 * np.random.randn(obs_dim) + 0.05 * np.mean(action) * np.ones(obs_dim)

        # Compute all harmonies
        metrics = {
            'h1': compute_h1_integration(hidden_states),
            'h2': compute_h2_diversity(action_history),
            'h3': compute_h3_prediction(state_history, action_history),
            'h4': compute_h4_behavioral_entropy(action_history),
            'h5': compute_h5_temporal_coherence(hidden_states),
            'h6': compute_h6_balance(action_history),
            'h7': compute_h7_growth(h1_history) if h1_history else 0.5,
        }

        # Full K
        k_full = compute_full_k(metrics)

        # Simple K for comparison
        try:
            r, _ = pearsonr(obs_norms, act_norms)
            k_simple = 2.0 * abs(r) if not np.isnan(r) else 0.0
        except:
            k_simple = 0.0

        k_values.append(k_full)
        simple_k_values.append(k_simple)

    return (
        np.mean(k_values),
        np.mean(simple_k_values)
    )


# ============== CartPole Environment ==============

class CartPoleEnv:
    """Simplified CartPole."""
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = bool(x < -self.x_threshold or x > self.x_threshold or
                    theta < -self.theta_threshold or theta > self.theta_threshold)
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done


class SimplePolicy:
    """Policy network for CartPole."""
    def __init__(self):
        self.W1 = np.random.randn(12, 4) * 0.5
        self.b1 = np.zeros(12)
        self.W2 = np.random.randn(10, 12) * 0.5
        self.b2 = np.zeros(10)
        self.W3 = np.random.randn(6, 10) * 0.5
        self.b3 = np.zeros(6)
        self.W4 = np.random.randn(2, 6) * 0.5
        self.b4 = np.zeros(2)

    def forward(self, obs):
        h1 = np.maximum(0, self.W1 @ obs + self.b1)
        h1 = h1 / (np.linalg.norm(h1) + 1e-8)
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)
        h2 = h2 / (np.linalg.norm(h2) + 1e-8)
        h3 = np.maximum(0, self.W3 @ h2 + self.b3)
        h3 = h3 / (np.linalg.norm(h3) + 1e-8)
        return self.W4 @ h3 + self.b4

    def act(self, obs):
        return np.argmax(self.forward(obs))

    def get_params(self):
        return np.concatenate([
            self.W1.flatten(), self.b1, self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3, self.W4.flatten(), self.b4
        ])

    def set_params(self, params):
        offset = 0
        self.W1 = params[offset:offset+48].reshape(12, 4); offset += 48
        self.b1 = params[offset:offset+12]; offset += 12
        self.W2 = params[offset:offset+120].reshape(10, 12); offset += 120
        self.b2 = params[offset:offset+10]; offset += 10
        self.W3 = params[offset:offset+60].reshape(6, 10); offset += 60
        self.b3 = params[offset:offset+6]; offset += 6
        self.W4 = params[offset:offset+12].reshape(2, 6); offset += 12
        self.b4 = params[offset:offset+2]


def evaluate_policy(policy, env, n_episodes=5):
    """Evaluate policy performance."""
    total_reward = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = policy.act(obs)
            obs, reward, done = env.step(action)
            total_reward += reward
            steps += 1
    return total_reward / n_episodes


def compute_policy_k_indices(policy, env):
    """Compute both K indices for a policy on CartPole."""
    hidden_states = []
    action_history = []
    state_history = []
    obs_norms, act_norms = [], []

    for _ in range(3):  # 3 episodes
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            logits = policy.forward(obs)
            action = np.argmax(logits)

            # Store for metrics
            obs_norms.append(np.linalg.norm(obs))
            act_norms.append(np.linalg.norm(logits))
            action_history.append(logits.copy())
            state_history.append(obs.copy())

            obs, _, done = env.step(action)
            steps += 1

    # Simple K
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        simple_k = 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        simple_k = 0.0

    # Full K (simplified for CartPole)
    metrics = {
        'h1': 0.5,  # No hidden states tracked
        'h2': compute_h2_diversity(action_history),
        'h3': 0.5,  # Prediction not applicable
        'h4': compute_h4_behavioral_entropy(action_history),
        'h5': 0.5,  # No hidden states
        'h6': compute_h6_balance(action_history),
        'h7': 0.5,  # No growth tracking
    }
    full_k = compute_full_k(metrics)

    return full_k, simple_k


# ============== Main Experiment ==============

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 Full 7-Harmony K-Index Optimization                       ║")
    print("║                                                                ║")
    print("║  Optimize for REAL coherence, then test vs performance        ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)

    # Phase 1: Optimize for Full K-Index
    print("Phase 1: Optimizing for Full 7-Harmony K-Index...\n")

    cmaes = CMAES(dim=332, population_size=30, sigma=0.5)

    best_full = 0.0
    best_simple = 0.0

    print("Gen │  Full K  │ Simple K")
    print("────┼──────────┼──────────")

    for gen in range(1, 41):
        pop = cmaes.ask()
        results = [evaluate_full_k(p) for p in pop]
        full_k = [r[0] for r in results]
        simple_k = [r[1] for r in results]

        # Optimize for FULL K
        cmaes.tell(pop, full_k)

        gen_best_full = max(full_k)
        gen_best_simple = max(simple_k)

        if gen_best_full > best_full:
            best_full = gen_best_full
        if gen_best_simple > best_simple:
            best_simple = gen_best_simple

        if gen % 10 == 0 or gen == 1:
            print(f"{gen:3d} │ {gen_best_full:8.4f} │ {gen_best_simple:8.4f}")

    print(f"\n✅ Full K optimized: {best_full:.4f}")
    print(f"   (Simple K during optimization: {best_simple:.4f})")

    # Phase 2: Test on CartPole
    print("\n" + "═" * 55)
    print("\nPhase 2: Testing K-Index vs CartPole Performance...\n")

    env = CartPoleEnv()
    policy = SimplePolicy()
    n_params = len(policy.get_params())

    cmaes_rl = CMAES(dim=n_params, population_size=20, sigma=0.5)

    history = []

    print("Gen │ Performance │  Full K  │ Simple K")
    print("────┼─────────────┼──────────┼──────────")

    for gen in range(1, 31):
        pop = cmaes_rl.ask()

        fitness = []
        full_k_vals = []
        simple_k_vals = []

        for params in pop:
            policy.set_params(params)
            perf = evaluate_policy(policy, env, n_episodes=3)
            full_k, simple_k = compute_policy_k_indices(policy, env)

            fitness.append(perf)
            full_k_vals.append(full_k)
            simple_k_vals.append(simple_k)

        best_fitness = cmaes_rl.tell(pop, fitness)
        best_idx = np.argmax(fitness)

        history.append({
            'gen': gen,
            'performance': float(best_fitness),
            'full_k': float(full_k_vals[best_idx]),
            'simple_k': float(simple_k_vals[best_idx])
        })

        if gen % 5 == 0 or gen == 1:
            print(f"{gen:3d} │ {best_fitness:11.1f} │ {full_k_vals[best_idx]:8.4f} │ {simple_k_vals[best_idx]:8.4f}")

    # Analyze correlations
    performances = [h['performance'] for h in history]
    full_k_history = [h['full_k'] for h in history]
    simple_k_history = [h['simple_k'] for h in history]

    corr_full, p_full = pearsonr(performances, full_k_history)
    corr_simple, p_simple = pearsonr(performances, simple_k_history)

    print("\n" + "═" * 55)
    print("\n📊 Performance Correlation Analysis:")
    print(f"  Full K vs Performance:   r = {corr_full:+.4f} (p = {p_full:.4f})")
    print(f"  Simple K vs Performance: r = {corr_simple:+.4f} (p = {p_simple:.4f})")

    # Determine winner
    if abs(corr_full) > abs(corr_simple) + 0.1:
        print(f"\n  ✅ FULL K-Index is BETTER predictor!")
        winner = "FULL"
    elif abs(corr_simple) > abs(corr_full) + 0.1:
        print(f"\n  ⚠️  Simple K-Index is better predictor")
        winner = "SIMPLE"
    else:
        print(f"\n  🔶 Neither is a strong predictor")
        winner = "NEITHER"

    print(f"\n📈 Final Results:")
    print(f"  Best Performance: {max(performances):.1f} steps")
    print(f"  Final Full K: {full_k_history[-1]:.4f}")
    print(f"  Final Simple K: {simple_k_history[-1]:.4f}")

    # Save
    Path('logs/full_k_optimization').mkdir(parents=True, exist_ok=True)
    with open('logs/full_k_optimization/results.json', 'w') as f:
        json.dump({
            'phase1_best_full': float(best_full),
            'phase1_best_simple': float(best_simple),
            'history': history,
            'corr_full': float(corr_full),
            'corr_simple': float(corr_simple),
            'winner': winner,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/full_k_optimization/")
    print('\n"True coherence encompasses more than correlation." 💚\n')


if __name__ == '__main__':
    main()
