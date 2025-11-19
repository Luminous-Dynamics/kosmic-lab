#!/usr/bin/env python3
"""
🧠 K-Index Real RL Validation

Critical test: Does high K-Index correlate with actual task performance?

Hypothesis: K-Index should increase as agent performance improves during training.
If K doesn't correlate with performance, it's just a mathematical artifact.

Uses CartPole-v1 (simple, fast to train, clear success metric).
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

class SimplePolicy:
    """Simple policy network matching our K-Index architecture."""

    def __init__(self, obs_dim=4, h1=12, h2=10, h3=6, act_dim=2):
        """4-layer architecture: obs→h1→h2→h3→act"""
        self.obs_dim = obs_dim
        self.h1, self.h2, self.h3 = h1, h2, h3
        self.act_dim = act_dim

        # Initialize weights (same structure as K-Index experiments)
        self.W1 = np.random.randn(h1, obs_dim) * 0.5
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h2, h1) * 0.5
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h3, h2) * 0.5
        self.b3 = np.zeros(h3)
        self.W4 = np.random.randn(act_dim, h3) * 0.5
        self.b4 = np.zeros(act_dim)

    def forward(self, obs):
        """Forward pass with layer normalization."""
        h1 = np.maximum(0, self.W1 @ obs + self.b1)
        h1 = h1 / (np.linalg.norm(h1) + 1e-8)

        h2 = np.maximum(0, self.W2 @ h1 + self.b2)
        h2 = h2 / (np.linalg.norm(h2) + 1e-8)

        h3 = np.maximum(0, self.W3 @ h2 + self.b3)
        h3 = h3 / (np.linalg.norm(h3) + 1e-8)

        logits = self.W4 @ h3 + self.b4
        return logits

    def act(self, obs):
        """Select action (argmax for simplicity)."""
        logits = self.forward(obs)
        return np.argmax(logits)

    def get_params(self):
        """Flatten all parameters."""
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3,
            self.W4.flatten(), self.b4
        ])

    def set_params(self, params):
        """Set parameters from flattened array."""
        offset = 0

        size = self.h1 * self.obs_dim
        self.W1 = params[offset:offset+size].reshape(self.h1, self.obs_dim)
        offset += size

        self.b1 = params[offset:offset+self.h1]
        offset += self.h1

        size = self.h2 * self.h1
        self.W2 = params[offset:offset+size].reshape(self.h2, self.h1)
        offset += size

        self.b2 = params[offset:offset+self.h2]
        offset += self.h2

        size = self.h3 * self.h2
        self.W3 = params[offset:offset+size].reshape(self.h3, self.h2)
        offset += size

        self.b3 = params[offset:offset+self.h3]
        offset += self.h3

        size = self.act_dim * self.h3
        self.W4 = params[offset:offset+size].reshape(self.act_dim, self.h3)
        offset += size

        self.b4 = params[offset:offset+self.act_dim]


class CartPoleEnv:
    """Simplified CartPole environment (no gym dependency)."""

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
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold or theta > self.theta_threshold
        )

        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done


def compute_k_index(policy, env, n_episodes=5):
    """Compute K-Index for a policy on actual environment."""
    all_obs_norms = []
    all_act_norms = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            logits = policy.forward(obs)
            action = np.argmax(logits)

            all_obs_norms.append(np.linalg.norm(obs))
            all_act_norms.append(np.linalg.norm(logits))

            obs, _, done = env.step(action)
            steps += 1

    if len(all_obs_norms) < 2:
        return 0.0

    try:
        r, _ = pearsonr(all_obs_norms, all_act_norms)
        return 0.0 if np.isnan(r) else 2.0 * abs(r)
    except:
        return 0.0


def evaluate_policy(policy, env, n_episodes=10):
    """Evaluate policy performance (average episode length)."""
    total_reward = 0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 500:
            action = policy.act(obs)
            obs, reward, done = env.step(action)
            episode_reward += reward
            steps += 1

        total_reward += episode_reward

    return total_reward / n_episodes


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


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🧠 K-Index Real RL Validation                                ║")
    print("║                                                                ║")
    print("║  Hypothesis: K-Index correlates with task performance        ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)

    # Setup
    env = CartPoleEnv()
    policy = SimplePolicy(obs_dim=4, h1=12, h2=10, h3=6, act_dim=2)

    # Calculate parameter count
    n_params = len(policy.get_params())
    print(f"Policy architecture: 4→12→10→6→2 ({n_params} params)\n")

    # CMA-ES optimizer
    cmaes = CMAES(dim=n_params, population_size=20, sigma=0.5)

    # Training loop
    GENERATIONS = 30
    history = []

    print("Gen │ Performance │  K-Index  │ Correlation")
    print("────┼─────────────┼───────────┼────────────")

    for gen in range(1, GENERATIONS + 1):
        # Generate population
        population = cmaes.ask()

        # Evaluate each candidate
        fitness = []
        for params in population:
            policy.set_params(params)
            perf = evaluate_policy(policy, env, n_episodes=3)
            fitness.append(perf)

        # Update CMA-ES
        best_fitness = cmaes.tell(population, fitness)

        # Get best policy and compute K-Index
        best_idx = np.argmax(fitness)
        policy.set_params(population[best_idx])
        k_index = compute_k_index(policy, env, n_episodes=5)

        history.append({
            'gen': gen,
            'performance': float(best_fitness),
            'k_index': float(k_index)
        })

        # Print progress
        if gen % 5 == 0 or gen == 1:
            print(f"{gen:3d} │ {best_fitness:11.1f} │ {k_index:9.4f} │")

    # Analyze correlation between K-Index and Performance
    performances = [h['performance'] for h in history]
    k_indices = [h['k_index'] for h in history]

    correlation, p_value = pearsonr(performances, k_indices)

    print("\n" + "═" * 55)
    print("\n📊 Correlation Analysis:")
    print(f"  K-Index vs Performance: r = {correlation:.4f} (p = {p_value:.4f})")

    if abs(correlation) > 0.5 and p_value < 0.05:
        print(f"\n  ✅ SIGNIFICANT CORRELATION FOUND!")
        print(f"  K-Index {'DOES' if correlation > 0 else 'inversely'} correlate with performance")
        verdict = "VALIDATED"
    elif abs(correlation) > 0.3:
        print(f"\n  🔶 WEAK CORRELATION")
        print(f"  K-Index shows weak relationship with performance")
        verdict = "WEAK"
    else:
        print(f"\n  ❌ NO SIGNIFICANT CORRELATION")
        print(f"  K-Index does NOT predict task performance")
        verdict = "NOT_VALIDATED"

    # Final stats
    print(f"\n📈 Final Results:")
    print(f"  Best Performance: {max(performances):.1f} steps")
    print(f"  Final K-Index: {k_indices[-1]:.4f}")
    print(f"  K-Index range: [{min(k_indices):.4f}, {max(k_indices):.4f}]")

    # Save results
    Path('logs/k_index_validation').mkdir(parents=True, exist_ok=True)
    with open('logs/k_index_validation/cartpole_results.json', 'w') as f:
        json.dump({
            'history': history,
            'correlation': float(correlation),
            'p_value': float(p_value),
            'verdict': verdict,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/k_index_validation/")
    print('\n"Truth emerges from honest testing." 💚\n')

    return correlation, verdict


if __name__ == '__main__':
    main()
