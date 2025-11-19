#!/usr/bin/env python3
"""
🎯 Task-Predictive Metrics Discovery

Following the paradigm shift: K-Index ANTI-correlates with performance.
Now we find what DOES predict task success.

Candidate Metrics:
1. Behavioral Diversity (entropy of actions)
2. Context-Sensitivity (variance in response to similar inputs)
3. Adaptation Rate (how quickly agent changes behavior)
4. Mutual Information (state→action information transfer)

Hypothesis: Low-K "failures" may be high performers.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, entropy
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
        # 4→16→8→2 network
        self.W1 = np.random.randn(16, 4) * 0.5
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(8, 16) * 0.5
        self.b2 = np.zeros(8)
        self.W3 = np.random.randn(2, 8) * 0.5
        self.b3 = np.zeros(2)

    def forward(self, obs):
        h1 = np.maximum(0, self.W1 @ obs + self.b1)
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def act(self, obs):
        return np.argmax(self.forward(obs))

    def get_params(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_params(self, params):
        # 4→16→8→2: W1=64, b1=16, W2=128, b2=8, W3=16, b3=2 = 234 total
        idx = 0
        self.W1 = params[idx:idx+64].reshape(16, 4); idx += 64
        self.b1 = params[idx:idx+16]; idx += 16
        self.W2 = params[idx:idx+128].reshape(8, 16); idx += 128
        self.b2 = params[idx:idx+8]; idx += 8
        self.W3 = params[idx:idx+16].reshape(2, 8); idx += 16
        self.b3 = params[idx:idx+2]

    @staticmethod
    def param_count():
        return 64 + 16 + 128 + 8 + 16 + 2  # = 234


def evaluate_policy(policy, env, n_episodes=5):
    """Evaluate policy performance."""
    total = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(500):
            obs, r, done = env.step(policy.act(obs))
            total += r
            if done:
                break
    return total / n_episodes


def compute_all_metrics(policy, env):
    """Compute all candidate metrics for a policy."""
    obs_norms, act_norms = [], []
    actions, observations = [], []
    logits_history = []

    for _ in range(3):
        obs = env.reset()
        for _ in range(200):
            logits = policy.forward(obs)
            action = np.argmax(logits)

            obs_norms.append(np.linalg.norm(obs))
            act_norms.append(np.linalg.norm(logits))
            actions.append(action)
            observations.append(obs.copy())
            logits_history.append(logits.copy())

            obs, _, done = env.step(action)
            if done:
                break

    metrics = {}

    # 1. Simple K-Index (correlation)
    try:
        r, _ = pearsonr(obs_norms, act_norms)
        metrics['simple_k'] = 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        metrics['simple_k'] = 0.0

    # 2. Behavioral Diversity (action entropy)
    action_counts = Counter(actions)
    if len(action_counts) > 1:
        probs = np.array([c / len(actions) for c in action_counts.values()])
        metrics['behavioral_diversity'] = entropy(probs) / np.log(len(action_counts))
    else:
        metrics['behavioral_diversity'] = 0.0

    # 3. Context-Sensitivity (response variance for similar inputs)
    # Group observations by magnitude and measure action variance
    if len(observations) > 10:
        obs_mags = [np.linalg.norm(o) for o in observations]
        # Bin observations
        bins = np.linspace(min(obs_mags), max(obs_mags) + 0.01, 6)
        bin_actions = {i: [] for i in range(5)}
        for mag, act in zip(obs_mags, actions):
            for i in range(5):
                if bins[i] <= mag < bins[i+1]:
                    bin_actions[i].append(act)
                    break

        # Measure action variance within bins
        variances = []
        for acts in bin_actions.values():
            if len(acts) > 1:
                variances.append(np.var(acts))
        metrics['context_sensitivity'] = np.mean(variances) if variances else 0.0
    else:
        metrics['context_sensitivity'] = 0.0

    # 4. Adaptation Rate (how much behavior changes over time)
    if len(actions) > 20:
        # Compare first half to second half
        mid = len(actions) // 2
        first_counts = Counter(actions[:mid])
        second_counts = Counter(actions[mid:])

        # Distribution shift
        p1 = np.array([first_counts.get(i, 0) / mid for i in range(2)])
        p2 = np.array([second_counts.get(i, 0) / (len(actions) - mid) for i in range(2)])
        metrics['adaptation_rate'] = np.linalg.norm(p2 - p1)
    else:
        metrics['adaptation_rate'] = 0.0

    # 5. Response Magnitude Variance (opposite of proportional response)
    if len(logits_history) > 5:
        logit_mags = [np.linalg.norm(l) for l in logits_history]
        metrics['response_variance'] = np.std(logit_mags)
    else:
        metrics['response_variance'] = 0.0

    # 6. Decision Boundary Sharpness (softmax confidence)
    if len(logits_history) > 5:
        confidences = []
        for logits in logits_history:
            exp_logits = np.exp(logits - np.max(logits))
            softmax = exp_logits / exp_logits.sum()
            confidences.append(np.max(softmax))
        metrics['decision_sharpness'] = np.mean(confidences)
    else:
        metrics['decision_sharpness'] = 0.5

    # 7. State-Action Mutual Information (discretized)
    if len(observations) > 20:
        # Discretize observations
        obs_mags = [np.linalg.norm(o) for o in observations]
        bins = np.linspace(min(obs_mags), max(obs_mags) + 0.01, 5)
        obs_bins = np.digitize(obs_mags, bins) - 1

        # Joint distribution
        joint = Counter(zip(obs_bins, actions))
        joint_probs = np.array([c / len(observations) for c in joint.values()])
        h_joint = entropy(joint_probs) if len(joint_probs) > 1 else 0

        # Marginals
        obs_probs = np.array([Counter(obs_bins)[i] / len(obs_bins) for i in range(4)])
        act_probs = np.array([Counter(actions)[i] / len(actions) for i in range(2)])
        h_obs = entropy(obs_probs[obs_probs > 0])
        h_act = entropy(act_probs[act_probs > 0])

        # MI = H(obs) + H(act) - H(joint)
        metrics['mutual_info'] = max(0, h_obs + h_act - h_joint)
    else:
        metrics['mutual_info'] = 0.0

    return metrics


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🎯 Task-Predictive Metrics Discovery                         ║")
    print("║                                                                ║")
    print("║  Finding what ACTUALLY correlates with CartPole performance   ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)
    env = CartPoleEnv()
    policy = Policy()

    # Optimize for PERFORMANCE (not K-Index!)
    cmaes = CMAES(dim=234, population_size=20, sigma=0.5)

    all_data = []

    print("Training policies optimized for TASK PERFORMANCE...\n")
    print("Gen │ Perf  │ Simple K │ Diversity │ Context │ Adapt │ MI")
    print("────┼───────┼──────────┼───────────┼─────────┼───────┼──────")

    for gen in range(1, 31):
        pop = cmaes.ask()
        performances = []
        all_metrics = []

        for params in pop:
            policy.set_params(params)
            perf = evaluate_policy(policy, env)
            metrics = compute_all_metrics(policy, env)
            metrics['performance'] = perf
            performances.append(perf)
            all_metrics.append(metrics)

        cmaes.tell(pop, performances)

        best_idx = np.argmax(performances)
        best_perf = performances[best_idx]
        best_metrics = all_metrics[best_idx]

        all_data.append(best_metrics)

        if gen % 5 == 0 or gen == 1:
            print(f"{gen:3d} │ {best_perf:5.1f} │ {best_metrics['simple_k']:8.4f} │ "
                  f"{best_metrics['behavioral_diversity']:9.4f} │ {best_metrics['context_sensitivity']:7.4f} │ "
                  f"{best_metrics['adaptation_rate']:5.3f} │ {best_metrics['mutual_info']:5.3f}")

    # Compute correlations
    print("\n" + "═" * 70)
    print("\n📊 Metric Correlations with Task Performance:\n")

    perfs = [d['performance'] for d in all_data]

    correlations = {}
    metric_names = ['simple_k', 'behavioral_diversity', 'context_sensitivity',
                   'adaptation_rate', 'response_variance', 'decision_sharpness', 'mutual_info']

    for metric in metric_names:
        values = [d[metric] for d in all_data]
        try:
            r, p = pearsonr(perfs, values)
            correlations[metric] = (r, p)
        except:
            correlations[metric] = (0.0, 1.0)

    # Sort by absolute correlation
    sorted_metrics = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)

    print("  Metric                    │  r      │ p-value │ Status")
    print("  ──────────────────────────┼─────────┼─────────┼────────")

    for metric, (r, p) in sorted_metrics:
        sig = "✅" if p < 0.05 else "  "
        direction = "+" if r > 0 else "-"
        print(f"  {metric:25s} │ {direction}{abs(r):.4f} │ {p:.4f}  │ {sig}")

    # Find best predictor
    best_metric = sorted_metrics[0][0]
    best_r, best_p = sorted_metrics[0][1]

    print(f"\n🏆 Best Predictor: {best_metric}")
    print(f"   Correlation: r = {best_r:+.4f} (p = {best_p:.4f})")

    if best_r > 0:
        print(f"   Interpretation: Higher {best_metric} → Better performance")
    else:
        print(f"   Interpretation: Lower {best_metric} → Better performance")

    # Compare with K-Index
    k_r, k_p = correlations['simple_k']
    print(f"\n📉 K-Index Comparison:")
    print(f"   K-Index correlation: r = {k_r:+.4f} (p = {k_p:.4f})")

    if abs(best_r) > abs(k_r):
        improvement = abs(best_r) / (abs(k_r) + 0.001)
        print(f"   {best_metric} is {improvement:.1f}x better predictor than K-Index!")

    # Save results
    Path('logs/task_predictive_metrics').mkdir(parents=True, exist_ok=True)
    with open('logs/task_predictive_metrics/results.json', 'w') as f:
        json.dump({
            'history': all_data,
            'correlations': {k: {'r': float(v[0]), 'p': float(v[1])}
                           for k, v in correlations.items()},
            'best_metric': best_metric,
            'best_correlation': float(best_r),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved to logs/task_predictive_metrics/")
    print('\n"Find what matters, measure what predicts." 💚\n')


if __name__ == '__main__':
    main()
