#!/usr/bin/env python3
"""
Diversity-Controlled Analysis

Tests if policy convergence explains the weaker trained-agent correlation.

Hypotheses:
1. All agents converge → less diversity → weaker correlation
2. Partial correlation controlling for diversity should be stronger

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 diversity_analysis.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class FlexibilityTracker:
    def __init__(self):
        self.obs_history = []
        self.action_history = []

    def record(self, obs, action):
        self.obs_history.append(np.array(obs).flatten())
        self.action_history.append(np.array(action).flatten())

    def get_flexibility(self):
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-100:])
        actions = np.concatenate(self.action_history[-100:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def reset(self):
        self.obs_history = []
        self.action_history = []


class Policy:
    def __init__(self, obs_dim, action_dim):
        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
        self.lr = 0.003

    def get_action(self, obs):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        action = mean + std * np.random.randn(len(mean))
        return np.clip(action, -1, 1)

    def update(self, obs, action, advantage):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)
        self.weights += self.lr * np.clip(advantage, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)

    def get_policy_vector(self):
        """Return flattened policy parameters for diversity measurement."""
        return np.concatenate([self.weights.flatten(), self.log_std])


class CoordinationEnv:
    def __init__(self, n_agents=4, obs_dim=6, action_dim=3):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def reset(self):
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.target = np.random.randn(self.obs_dim)
        self.steps = 0
        return [self.state + np.random.randn(self.obs_dim) * 0.05 for _ in range(self.n_agents)]

    def step(self, actions):
        action_mean = np.mean(actions, axis=0)
        if len(action_mean) < self.obs_dim:
            action_mean = np.pad(action_mean, (0, self.obs_dim - len(action_mean)))
        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)
        coord = 1 - np.std([np.mean(a) for a in actions])
        reward = -dist + coord * 0.2
        self.steps += 1
        done = dist < 0.3 or self.steps >= 80
        return [self.state + np.random.randn(self.obs_dim) * 0.05 for _ in range(self.n_agents)], [reward] * self.n_agents, done


def train_and_evaluate(n_training_episodes=100):
    """Train team and return metrics including policy diversity."""
    n_agents = 4
    obs_dim = 6
    action_dim = 3

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [Policy(obs_dim, action_dim) for _ in range(n_agents)]

    # Track diversity during training
    early_diversity = None
    late_diversity = None

    for ep in range(n_training_episodes):
        obs_list = env.reset()
        episode_data = [[] for _ in range(n_agents)]

        for step in range(80):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            next_obs, rewards, done = env.step(actions)
            for i in range(n_agents):
                episode_data[i].append({'obs': obs_list[i], 'action': actions[i], 'reward': rewards[i]})
            obs_list = next_obs
            if done:
                break

        for i in range(n_agents):
            returns = []
            G = 0
            for t in reversed(range(len(episode_data[i]))):
                G = episode_data[i][t]['reward'] + 0.99 * G
                returns.insert(0, G)
            if len(returns) > 1:
                returns = np.array(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                for t, G in enumerate(returns):
                    policies[i].update(episode_data[i][t]['obs'], episode_data[i][t]['action'], G)

        # Measure diversity at early and late training
        if ep == 10:
            policy_vectors = [p.get_policy_vector() for p in policies]
            early_diversity = np.std([np.linalg.norm(pv) for pv in policy_vectors])

    # Final policy diversity
    policy_vectors = [p.get_policy_vector() for p in policies]
    late_diversity = np.std([np.linalg.norm(pv) for pv in policy_vectors])

    # Evaluation
    eval_rewards = []
    eval_flex = []
    for _ in range(5):
        obs_list = env.reset()
        trackers = [FlexibilityTracker() for _ in range(n_agents)]
        total_reward = 0
        for step in range(80):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            for i, (obs, action) in enumerate(zip(obs_list, actions)):
                trackers[i].record(obs, action)
            obs_list, rewards, done = env.step(actions)
            total_reward += rewards[0]
            if done:
                break
        eval_rewards.append(total_reward)
        eval_flex.append(np.mean([t.get_flexibility() for t in trackers]))

    # Policy norm as diversity proxy
    policy_norm = np.mean([np.linalg.norm(p.get_policy_vector()) for p in policies])

    return {
        'flexibility': np.mean(eval_flex),
        'reward': np.mean(eval_rewards),
        'policy_norm': policy_norm,
        'early_diversity': early_diversity,
        'late_diversity': late_diversity
    }


def partial_correlation(x, y, z):
    """Compute partial correlation of x and y controlling for z."""
    # Residualize x on z
    slope_xz = np.cov(x, z)[0, 1] / (np.var(z) + 1e-8)
    x_resid = x - slope_xz * z

    # Residualize y on z
    slope_yz = np.cov(y, z)[0, 1] / (np.var(z) + 1e-8)
    y_resid = y - slope_yz * z

    return stats.pearsonr(x_resid, y_resid)


def main():
    print("\n" + "=" * 70)
    print("DIVERSITY-CONTROLLED ANALYSIS")
    print("=" * 70)

    n_teams = 200
    print(f"\nTraining {n_teams} teams with diversity tracking...")

    results = []
    for i in range(n_teams):
        result = train_and_evaluate(n_training_episodes=100)
        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  Teams {i+1}/{n_teams} complete")

    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])
    policy_norm = np.array([r['policy_norm'] for r in results])

    # Basic correlation
    r_basic, p_basic = stats.pearsonr(flexibility, rewards)

    # Partial correlation controlling for policy norm
    r_partial, p_partial = partial_correlation(flexibility, rewards, policy_norm)

    # Correlation within diversity strata
    median_norm = np.median(policy_norm)
    high_div_idx = policy_norm > median_norm
    low_div_idx = policy_norm <= median_norm

    r_high_div, p_high_div = stats.pearsonr(flexibility[high_div_idx], rewards[high_div_idx])
    r_low_div, p_low_div = stats.pearsonr(flexibility[low_div_idx], rewards[low_div_idx])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Analysis':<35} {'r':>10} {'p':>15}")
    print("-" * 70)

    sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    print(f"{'Basic correlation':<35} {r_basic:>+10.3f} {p_basic:>15.2e} {sig(p_basic)}")
    print(f"{'Partial (controlling diversity)':<35} {r_partial:>+10.3f} {p_partial:>15.2e} {sig(p_partial)}")
    print(f"{'High diversity teams only':<35} {r_high_div:>+10.3f} {p_high_div:>15.2e} {sig(p_high_div)}")
    print(f"{'Low diversity teams only':<35} {r_low_div:>+10.3f} {p_low_div:>15.2e} {sig(p_low_div)}")

    # Test convergence hypothesis
    print("\n" + "-" * 70)
    print("CONVERGENCE HYPOTHESIS TEST")
    print("-" * 70)

    if r_partial > r_basic + 0.05:
        print("✓ Partial correlation stronger than basic")
        print("  → Diversity explains some of the attenuation")
    else:
        print("✗ Partial correlation not stronger")
        print("  → Diversity does not explain the attenuation")

    if r_high_div > r_low_div + 0.1:
        print("✓ Higher correlation in high-diversity teams")
        print(f"  High diversity: r = {r_high_div:+.3f}")
        print(f"  Low diversity: r = {r_low_div:+.3f}")
    else:
        print("→ Similar correlation across diversity levels")

    # Diversity reduction during training
    early_div = np.mean([r['early_diversity'] for r in results if r['early_diversity'] is not None])
    late_div = np.mean([r['late_diversity'] for r in results])

    print(f"\nDiversity during training:")
    print(f"  Early (ep 10): {early_div:.3f}")
    print(f"  Late (ep 100): {late_div:.3f}")
    print(f"  Change: {late_div - early_div:+.3f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diversity_analysis_{timestamp}.npz"
    np.savez(filename,
             flexibility=flexibility, rewards=rewards, policy_norm=policy_norm,
             r_basic=r_basic, r_partial=r_partial, r_high_div=r_high_div, r_low_div=r_low_div)
    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
