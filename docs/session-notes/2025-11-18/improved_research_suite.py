#!/usr/bin/env python3
"""
Improved Research Suite

Addresses key improvements:
1. Entropy regularization to maintain diversity
2. Fine-tuned flexibility bonus levels
3. Larger sample sizes for statistical power
4. Comparison of diversity preservation methods

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u improved_research_suite.py"
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


class EntropyRegularizedPolicy:
    """Policy with entropy regularization to maintain diversity."""

    def __init__(self, obs_dim, action_dim, entropy_coef=0.0, flex_bonus=0.0):
        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
        self.lr = 0.003
        self.entropy_coef = entropy_coef
        self.flex_bonus = flex_bonus
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_history = []
        self.action_history = []

    def get_action(self, obs):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        action = mean + std * np.random.randn(len(mean))
        action = np.clip(action, -1, 1)
        self.obs_history.append(obs.flatten())
        self.action_history.append(action.flatten())
        return action

    def get_flexibility(self):
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-50:])
        actions = np.concatenate(self.action_history[-50:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def get_entropy(self):
        """Gaussian entropy: 0.5 * log(2πe * σ²)"""
        std = np.exp(np.clip(self.log_std, -3, 0))
        return 0.5 * np.sum(np.log(2 * np.pi * np.e * std**2))

    def update(self, obs, action, advantage):
        # Compute bonuses
        flex = self.get_flexibility()
        entropy = self.get_entropy()

        # Augmented reward
        augmented_reward = advantage + self.flex_bonus * flex + self.entropy_coef * entropy

        # Policy gradient
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)

        self.weights += self.lr * np.clip(augmented_reward, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)

        # Update log_std towards higher entropy if entropy_coef > 0
        if self.entropy_coef > 0:
            self.log_std += self.lr * self.entropy_coef * 0.1
            self.log_std = np.clip(self.log_std, -3, 0)

    def get_policy_vector(self):
        return np.concatenate([self.weights.flatten(), self.log_std])

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


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


def train_team(n_training_episodes=80, entropy_coef=0.0, flex_bonus=0.0):
    """Train one team with specified regularization."""
    n_agents = 4
    obs_dim = 6
    action_dim = 3

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [EntropyRegularizedPolicy(obs_dim, action_dim, entropy_coef, flex_bonus)
                for _ in range(n_agents)]

    for ep in range(n_training_episodes):
        obs_list = env.reset()
        for p in policies:
            p.reset_history()
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

    # Compute diversity metrics
    policy_vectors = [p.get_policy_vector() for p in policies]
    policy_norm = np.mean([np.linalg.norm(pv) for pv in policy_vectors])
    policy_diversity = np.std([np.linalg.norm(pv) for pv in policy_vectors])

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

    return {
        'flexibility': np.mean(eval_flex),
        'reward': np.mean(eval_rewards),
        'policy_norm': policy_norm,
        'policy_diversity': policy_diversity
    }


def main():
    print("\n" + "=" * 70)
    print("IMPROVED RESEARCH SUITE")
    print("=" * 70)

    # ================================================================
    # EXPERIMENT 1: Entropy Regularization Comparison (n=80)
    # ================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: Entropy Regularization Comparison")
    print("-" * 70)

    entropy_levels = [0.0, 0.05, 0.1, 0.2]
    n_per_condition = 20

    entropy_results = {}
    for entropy in entropy_levels:
        results = []
        for _ in range(n_per_condition):
            result = train_team(n_training_episodes=80, entropy_coef=entropy)
            results.append(result)
        entropy_results[entropy] = results

        flex = np.array([r['flexibility'] for r in results])
        rewards = np.array([r['reward'] for r in results])
        diversity = np.array([r['policy_diversity'] for r in results])
        r, p = stats.pearsonr(flex, rewards)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  Entropy {entropy}: r={r:+.3f}{sig}, div={diversity.mean():.3f}, reward={rewards.mean():.1f}")

    # Best entropy for flex-reward correlation
    best_entropy = max(entropy_levels,
                       key=lambda e: stats.pearsonr(
                           [r['flexibility'] for r in entropy_results[e]],
                           [r['reward'] for r in entropy_results[e]])[0])

    print(f"\n  Best entropy for correlation: {best_entropy}")

    # ================================================================
    # EXPERIMENT 2: Fine-tuned Flexibility Bonus (n=100)
    # ================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Fine-tuned Flexibility Bonus")
    print("-" * 70)

    flex_bonuses = [0.0, 0.15, 0.25, 0.35, 0.45]
    n_per_condition = 20

    flex_results = {}
    for bonus in flex_bonuses:
        results = []
        for _ in range(n_per_condition):
            result = train_team(n_training_episodes=80, flex_bonus=bonus)
            results.append(result)
        flex_results[bonus] = results

        rewards = np.array([r['reward'] for r in results])
        flexibility = np.array([r['flexibility'] for r in results])
        print(f"  Bonus {bonus}: reward={rewards.mean():.1f}±{rewards.std():.1f}, flex={flexibility.mean():+.3f}")

    # Compare to baseline
    baseline = [r['reward'] for r in flex_results[0.0]]
    best_bonus = 0.0
    best_d = 0.0

    for bonus in flex_bonuses[1:]:
        treatment = [r['reward'] for r in flex_results[bonus]]
        d = (np.mean(treatment) - np.mean(baseline)) / np.sqrt((np.std(treatment)**2 + np.std(baseline)**2) / 2)
        t, p = stats.ttest_ind(treatment, baseline)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  Bonus {bonus} vs baseline: d={d:+.3f}, p={p:.3f}{sig}")
        if d > best_d:
            best_d = d
            best_bonus = bonus

    print(f"\n  Optimal bonus: {best_bonus} (d={best_d:+.3f})")

    # ================================================================
    # EXPERIMENT 3: Combined Entropy + Flex Bonus (n=80)
    # ================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Combined Entropy + Flexibility Bonus")
    print("-" * 70)

    # Use best from previous experiments
    conditions = [
        (0.0, 0.0, "Baseline"),
        (best_entropy, 0.0, f"Entropy {best_entropy}"),
        (0.0, best_bonus, f"Flex {best_bonus}"),
        (best_entropy, best_bonus, "Combined"),
    ]
    n_per_condition = 25

    combined_results = {}
    for entropy, bonus, name in conditions:
        results = []
        for _ in range(n_per_condition):
            result = train_team(n_training_episodes=80, entropy_coef=entropy, flex_bonus=bonus)
            results.append(result)
        combined_results[name] = results

        flex = np.array([r['flexibility'] for r in results])
        rewards = np.array([r['reward'] for r in results])
        r, p = stats.pearsonr(flex, rewards)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {name}: r={r:+.3f}{sig}, reward={rewards.mean():.1f}")

    # Compare combined to baseline
    baseline_rewards = [r['reward'] for r in combined_results["Baseline"]]
    combined_rewards = [r['reward'] for r in combined_results["Combined"]]
    d_combined = (np.mean(combined_rewards) - np.mean(baseline_rewards)) / \
                 np.sqrt((np.std(combined_rewards)**2 + np.std(baseline_rewards)**2) / 2)
    t, p = stats.ttest_ind(combined_rewards, baseline_rewards)

    print(f"\n  Combined vs Baseline: d={d_combined:+.3f}, p={p:.3f}")

    # ================================================================
    # EXPERIMENT 4: Large-Scale Validation with Best Method (n=100)
    # ================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 4: Large-Scale Validation with Best Method")
    print("-" * 70)

    n_teams = 100
    results = []
    for i in range(n_teams):
        result = train_team(n_training_episodes=80, entropy_coef=best_entropy, flex_bonus=best_bonus)
        results.append(result)
        if (i + 1) % 25 == 0:
            print(f"  Teams {i+1}/{n_teams} complete")

    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])
    diversity = np.array([r['policy_diversity'] for r in results])

    r, p = stats.pearsonr(flexibility, rewards)
    rho, p_rho = stats.spearmanr(flexibility, rewards)

    # Bootstrap CI
    boot_r = []
    for _ in range(500):
        idx = np.random.choice(len(flexibility), len(flexibility), replace=True)
        boot_r.append(stats.pearsonr(flexibility[idx], rewards[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    # Effect size
    median_flex = np.median(flexibility)
    high = rewards[flexibility > median_flex]
    low = rewards[flexibility <= median_flex]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\n  Pearson r = {r:+.3f}, p = {p:.4f} {sig}")
    print(f"  95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"  Cohen's d = {d:+.3f}")
    print(f"  Mean diversity: {diversity.mean():.3f}")

    # Diversity subgroup analysis
    median_div = np.median(diversity)
    high_div_idx = diversity > median_div
    low_div_idx = diversity <= median_div

    r_high, p_high = stats.pearsonr(flexibility[high_div_idx], rewards[high_div_idx])
    r_low, p_low = stats.pearsonr(flexibility[low_div_idx], rewards[low_div_idx])

    print(f"\n  High-diversity teams: r = {r_high:+.3f} (p={p_high:.3f})")
    print(f"  Low-diversity teams: r = {r_low:+.3f} (p={p_low:.3f})")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("IMPROVED RESEARCH SUMMARY")
    print("=" * 70)

    print(f"\n1. ENTROPY REGULARIZATION:")
    print(f"   Best entropy coefficient: {best_entropy}")

    print(f"\n2. FLEXIBILITY BONUS:")
    print(f"   Optimal bonus: {best_bonus} (d = {best_d:+.3f})")

    print(f"\n3. COMBINED METHOD:")
    print(f"   Improvement over baseline: d = {d_combined:+.3f}")

    print(f"\n4. LARGE-SCALE VALIDATION (n={n_teams}):")
    if p < 0.05:
        print(f"   ✓ SIGNIFICANT: r = {r:+.3f}, p = {p:.4f}")
    elif p < 0.10:
        print(f"   ⚠️ MARGINAL: r = {r:+.3f}, p = {p:.4f}")
    else:
        print(f"   ✗ Not significant: r = {r:+.3f}, p = {p:.4f}")

    print(f"   Effect size: d = {d:+.3f}")
    print(f"   High-diversity: r = {r_high:+.3f}")
    print(f"   Low-diversity: r = {r_low:+.3f}")

    # Overall conclusion
    print("\n" + "-" * 70)
    print("OVERALL CONCLUSION:")
    print("-" * 70)

    if p < 0.05 and d > 0.3:
        print("✓ STRONG: Flexibility predicts coordination with improved methods")
        print(f"   Method: entropy={best_entropy}, flex_bonus={best_bonus}")
    elif p < 0.10:
        print("⚠️ PROMISING: Trend toward significance with improved methods")
        print("   Recommendation: Scale to n=200 for definitive result")
    else:
        print("→ NEEDS WORK: Effect not significant even with improvements")
        print("   Recommendation: Try alternative diversity preservation methods")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"improved_research_{timestamp}.npz"
    np.savez(filename,
             flexibility=flexibility, rewards=rewards, diversity=diversity,
             r=r, p=p, d=d, ci_low=ci_low, ci_high=ci_high,
             best_entropy=best_entropy, best_bonus=best_bonus,
             r_high=r_high, r_low=r_low)
    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
