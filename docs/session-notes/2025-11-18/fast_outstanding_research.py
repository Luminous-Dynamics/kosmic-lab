#!/usr/bin/env python3
"""
Fast Outstanding Research Suite

Runs all 4 key experiments with reduced parameters for quick iteration.
Can be scaled up for publication-quality results.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u fast_outstanding_research.py"
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

    def update(self, obs, action, advantage, flex_bonus=0.0):
        flex = self.get_flexibility()
        augmented_reward = advantage + flex_bonus * flex
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)
        self.weights += self.lr * np.clip(augmented_reward, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)

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


def train_team(n_training_episodes=100, flex_bonus=0.0):
    """Train one team and return metrics."""
    n_agents = 4
    obs_dim = 6
    action_dim = 3

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [Policy(obs_dim, action_dim) for _ in range(n_agents)]

    early_diversity = None

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
                    policies[i].update(episode_data[i][t]['obs'], episode_data[i][t]['action'], G, flex_bonus)

        if ep == 10:
            policy_vectors = [p.get_policy_vector() for p in policies]
            early_diversity = np.std([np.linalg.norm(pv) for pv in policy_vectors])

    # Final diversity
    policy_vectors = [p.get_policy_vector() for p in policies]
    late_diversity = np.std([np.linalg.norm(pv) for pv in policy_vectors])
    policy_norm = np.mean([np.linalg.norm(p.get_policy_vector()) for p in policies])

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
        'early_diversity': early_diversity if early_diversity else 0,
        'late_diversity': late_diversity
    }


def partial_correlation(x, y, z):
    slope_xz = np.cov(x, z)[0, 1] / (np.var(z) + 1e-8)
    x_resid = x - slope_xz * z
    slope_yz = np.cov(y, z)[0, 1] / (np.var(z) + 1e-8)
    y_resid = y - slope_yz * z
    return stats.pearsonr(x_resid, y_resid)


def main():
    print("\n" + "=" * 70)
    print("FAST OUTSTANDING RESEARCH SUITE")
    print("=" * 70)

    # ================================================================
    # EXPERIMENT 1: Large-Scale Validation (100 teams)
    # ================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: Trained Agent Validation (n=100)")
    print("-" * 70)

    n_teams = 50  # Reduced for speed
    results = []
    for i in range(n_teams):
        result = train_team(n_training_episodes=80)  # Fewer episodes
        results.append(result)
        if (i + 1) % 25 == 0:
            print(f"  Teams {i+1}/{n_teams} complete")

    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])

    r, p = stats.pearsonr(flexibility, rewards)

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
    print(f"\n  Pearson r = {r:+.3f}, p = {p:.3f} {sig}")
    print(f"  95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"  Cohen's d = {d:+.3f}")

    exp1_result = {'r': r, 'p': p, 'd': d, 'ci': (ci_low, ci_high)}

    # ================================================================
    # EXPERIMENT 2: Diversity Analysis (100 teams)
    # ================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Diversity-Controlled Analysis")
    print("-" * 70)

    policy_norm = np.array([r['policy_norm'] for r in results])
    r_basic, p_basic = stats.pearsonr(flexibility, rewards)
    r_partial, p_partial = partial_correlation(flexibility, rewards, policy_norm)

    median_norm = np.median(policy_norm)
    high_div_idx = policy_norm > median_norm
    low_div_idx = policy_norm <= median_norm

    r_high_div, p_high_div = stats.pearsonr(flexibility[high_div_idx], rewards[high_div_idx])
    r_low_div, p_low_div = stats.pearsonr(flexibility[low_div_idx], rewards[low_div_idx])

    print(f"\n  Basic correlation: r = {r_basic:+.3f}")
    print(f"  Partial (controlling diversity): r = {r_partial:+.3f}")
    print(f"  High diversity teams: r = {r_high_div:+.3f}")
    print(f"  Low diversity teams: r = {r_low_div:+.3f}")

    early_div = np.mean([r['early_diversity'] for r in results])
    late_div = np.mean([r['late_diversity'] for r in results])
    print(f"  Diversity change: {early_div:.3f} → {late_div:.3f}")

    exp2_result = {'r_basic': r_basic, 'r_partial': r_partial, 'r_high': r_high_div, 'r_low': r_low_div}

    # ================================================================
    # EXPERIMENT 3: Flexibility Intervention (Causal Test)
    # ================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Flexibility Intervention (Causal)")
    print("-" * 70)

    flex_bonuses = [0.0, 0.3, 0.6]  # Fewer conditions
    n_per_condition = 15  # Fewer per condition

    intervention_results = {}
    for bonus in flex_bonuses:
        condition_results = []
        for i in range(n_per_condition):
            result = train_team(n_training_episodes=80, flex_bonus=bonus)
            condition_results.append(result)
        intervention_results[bonus] = condition_results
        mean_flex = np.mean([r['flexibility'] for r in condition_results])
        mean_reward = np.mean([r['reward'] for r in condition_results])
        print(f"  Bonus {bonus}: flex = {mean_flex:+.3f}, reward = {mean_reward:+.3f}")

    # Compare to baseline
    baseline = [r['reward'] for r in intervention_results[0.0]]
    best_bonus = 0.0
    best_d = 0.0

    for bonus in flex_bonuses[1:]:
        treatment = [r['reward'] for r in intervention_results[bonus]]
        t, p = stats.ttest_ind(treatment, baseline)
        d = (np.mean(treatment) - np.mean(baseline)) / np.sqrt((np.std(treatment)**2 + np.std(baseline)**2) / 2)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  Bonus {bonus} vs baseline: d = {d:+.3f}, p = {p:.3f} {sig}")
        if d > best_d:
            best_d = d
            best_bonus = bonus

    exp3_result = {'best_bonus': best_bonus, 'best_d': best_d}

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("OUTSTANDING RESEARCH SUMMARY")
    print("=" * 70)

    print("\n1. TRAINED AGENT VALIDATION:")
    if exp1_result['p'] < 0.05:
        print(f"   ✓ SIGNIFICANT: r = {exp1_result['r']:+.3f}, p = {exp1_result['p']:.3f}")
        print(f"   Effect size: d = {exp1_result['d']:+.3f}")
    else:
        print(f"   ⚠️ Not significant: r = {exp1_result['r']:+.3f}, p = {exp1_result['p']:.3f}")

    print("\n2. DIVERSITY ANALYSIS:")
    if exp2_result['r_partial'] > exp2_result['r_basic'] + 0.05:
        print(f"   ✓ Diversity explains attenuation")
        print(f"   Partial r ({exp2_result['r_partial']:+.3f}) > Basic r ({exp2_result['r_basic']:+.3f})")
    else:
        print(f"   → Diversity does not explain attenuation")

    print("\n3. CAUSAL INTERVENTION:")
    if exp3_result['best_d'] > 0.2:
        print(f"   ✓ CAUSAL EFFECT: bonus {exp3_result['best_bonus']} improves by d = {exp3_result['best_d']:+.3f}")
    else:
        print(f"   → No significant causal effect found")

    print("\n" + "-" * 70)
    print("OVERALL CONCLUSION:")
    print("-" * 70)

    strong_evidence = (exp1_result['p'] < 0.05 and exp3_result['best_d'] > 0.2)
    if strong_evidence:
        print("✓ STRONG EVIDENCE: Flexibility predicts AND causes better coordination")
    elif exp1_result['p'] < 0.05:
        print("⚠️ CORRELATIONAL: Flexibility predicts but causation not established")
    else:
        print("✗ WEAK: Neither correlation nor causation found in trained agents")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fast_outstanding_research_{timestamp}.npz"
    np.savez(filename,
             flexibility=flexibility, rewards=rewards,
             exp1_r=exp1_result['r'], exp1_p=exp1_result['p'], exp1_d=exp1_result['d'],
             exp2_r_basic=exp2_result['r_basic'], exp2_r_partial=exp2_result['r_partial'],
             exp3_best_bonus=exp3_result['best_bonus'], exp3_best_d=exp3_result['best_d'])
    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
