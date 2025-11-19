#!/usr/bin/env python3
"""
Flexibility Intervention Study (Causal Test)

Tests: Does increasing flexibility CAUSE better coordination?

Approach:
1. Train with flexibility bonus vs without
2. Compare final performance
3. This moves from correlation to causation

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 flexibility_intervention.py"
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


class InterventionPolicy:
    """Policy with optional flexibility bonus."""

    def __init__(self, obs_dim, action_dim, flex_bonus=0.0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.flex_bonus = flex_bonus

        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
        self.lr = 0.003

        self.obs_history = []
        self.action_history = []

    def get_action(self, obs):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        action = mean + std * np.random.randn(self.action_dim)
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

    def update(self, obs, action, reward):
        """Update with optional flexibility bonus."""
        # Add flexibility bonus to reward
        flex = self.get_flexibility()
        augmented_reward = reward + self.flex_bonus * flex

        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)

        self.weights += self.lr * np.clip(augmented_reward, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)

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


def train_with_intervention(flex_bonus=0.0, n_episodes=150):
    """Train team with specified flexibility bonus."""
    n_agents = 4
    obs_dim = 6
    action_dim = 3

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [InterventionPolicy(obs_dim, action_dim, flex_bonus) for _ in range(n_agents)]

    for ep in range(n_episodes):
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

        # Update with returns
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

    # Evaluation (without flexibility bonus - pure task performance)
    eval_rewards = []
    eval_flex = []
    for _ in range(10):
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

    return np.mean(eval_flex), np.mean(eval_rewards)


def main():
    print("\n" + "=" * 70)
    print("FLEXIBILITY INTERVENTION STUDY (CAUSAL TEST)")
    print("=" * 70)

    # Test different flexibility bonus levels
    flex_bonuses = [0.0, 0.1, 0.2, 0.5, 1.0]
    n_per_condition = 50

    all_results = {}

    for bonus in flex_bonuses:
        print(f"\nCondition: flex_bonus = {bonus}...")
        results = []
        for i in range(n_per_condition):
            flex, reward = train_with_intervention(flex_bonus=bonus, n_episodes=150)
            results.append({'flexibility': flex, 'reward': reward})

        all_results[bonus] = results
        mean_flex = np.mean([r['flexibility'] for r in results])
        mean_reward = np.mean([r['reward'] for r in results])
        print(f"  Mean flex: {mean_flex:+.3f}, Mean reward: {mean_reward:+.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("INTERVENTION RESULTS")
    print("=" * 70)

    print(f"\n{'Flex Bonus':>12} {'Mean Flex':>12} {'Mean Reward':>14} {'n'}")
    print("-" * 50)

    baseline_reward = np.mean([r['reward'] for r in all_results[0.0]])

    for bonus, results in all_results.items():
        flex = np.mean([r['flexibility'] for r in results])
        reward = np.mean([r['reward'] for r in results])
        print(f"{bonus:>12.1f} {flex:>+12.3f} {reward:>+14.3f} {len(results)}")

    # Statistical tests
    print("\n" + "-" * 70)
    print("CAUSAL ANALYSIS: Does flexibility bonus improve performance?")
    print("-" * 70)

    baseline = [r['reward'] for r in all_results[0.0]]
    best_bonus = 0.0
    best_d = 0.0

    for bonus in flex_bonuses[1:]:
        treatment = [r['reward'] for r in all_results[bonus]]
        t, p = stats.ttest_ind(treatment, baseline)
        d = (np.mean(treatment) - np.mean(baseline)) / np.sqrt((np.std(treatment)**2 + np.std(baseline)**2) / 2)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        direction = "better" if d > 0 else "worse"
        print(f"Bonus {bonus}: d = {d:+.3f} ({direction}), t = {t:.2f}, p = {p:.3f} {sig}")

        if d > best_d:
            best_d = d
            best_bonus = bonus

    # Dose-response analysis
    print("\n" + "-" * 70)
    print("DOSE-RESPONSE ANALYSIS")
    print("-" * 70)

    bonuses = []
    mean_rewards = []
    mean_flex = []
    for bonus, results in all_results.items():
        bonuses.append(bonus)
        mean_rewards.append(np.mean([r['reward'] for r in results]))
        mean_flex.append(np.mean([r['flexibility'] for r in results]))

    # Correlation between bonus and reward
    r_dose, p_dose = stats.pearsonr(bonuses, mean_rewards)
    print(f"Bonus → Reward: r = {r_dose:+.3f}, p = {p_dose:.3f}")

    # Correlation between bonus and flexibility
    r_flex, p_flex = stats.pearsonr(bonuses, mean_flex)
    print(f"Bonus → Flexibility: r = {r_flex:+.3f}, p = {p_flex:.3f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CAUSAL CONCLUSION")
    print("=" * 70)

    if best_d > 0.3 and stats.ttest_ind([r['reward'] for r in all_results[best_bonus]], baseline)[1] < 0.05:
        print(f"\n✓ CAUSAL EFFECT ESTABLISHED")
        print(f"  Best bonus: {best_bonus}")
        print(f"  Effect size: d = {best_d:+.3f}")
        print(f"  Increasing flexibility CAUSES better coordination")
    elif best_d > 0:
        print(f"\n⚠️ POSITIVE TREND but not significant")
        print(f"  Best bonus: {best_bonus}, d = {best_d:+.3f}")
        print(f"  Need larger sample size to confirm")
    else:
        print(f"\n✗ NO CAUSAL EFFECT")
        print(f"  Flexibility bonus did not improve performance")

    # Check if intervention increased flexibility
    baseline_flex = np.mean([r['flexibility'] for r in all_results[0.0]])
    best_flex = np.mean([r['flexibility'] for r in all_results[best_bonus]])

    if best_flex > baseline_flex:
        print(f"\n✓ Intervention increased flexibility: {baseline_flex:+.3f} → {best_flex:+.3f}")
    else:
        print(f"\n⚠️ Intervention did not increase flexibility")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flexibility_intervention_{timestamp}.npz"

    save_data = {}
    for bonus, results in all_results.items():
        key = f"bonus_{str(bonus).replace('.', '_')}"
        save_data[f"{key}_flex"] = [r['flexibility'] for r in results]
        save_data[f"{key}_reward"] = [r['reward'] for r in results]

    np.savez(filename, **save_data)
    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
