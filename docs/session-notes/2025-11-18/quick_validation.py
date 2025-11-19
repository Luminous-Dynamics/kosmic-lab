#!/usr/bin/env python3
"""
Quick Validation - Minimal experiment to complete within timeout.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u quick_validation.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Policy:
    def __init__(self, obs_dim, action_dim, entropy_coef=0.0, flex_bonus=0.0):
        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
        self.lr = 0.003
        self.entropy_coef = entropy_coef
        self.flex_bonus = flex_bonus
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

    def update(self, obs, action, advantage):
        flex = self.get_flexibility()
        std = np.exp(np.clip(self.log_std, -3, 0))
        entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * std**2))
        augmented_reward = advantage + self.flex_bonus * flex + self.entropy_coef * entropy

        mean = np.tanh(self.weights @ obs)
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)
        self.weights += self.lr * np.clip(augmented_reward, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)
        if self.entropy_coef > 0:
            self.log_std += self.lr * self.entropy_coef * 0.1
            self.log_std = np.clip(self.log_std, -3, 0)

    def get_policy_vector(self):
        return np.concatenate([self.weights.flatten(), self.log_std])

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


class Env:
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
        done = dist < 0.3 or self.steps >= 60
        return [self.state + np.random.randn(self.obs_dim) * 0.05 for _ in range(self.n_agents)], [reward] * self.n_agents, done


def train(n_ep=60, entropy=0.0, flex=0.0):
    env = Env()
    policies = [Policy(6, 3, entropy, flex) for _ in range(4)]

    for ep in range(n_ep):
        obs_list = env.reset()
        for p in policies:
            p.reset_history()
        data = [[] for _ in range(4)]

        for step in range(60):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            next_obs, rewards, done = env.step(actions)
            for i in range(4):
                data[i].append({'obs': obs_list[i], 'action': actions[i], 'reward': rewards[i]})
            obs_list = next_obs
            if done:
                break

        for i in range(4):
            G = 0
            returns = []
            for t in reversed(range(len(data[i]))):
                G = data[i][t]['reward'] + 0.99 * G
                returns.insert(0, G)
            if len(returns) > 1:
                returns = np.array(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                for t, G in enumerate(returns):
                    policies[i].update(data[i][t]['obs'], data[i][t]['action'], G)

    # Eval
    rewards = []
    flexes = []
    for _ in range(3):
        obs_list = env.reset()
        total = 0
        for p in policies:
            p.reset_history()
        for step in range(60):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            obs_list, rews, done = env.step(actions)
            total += rews[0]
            if done:
                break
        rewards.append(total)
        flexes.append(np.mean([p.get_flexibility() for p in policies]))

    diversity = np.std([np.linalg.norm(p.get_policy_vector()) for p in policies])
    return np.mean(flexes), np.mean(rewards), diversity


def main():
    print("\n" + "=" * 60)
    print("QUICK VALIDATION SUITE")
    print("=" * 60)

    # Test conditions
    conditions = [
        (0.0, 0.0, "Baseline"),
        (0.1, 0.0, "Entropy 0.1"),
        (0.0, 0.25, "Flex 0.25"),
        (0.1, 0.25, "Combined"),
    ]

    all_results = {}
    for entropy, flex_bonus, name in conditions:
        print(f"\n{name}:")
        results = []
        for i in range(30):
            f, r, d = train(n_ep=60, entropy=entropy, flex=flex_bonus)
            results.append({'flex': f, 'reward': r, 'div': d})
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/30 complete")

        all_results[name] = results
        flexibility = np.array([x['flex'] for x in results])
        rewards = np.array([x['reward'] for x in results])
        r, p = stats.pearsonr(flexibility, rewards)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  r = {r:+.3f}, p = {p:.3f}{sig}, reward = {rewards.mean():.1f}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    baseline = [x['reward'] for x in all_results["Baseline"]]
    for name in ["Entropy 0.1", "Flex 0.25", "Combined"]:
        treatment = [x['reward'] for x in all_results[name]]
        d = (np.mean(treatment) - np.mean(baseline)) / np.sqrt((np.std(treatment)**2 + np.std(baseline)**2) / 2)
        t, p = stats.ttest_ind(treatment, baseline)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{name} vs Baseline: d = {d:+.3f}, p = {p:.3f}{sig}")

    # Best condition correlation
    print("\n" + "-" * 60)
    combined_flex = np.array([x['flex'] for x in all_results["Combined"]])
    combined_reward = np.array([x['reward'] for x in all_results["Combined"]])
    r, p = stats.pearsonr(combined_flex, combined_reward)

    # Effect size
    median = np.median(combined_flex)
    high = combined_reward[combined_flex > median]
    low = combined_reward[combined_flex <= median]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"Combined condition: r = {r:+.3f}, d = {d:+.3f}, p = {p:.3f} {sig}")

    if p < 0.05:
        print("\n✓ SIGNIFICANT: Flexibility predicts coordination with improved method")
    elif p < 0.10:
        print("\n⚠️ MARGINAL: Trend visible, scale up for confirmation")
    else:
        print("\n→ Not significant at this sample size")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_validation_{timestamp}.npz"
    np.savez(filename, **{k: [(x['flex'], x['reward'], x['div']) for x in v] for k, v in all_results.items()})
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
