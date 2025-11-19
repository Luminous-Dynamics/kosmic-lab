#!/usr/bin/env python3
"""
Focused Validation - Test best condition (entropy=0.1, flex=0.25) with n=80

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u focused_validation.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Policy:
    def __init__(self, obs_dim, action_dim, entropy_coef=0.1, flex_bonus=0.25):
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
        augmented = advantage + self.flex_bonus * flex + self.entropy_coef * entropy

        mean = np.tanh(self.weights @ obs)
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)
        self.weights += self.lr * np.clip(augmented, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)
        self.log_std += self.lr * self.entropy_coef * 0.1
        self.log_std = np.clip(self.log_std, -3, 0)

    def get_policy_vector(self):
        return np.concatenate([self.weights.flatten(), self.log_std])

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


class Env:
    def __init__(self):
        self.n_agents = 4
        self.obs_dim = 6
        self.action_dim = 3

    def reset(self):
        self.state = np.random.randn(6) * 0.1
        self.target = np.random.randn(6)
        self.steps = 0
        return [self.state + np.random.randn(6) * 0.05 for _ in range(4)]

    def step(self, actions):
        action_mean = np.mean(actions, axis=0)
        if len(action_mean) < 6:
            action_mean = np.pad(action_mean, (0, 6 - len(action_mean)))
        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)
        coord = 1 - np.std([np.mean(a) for a in actions])
        reward = -dist + coord * 0.2
        self.steps += 1
        done = dist < 0.3 or self.steps >= 60
        return [self.state + np.random.randn(6) * 0.05 for _ in range(4)], [reward] * 4, done


def train_team():
    env = Env()
    policies = [Policy(6, 3) for _ in range(4)]

    for ep in range(60):
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
    print("FOCUSED VALIDATION (n=80, entropy=0.1, flex=0.25)")
    print("=" * 60)

    n_teams = 80
    results = []
    for i in range(n_teams):
        f, r, d = train_team()
        results.append({'flex': f, 'reward': r, 'div': d})
        if (i + 1) % 20 == 0:
            print(f"  Teams {i+1}/{n_teams} complete")

    flexibility = np.array([x['flex'] for x in results])
    rewards = np.array([x['reward'] for x in results])
    diversity = np.array([x['div'] for x in results])

    r, p = stats.pearsonr(flexibility, rewards)
    rho, p_rho = stats.spearmanr(flexibility, rewards)

    # Bootstrap CI
    boot_r = []
    for _ in range(500):
        idx = np.random.choice(len(flexibility), len(flexibility), replace=True)
        boot_r.append(stats.pearsonr(flexibility[idx], rewards[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    # Effect size
    median = np.median(flexibility)
    high = rewards[flexibility > median]
    low = rewards[flexibility <= median]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    # Diversity subgroups
    med_div = np.median(diversity)
    r_high, p_high = stats.pearsonr(flexibility[diversity > med_div], rewards[diversity > med_div])
    r_low, p_low = stats.pearsonr(flexibility[diversity <= med_div], rewards[diversity <= med_div])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\nPearson r = {r:+.3f}, p = {p:.4f} {sig}")
    print(f"95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"Spearman ρ = {rho:+.3f}")
    print(f"Cohen's d = {d:+.3f}")

    print(f"\nHigh-diversity: r = {r_high:+.3f} (p={p_high:.3f})")
    print(f"Low-diversity: r = {r_low:+.3f} (p={p_low:.3f})")

    print(f"\nMean reward: {rewards.mean():.1f}")
    print(f"Mean flexibility: {flexibility.mean():+.3f}")
    print(f"Mean diversity: {diversity.mean():.3f}")

    print("\n" + "-" * 60)
    if p < 0.05:
        print("✓ SIGNIFICANT: Flexibility predicts coordination")
        print(f"  With entropy=0.1, flex_bonus=0.25")
    elif p < 0.10:
        print("⚠️ MARGINAL: Trend visible")
        print("  Recommendation: Scale to n=150")
    else:
        print("→ Not significant at n=80")
        print("  But check diversity subgroups above")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"focused_validation_{timestamp}.npz"
    np.savez(filename, flexibility=flexibility, rewards=rewards, diversity=diversity,
             r=r, p=p, d=d, ci_low=ci_low, ci_high=ci_high, r_high=r_high, r_low=r_low)
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
