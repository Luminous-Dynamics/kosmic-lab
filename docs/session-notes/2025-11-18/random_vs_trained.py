#!/usr/bin/env python3
"""
Random vs Trained Comparison

Directly compares the flexibility-reward correlation in:
1. Random policies (no training)
2. Trained policies (with entropy + flex bonus)

This establishes the fundamental difference.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u random_vs_trained.py"
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
        augmented = advantage + self.flex_bonus * flex + self.entropy_coef * entropy

        mean = np.tanh(self.weights @ obs)
        grad = (action - mean) / (std ** 2 + 1e-8)
        grad = np.clip(grad, -3, 3)
        self.weights += self.lr * np.clip(augmented, -5, 5) * np.outer(grad * (1 - mean**2), obs)
        self.weights = np.clip(self.weights, -5, 5)
        if self.entropy_coef > 0:
            self.log_std += self.lr * self.entropy_coef * 0.1
            self.log_std = np.clip(self.log_std, -3, 0)

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


def eval_team(policies):
    """Evaluate a team (random or trained)."""
    env = Env()
    rewards = []
    flexes = []

    for _ in range(3):
        obs_list = env.reset()
        for p in policies:
            p.reset_history()
        total = 0
        for step in range(60):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            obs_list, rews, done = env.step(actions)
            total += rews[0]
            if done:
                break
        rewards.append(total)
        flexes.append(np.mean([p.get_flexibility() for p in policies]))

    return np.mean(flexes), np.mean(rewards)


def train_team(policies, n_ep=60):
    """Train policies."""
    env = Env()

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


def main():
    print("\n" + "=" * 60)
    print("RANDOM VS TRAINED COMPARISON")
    print("=" * 60)

    n_teams = 80

    # ================================================================
    # RANDOM POLICIES
    # ================================================================
    print("\nRandom Policies:")
    random_results = []
    for i in range(n_teams):
        policies = [Policy(6, 3) for _ in range(4)]  # Random init
        f, r = eval_team(policies)  # No training
        random_results.append({'flex': f, 'reward': r})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_teams} complete")

    random_flex = np.array([x['flex'] for x in random_results])
    random_reward = np.array([x['reward'] for x in random_results])
    r_random, p_random = stats.pearsonr(random_flex, random_reward)

    # ================================================================
    # TRAINED POLICIES
    # ================================================================
    print("\nTrained Policies (entropy=0.1, flex=0.25):")
    trained_results = []
    for i in range(n_teams):
        policies = [Policy(6, 3, entropy_coef=0.1, flex_bonus=0.25) for _ in range(4)]
        train_team(policies, n_ep=60)  # Train them
        f, r = eval_team(policies)
        trained_results.append({'flex': f, 'reward': r})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_teams} complete")

    trained_flex = np.array([x['flex'] for x in trained_results])
    trained_reward = np.array([x['reward'] for x in trained_results])
    r_trained, p_trained = stats.pearsonr(trained_flex, trained_reward)

    # ================================================================
    # COMPARISON
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    sig_r = '***' if p_random < 0.001 else '**' if p_random < 0.01 else '*' if p_random < 0.05 else 'ns'
    sig_t = '***' if p_trained < 0.001 else '**' if p_trained < 0.01 else '*' if p_trained < 0.05 else 'ns'

    print(f"\n{'Condition':<20} {'r':>10} {'p':>12} {'Mean Reward':>15}")
    print("-" * 60)
    print(f"{'Random'::<20} {r_random:>+10.3f} {p_random:>12.4f} {sig_r:>3} {random_reward.mean():>12.1f}")
    print(f"{'Trained'::<20} {r_trained:>+10.3f} {p_trained:>12.4f} {sig_t:>3} {trained_reward.mean():>12.1f}")

    # Difference in correlations
    z_random = 0.5 * np.log((1 + r_random) / (1 - r_random + 1e-8))
    z_trained = 0.5 * np.log((1 + r_trained) / (1 - r_trained + 1e-8))
    z_diff = (z_random - z_trained) / np.sqrt(1/(n_teams-3) + 1/(n_teams-3))
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

    print(f"\nCorrelation difference: Δr = {r_random - r_trained:+.3f}")
    print(f"Fisher z-test: z = {z_diff:.2f}, p = {p_diff:.3f}")

    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    if r_random > 0.3 and p_random < 0.05:
        print("✓ Random policies: Strong flex-reward correlation")
    else:
        print("→ Random policies: Weak/no correlation")

    if r_trained > 0.3 and p_trained < 0.05:
        print("✓ Trained policies: Strong flex-reward correlation")
    else:
        print("→ Trained policies: Weak/no correlation")

    if p_diff < 0.05:
        print(f"\n⚠️ SIGNIFICANT DIFFERENCE between random and trained")
        print(f"   Training reduces correlation by {r_random - r_trained:+.3f}")
    else:
        print(f"\n→ No significant difference between conditions")

    print("\n" + "-" * 60)
    print("CONCLUSION")
    print("-" * 60)

    if r_random > r_trained + 0.2 and p_random < 0.05:
        print("The flexibility-reward relationship exists in random policies")
        print("but weakens or disappears after training.")
        print("\nThis suggests:")
        print("  1. Training causes policy convergence")
        print("  2. Convergence reduces the variance needed for correlation")
        print("  3. The effect is cross-sectional, not a training signal")
    else:
        print("Results don't show the expected pattern.")
        print("Need to investigate further.")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"random_vs_trained_{timestamp}.npz"
    np.savez(filename,
             random_flex=random_flex, random_reward=random_reward,
             trained_flex=trained_flex, trained_reward=trained_reward,
             r_random=r_random, r_trained=r_trained,
             p_random=p_random, p_trained=p_trained)
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
