#!/usr/bin/env python3
"""
Entropy-Regularized RL Training

Addresses the finding that vanilla policy gradient makes agents rigid.
Uses entropy bonus to maintain flexibility during training.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 rl_entropy_regularized.py"
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from datetime import datetime


class FlexibilityTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.obs_history = []
        self.action_history = []

    def record(self, obs: np.ndarray, action: np.ndarray):
        self.obs_history.append(obs.flatten())
        self.action_history.append(action.flatten())

    def get_flexibility(self) -> float:
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-self.window_size:])
        actions = np.concatenate(self.action_history[-self.window_size:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        if np.isnan(corr):
            return 0.0
        return -abs(corr) * 2.0

    def reset(self):
        self.obs_history = []
        self.action_history = []


class SoftActorPolicy:
    """Soft actor with entropy regularization."""

    def __init__(self, obs_dim: int, action_dim: int, entropy_coef: float = 0.1):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.entropy_coef = entropy_coef

        # Simple linear policy with learnable mean and log_std
        self.weights = np.random.randn(action_dim, obs_dim) * 0.1
        self.log_std = np.zeros(action_dim)
        self.lr = 0.001

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Get action, log_prob, and entropy."""
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -2, 1))

        noise = np.random.randn(self.action_dim)
        action = mean + std * noise
        action = np.clip(action, -1, 1)

        # Log probability
        log_prob = -0.5 * np.sum(((action - mean) / (std + 1e-8)) ** 2 + np.log(2 * np.pi) + 2 * self.log_std)

        # Entropy (differential entropy of Gaussian)
        entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * std ** 2))

        return action, log_prob, entropy

    def update(self, obs: np.ndarray, action: np.ndarray, advantage: float, entropy: float):
        """Update with entropy bonus."""
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -2, 1))

        # Policy gradient with entropy bonus
        total_advantage = advantage + self.entropy_coef * entropy

        # Gradient for mean (simplified)
        grad_mean = (action - mean) / (std ** 2 + 1e-8)
        grad_mean = np.clip(grad_mean, -5, 5)

        # Weight update
        grad_weights = np.outer(grad_mean * (1 - mean ** 2), obs)
        self.weights += self.lr * np.clip(total_advantage, -10, 10) * np.clip(grad_weights, -1, 1)

        # Log std update (encourage higher entropy)
        grad_log_std = ((action - mean) ** 2 / (std ** 2 + 1e-8) - 1) + self.entropy_coef
        self.log_std += self.lr * 0.1 * np.clip(grad_log_std, -5, 5)
        self.log_std = np.clip(self.log_std, -2, 1)


class CoordinationEnv:
    def __init__(self, n_agents: int = 4, obs_dim: int = 8, action_dim: int = 4):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def reset(self):
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.target = np.random.randn(self.obs_dim)
        self.step_count = 0
        return [self.state + np.random.randn(self.obs_dim) * 0.05 for _ in range(self.n_agents)]

    def step(self, actions):
        action_mean = np.mean(actions, axis=0)
        # Pad action to match state dim
        if len(action_mean) < self.obs_dim:
            action_mean = np.pad(action_mean, (0, self.obs_dim - len(action_mean)))

        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)

        # Coordination bonus: reward similar actions
        action_var = np.mean([np.var(a) for a in actions])
        coord_bonus = -action_var * 0.5

        reward = -dist + coord_bonus
        self.step_count += 1
        done = dist < 0.3 or self.step_count >= 100

        obs = [self.state + np.random.randn(self.obs_dim) * 0.05 for _ in range(self.n_agents)]
        return obs, [reward] * self.n_agents, done


def train_with_entropy_reg(n_episodes: int = 500, entropy_coef: float = 0.1):
    """Train with entropy regularization."""
    n_agents = 4
    obs_dim = 8
    action_dim = 4

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [SoftActorPolicy(obs_dim, action_dim, entropy_coef) for _ in range(n_agents)]
    trackers = [FlexibilityTracker() for _ in range(n_agents)]

    data = {'episode': [], 'reward': [], 'flexibility': [], 'entropy': []}

    for ep in range(n_episodes):
        obs_list = env.reset()
        for t in trackers:
            t.reset()

        episode_data = [[] for _ in range(n_agents)]
        total_entropy = 0

        for step in range(100):
            actions = []
            for i in range(n_agents):
                action, log_prob, entropy = policies[i].get_action(obs_list[i])
                trackers[i].record(obs_list[i], action)
                actions.append(action)
                episode_data[i].append({'obs': obs_list[i].copy(), 'action': action.copy(),
                                       'log_prob': log_prob, 'entropy': entropy})
                total_entropy += entropy

            obs_list, rewards, done = env.step(actions)

            for i in range(n_agents):
                episode_data[i][-1]['reward'] = rewards[i]

            if done:
                break

        # Update policies
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
                    d = episode_data[i][t]
                    policies[i].update(d['obs'], d['action'], G, d['entropy'])

        # Track
        if ep % 10 == 0:
            mean_flex = np.mean([t.get_flexibility() for t in trackers])
            mean_reward = np.mean([sum(d['reward'] for d in ed) for ed in episode_data])
            mean_entropy = total_entropy / (len(episode_data[0]) * n_agents)

            data['episode'].append(ep)
            data['reward'].append(mean_reward)
            data['flexibility'].append(mean_flex)
            data['entropy'].append(mean_entropy)

    return data


def main():
    print("\n" + "=" * 70)
    print("ENTROPY-REGULARIZED RL TRAINING")
    print("=" * 70)

    # Test different entropy coefficients
    entropy_coefs = [0.0, 0.05, 0.1, 0.2]
    all_results = {}

    for coef in entropy_coefs:
        print(f"\nTraining with entropy_coef = {coef}...")
        data = train_with_entropy_reg(n_episodes=500, entropy_coef=coef)
        all_results[coef] = data

        # Analyze
        flex = np.array(data['flexibility'])
        reward = np.array(data['reward'])
        r, p = stats.pearsonr(flex, reward)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  Flex-reward: r = {r:+.3f}, p = {p:.2e} {sig}")
        print(f"  Final flex: {flex[-1]:+.3f}, Final reward: {reward[-1]:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("ENTROPY REGULARIZATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Entropy Coef':>12} {'r':>10} {'Final Flex':>12} {'Final Reward':>14}")
    print("-" * 60)

    for coef, data in all_results.items():
        flex = np.array(data['flexibility'])
        reward = np.array(data['reward'])
        r, _ = stats.pearsonr(flex, reward)
        print(f"{coef:>12.2f} {r:>+10.3f} {flex[-1]:>+12.3f} {reward[-1]:>+14.3f}")

    # Key finding
    print("\n" + "-" * 60)

    # Find best entropy coefficient
    best_coef = max(all_results.keys(),
                   key=lambda c: all_results[c]['reward'][-1])
    best_flex = all_results[best_coef]['flexibility'][-1]

    print(f"Best entropy coefficient: {best_coef}")
    print(f"Best final reward: {all_results[best_coef]['reward'][-1]:+.3f}")
    print(f"Flexibility at best: {best_flex:+.3f}")

    # Did entropy help maintain flexibility?
    no_ent_flex = all_results[0.0]['flexibility'][-1]
    high_ent_flex = all_results[0.2]['flexibility'][-1]

    if high_ent_flex > no_ent_flex:
        print(f"\n✓ Entropy regularization maintains flexibility")
        print(f"  Without entropy: flex = {no_ent_flex:+.3f}")
        print(f"  With entropy (0.2): flex = {high_ent_flex:+.3f}")
    else:
        print(f"\n✗ Entropy regularization did not improve flexibility")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"entropy_regularized_{timestamp}.npz"

    save_data = {}
    for coef, data in all_results.items():
        key = f"coef_{str(coef).replace('.', '_')}"
        save_data[f"{key}_episodes"] = data['episode']
        save_data[f"{key}_rewards"] = data['reward']
        save_data[f"{key}_flexibility"] = data['flexibility']

    np.savez(filename, **save_data)
    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
