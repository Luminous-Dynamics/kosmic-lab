#!/usr/bin/env python3
"""
PPO Implementation for Stable Multi-Agent Training

More stable than vanilla REINFORCE for cleaner flexibility signal.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 ppo_training.py"
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


class PPOPolicy:
    """PPO with clipped objective and GAE."""

    def __init__(self, obs_dim, action_dim, clip_epsilon=0.2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon

        # Policy network (simple linear)
        self.weights = np.random.randn(action_dim, obs_dim) * 0.1
        self.log_std = np.zeros(action_dim) - 0.5

        # Value network
        self.value_weights = np.random.randn(obs_dim) * 0.1
        self.value_bias = 0.0

        self.lr_policy = 0.0003
        self.lr_value = 0.001

    def get_action(self, obs):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -2, 0.5))
        action = mean + std * np.random.randn(self.action_dim)
        action = np.clip(action, -1, 1)

        log_prob = -0.5 * np.sum(((action - mean) / (std + 1e-8))**2 + np.log(2*np.pi) + 2*self.log_std)
        return action, log_prob

    def get_value(self, obs):
        return np.dot(self.value_weights, obs) + self.value_bias

    def get_log_prob(self, obs, action):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -2, 0.5))
        return -0.5 * np.sum(((action - mean) / (std + 1e-8))**2 + np.log(2*np.pi) + 2*self.log_std)

    def update(self, batch, n_epochs=4):
        """PPO update with multiple epochs."""
        obs_batch = np.array([t['obs'] for t in batch])
        action_batch = np.array([t['action'] for t in batch])
        advantage_batch = np.array([t['advantage'] for t in batch])
        return_batch = np.array([t['return'] for t in batch])
        old_log_prob_batch = np.array([t['log_prob'] for t in batch])

        # Normalize advantages
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

        for _ in range(n_epochs):
            for i in range(len(batch)):
                obs = obs_batch[i]
                action = action_batch[i]
                advantage = advantage_batch[i]
                ret = return_batch[i]
                old_log_prob = old_log_prob_batch[i]

                # New log probability
                new_log_prob = self.get_log_prob(obs, action)
                ratio = np.exp(new_log_prob - old_log_prob)

                # Clipped objective
                clip_adv = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                policy_loss = -min(ratio * advantage, clip_adv)

                # Policy gradient
                mean = np.tanh(self.weights @ obs)
                std = np.exp(np.clip(self.log_std, -2, 0.5))
                grad_mean = (action - mean) / (std**2 + 1e-8)
                grad_mean = np.clip(grad_mean, -5, 5)

                # Only update if not clipped
                if abs(ratio - 1) < self.clip_epsilon:
                    grad_weights = np.outer(grad_mean * (1 - mean**2), obs)
                    self.weights += self.lr_policy * advantage * np.clip(grad_weights, -0.5, 0.5)
                    self.weights = np.clip(self.weights, -3, 3)

                # Value update
                value = self.get_value(obs)
                value_error = ret - value
                self.value_weights += self.lr_value * value_error * obs
                self.value_bias += self.lr_value * value_error
                self.value_weights = np.clip(self.value_weights, -3, 3)


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


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def train_ppo_team(n_episodes=200):
    """Train one team with PPO."""
    n_agents = 4
    obs_dim = 6
    action_dim = 3

    env = CoordinationEnv(n_agents, obs_dim, action_dim)
    policies = [PPOPolicy(obs_dim, action_dim) for _ in range(n_agents)]

    for ep in range(n_episodes):
        obs_list = env.reset()
        episode_data = [[] for _ in range(n_agents)]

        for step in range(80):
            actions = []
            log_probs = []
            values = []

            for i, (policy, obs) in enumerate(zip(policies, obs_list)):
                action, log_prob = policy.get_action(obs)
                value = policy.get_value(obs)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            next_obs, rewards, done = env.step(actions)

            for i in range(n_agents):
                episode_data[i].append({
                    'obs': obs_list[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'log_prob': log_probs[i],
                    'value': values[i]
                })

            obs_list = next_obs
            if done:
                break

        # Compute GAE and returns for each agent
        for i in range(n_agents):
            rewards_i = [t['reward'] for t in episode_data[i]]
            values_i = [t['value'] for t in episode_data[i]]

            advantages = compute_gae(rewards_i, values_i)

            returns = []
            G = 0
            for r in reversed(rewards_i):
                G = r + 0.99 * G
                returns.insert(0, G)

            for t in range(len(episode_data[i])):
                episode_data[i][t]['advantage'] = advantages[t]
                episode_data[i][t]['return'] = returns[t]

            policies[i].update(episode_data[i])

    # Evaluation
    eval_rewards = []
    eval_flex = []
    for _ in range(5):
        obs_list = env.reset()
        trackers = [FlexibilityTracker() for _ in range(n_agents)]
        total_reward = 0
        for step in range(80):
            actions = [p.get_action(obs)[0] for p, obs in zip(policies, obs_list)]
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
    print("PPO TRAINING FOR STABLE MULTI-AGENT LEARNING")
    print("=" * 70)

    n_teams = 200
    print(f"\nTraining {n_teams} teams with PPO...")

    results = []
    for i in range(n_teams):
        flex, reward = train_ppo_team(n_episodes=200)
        results.append({'flexibility': flex, 'reward': reward})
        if (i + 1) % 50 == 0:
            print(f"  Teams {i+1}/{n_teams} complete")

    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])

    r, p = stats.pearsonr(flexibility, rewards)
    rho, p_rho = stats.spearmanr(flexibility, rewards)

    # Effect size
    median_flex = np.median(flexibility)
    high = rewards[flexibility > median_flex]
    low = rewards[flexibility <= median_flex]
    d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    print("\n" + "=" * 70)
    print("PPO RESULTS (n = 200)")
    print("=" * 70)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\nPearson r = {r:+.3f}, p = {p:.2e} {sig}")
    print(f"Spearman ρ = {rho:+.3f}, p = {p_rho:.2e}")
    print(f"Cohen's d = {d:+.3f}")
    print(f"Mean reward: {rewards.mean():+.3f}")
    print(f"Mean flexibility: {flexibility.mean():+.3f}")

    print("\n" + "-" * 70)
    if p < 0.05:
        print("✓ SIGNIFICANT with PPO training")
    else:
        print("✗ Not significant with PPO training")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ppo_training_{timestamp}.npz"
    np.savez(filename, flexibility=flexibility, rewards=rewards, r=r, p=p, d=d)
    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")

    return {'r': r, 'p': p, 'd': d}


if __name__ == '__main__':
    main()
