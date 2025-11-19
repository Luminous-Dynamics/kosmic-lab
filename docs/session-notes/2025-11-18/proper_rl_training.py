#!/usr/bin/env python3
"""
Proper RL Training with REINFORCE + Baseline

Tests if flexibility-reward relationship holds in trained policies.
Uses proper policy gradient with:
- Baseline for variance reduction
- Advantage normalization
- Multiple training runs
- Longer training (100 episodes)

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u proper_rl_training.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class TrainableAgent:
    """Agent with proper REINFORCE training."""

    def __init__(self, agent_id: int, obs_dim: int = 10, action_dim: int = 10, lr: float = 0.005):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr

        # Policy parameters (Gaussian policy)
        self.weights = np.random.randn(action_dim, obs_dim + 5) * 0.1
        self.log_std = np.ones(action_dim) * -0.5

        # Episode storage
        self.states = []
        self.actions = []
        self.rewards = []

        # For flexibility calculation
        self.obs_history = []
        self.action_history = []

    def act(self, obs: np.ndarray, messages: np.ndarray) -> np.ndarray:
        combined = np.concatenate([obs, messages])
        mean = np.tanh(self.weights @ combined)
        std = np.exp(np.clip(self.log_std, -2, 0))

        action = mean + std * np.random.randn(self.action_dim)
        action = np.clip(action, -1, 1)

        self.states.append(combined)
        self.actions.append(action)
        self.obs_history.append(obs)
        self.action_history.append(action)

        return action

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self):
        """REINFORCE update with baseline and proper gradients."""
        if len(self.rewards) < 2:
            return

        # Compute returns
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Baseline and advantage
        baseline = returns.mean()
        advantages = returns - baseline
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)

        # Policy gradient update
        std = np.exp(np.clip(self.log_std, -2, 0))

        for t, (state, action, adv) in enumerate(zip(self.states, self.actions, advantages)):
            mean = np.tanh(self.weights @ state)

            # Gradient of log probability
            # d/dw log N(a|mean,std) = (a - mean) / std^2 * d(mean)/dw
            # d(mean)/dw = (1 - mean^2) * state (tanh derivative)

            diff = (action - mean) / (std ** 2 + 1e-8)
            tanh_grad = 1 - mean ** 2

            # Weight gradient: outer product
            grad_w = np.outer(diff * tanh_grad, state)

            # Update weights
            self.weights += self.lr * adv * grad_w

            # Log std gradient (simplified)
            grad_log_std = ((action - mean) ** 2 / (std ** 2) - 1)
            self.log_std += self.lr * 0.1 * adv * grad_log_std

        # Clip log_std
        self.log_std = np.clip(self.log_std, -2, 0)

    def reset_episode(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def get_flexibility(self):
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-50:]).flatten()
        actions = np.array(self.action_history[-50:]).flatten()
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return -abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def create_message(self, obs):
        return obs[:5]


class Network:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def exchange(self, messages):
        return [np.mean([messages[j] for j in range(self.n_agents) if self.adj[i,j] > 0], axis=0)
                for i in range(self.n_agents)]


class Environment:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.state = np.zeros(10)
        self.target = np.random.randn(10)

    def reset(self):
        self.state = np.random.randn(10) * 0.1
        self.target = np.random.randn(10)
        return self.state

    def step(self, actions):
        action_mean = np.mean(actions, axis=0)
        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)
        coord = -np.mean([np.linalg.norm(a - action_mean) for a in actions])
        reward = -dist + 0.5 * coord
        done = dist < 0.2
        return self.state, reward, done


def train_team(n_episodes: int = 100, n_steps: int = 200):
    """Train a team and return learning curve."""
    agents = [TrainableAgent(i) for i in range(4)]
    network = Network(4)
    env = Environment(4)

    episode_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        for agent in agents:
            agent.reset_episode()

        total = 0
        for step in range(n_steps):
            observations = [state + np.random.randn(10) * 0.1 for _ in range(4)]
            messages = [agent.create_message(obs) for agent, obs in zip(agents, observations)]
            received = network.exchange(messages)
            actions = [agent.act(obs, msg) for agent, obs, msg in zip(agents, observations, received)]
            state, reward, done = env.step(actions)

            for agent in agents:
                agent.store_reward(reward)
            total += reward

            if done:
                break

        # Update all agents
        for agent in agents:
            agent.update()

        episode_rewards.append(total)

    return agents, episode_rewards


def evaluate_team(agents, n_episodes: int = 20, n_steps: int = 200):
    """Evaluate trained team."""
    network = Network(4)
    env = Environment(4)

    flex_values = []
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        for agent in agents:
            agent.obs_history = []
            agent.action_history = []

        total = 0
        for step in range(n_steps):
            observations = [state + np.random.randn(10) * 0.1 for _ in range(4)]
            messages = [agent.create_message(obs) for agent, obs in zip(agents, observations)]
            received = network.exchange(messages)
            actions = [agent.act(obs, msg) for agent, obs, msg in zip(agents, observations, received)]
            state, reward, done = env.step(actions)
            total += reward
            if done:
                break

        flex = np.mean([agent.get_flexibility() for agent in agents])
        flex_values.append(flex)
        rewards.append(total)

    return np.mean(flex_values), np.mean(rewards)


def main():
    print("\n" + "=" * 70)
    print("PROPER RL TRAINING TEST")
    print("=" * 70)

    n_teams = 40
    n_train_episodes = 100

    # Random teams baseline
    print("\n" + "-" * 70)
    print("RANDOM TEAMS BASELINE")
    print("-" * 70)

    print(f"\nEvaluating {n_teams} random teams...")
    random_data = []
    for i in range(n_teams):
        agents = [TrainableAgent(j) for j in range(4)]
        flex, reward = evaluate_team(agents, n_episodes=10)
        random_data.append((flex, reward))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_teams} complete")

    random_flex = np.array([x[0] for x in random_data])
    random_rew = np.array([x[1] for x in random_data])
    r_random, p_random = stats.pearsonr(random_flex, random_rew)

    sig = '***' if p_random < 0.001 else '**' if p_random < 0.01 else '*' if p_random < 0.05 else ''
    print(f"\nRandom: r = {r_random:+.3f}, p = {p_random:.4f} {sig}")
    print(f"Mean reward: {random_rew.mean():.1f}")

    # Trained teams
    print("\n" + "-" * 70)
    print(f"TRAINED TEAMS ({n_train_episodes} episodes each)")
    print("-" * 70)

    print(f"\nTraining {n_teams} teams...")
    trained_data = []
    learning_curves = []

    for i in range(n_teams):
        agents, curve = train_team(n_episodes=n_train_episodes)
        learning_curves.append(curve)
        flex, reward = evaluate_team(agents, n_episodes=10)
        trained_data.append((flex, reward))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_teams} complete")

    trained_flex = np.array([x[0] for x in trained_data])
    trained_rew = np.array([x[1] for x in trained_data])
    r_trained, p_trained = stats.pearsonr(trained_flex, trained_rew)

    sig = '***' if p_trained < 0.001 else '**' if p_trained < 0.01 else '*' if p_trained < 0.05 else ''
    print(f"\nTrained: r = {r_trained:+.3f}, p = {p_trained:.4f} {sig}")
    print(f"Mean reward: {trained_rew.mean():.1f}")

    # Learning analysis
    print("\n" + "-" * 70)
    print("LEARNING ANALYSIS")
    print("-" * 70)

    curves = np.array(learning_curves)
    early = curves[:, :10].mean(axis=1)
    late = curves[:, -10:].mean(axis=1)
    improvement = late - early

    print(f"\nEarly (ep 1-10):  {early.mean():.1f} ± {early.std():.1f}")
    print(f"Late (ep 91-100): {late.mean():.1f} ± {late.std():.1f}")
    print(f"Improvement:      {improvement.mean():+.1f} ± {improvement.std():.1f}")

    # Did training improve?
    t_stat, t_p = stats.ttest_ind(trained_rew, random_rew)
    d = (trained_rew.mean() - random_rew.mean()) / np.sqrt((trained_rew.std()**2 + random_rew.std()**2) / 2)

    print(f"\nTraining effect: d = {d:+.2f}, t = {t_stat:.2f}, p = {t_p:.4f}")

    # Flexibility predicts learning?
    r_learn, p_learn = stats.pearsonr(trained_flex, improvement)
    print(f"Flexibility ↔ Learning: r = {r_learn:+.3f}, p = {p_learn:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. Flexibility-Reward Correlation:")
    print(f"   Random:  r = {r_random:+.3f}")
    print(f"   Trained: r = {r_trained:+.3f}")

    if abs(r_trained - r_random) < 0.1:
        print("   → Relationship PERSISTS after training")
    elif r_trained > r_random:
        print("   → Training STRENGTHENS relationship")
    else:
        print("   → Training WEAKENS relationship")

    print("\n2. Training Effectiveness:")
    if d > 0.5:
        print(f"   ✓ Large improvement (d = {d:+.2f})")
    elif d > 0.2:
        print(f"   ✓ Moderate improvement (d = {d:+.2f})")
    elif d > 0:
        print(f"   → Small improvement (d = {d:+.2f})")
    else:
        print(f"   ✗ No improvement (d = {d:+.2f})")

    print("\n3. Does flexibility help learning?")
    if r_learn > 0.2 and p_learn < 0.05:
        print(f"   ✓ Yes (r = {r_learn:+.2f}, p = {p_learn:.3f})")
    else:
        print(f"   → No clear relationship (r = {r_learn:+.2f})")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proper_rl_training_{timestamp}.npz"
    np.savez(filename,
             r_random=r_random, r_trained=r_trained,
             random_rew=random_rew, trained_rew=trained_rew,
             improvement_d=d, r_learn=r_learn,
             learning_curves=curves)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
