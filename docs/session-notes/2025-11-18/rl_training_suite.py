#!/usr/bin/env python3
"""
Full RL Training Suite with Flexibility Tracking

Implements proper reinforcement learning to validate flexibility patterns
with trained agents rather than random policies.

Algorithms:
- REINFORCE for single-agent
- Independent REINFORCE for multi-agent
- Centralized critic for coordination

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 rl_training_suite.py"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime
import json


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class MLP:
    """Simple multi-layer perceptron with backprop."""

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        self.layers = []
        self.lr = learning_rate

        # Initialize weights with Xavier initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.layers.append({'w': w, 'b': b})

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass with activation caching."""
        activations = [x]
        for i, layer in enumerate(self.layers):
            x = x @ layer['w'] + layer['b']
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)  # ReLU
            activations.append(x)
        return x, activations

    def backward(self, activations: List, grad_output: np.ndarray):
        """Backward pass to compute gradients."""
        grads = []
        grad = grad_output

        for i in reversed(range(len(self.layers))):
            # Gradient for weights and biases
            grad_w = activations[i].reshape(-1, 1) @ grad.reshape(1, -1)
            grad_b = grad

            grads.insert(0, {'w': grad_w, 'b': grad_b})

            if i > 0:
                # Backprop through layer
                grad = (self.layers[i]['w'] @ grad.reshape(-1, 1)).flatten()
                # ReLU derivative
                grad = grad * (activations[i] > 0)

        return grads

    def update(self, grads: List, scale: float = 1.0):
        """Apply gradients with scaling."""
        for i, (layer, grad) in enumerate(zip(self.layers, grads)):
            layer['w'] += self.lr * scale * grad['w']
            layer['b'] += self.lr * scale * grad['b']


# =============================================================================
# POLICY NETWORKS
# =============================================================================

class CategoricalPolicy:
    """Policy for discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        self.network = MLP([obs_dim, hidden_dim, hidden_dim, action_dim], learning_rate=0.001)
        self.action_dim = action_dim

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Sample action and return log probability."""
        logits, _ = self.network.forward(obs)
        # Softmax
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()

        action = np.random.choice(self.action_dim, p=probs)
        log_prob = np.log(probs[action])

        return action, log_prob

    def update(self, obs: np.ndarray, action: int, advantage: float):
        """Policy gradient update."""
        logits, activations = self.network.forward(obs)
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()

        # Gradient of log probability
        grad = -probs.copy()
        grad[action] += 1

        # Scale by advantage
        self.network.backward(activations, grad)
        grads = self.network.backward(activations, grad)
        self.network.update(grads, advantage)


class GaussianPolicy:
    """Policy for continuous action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        self.mean_network = MLP([obs_dim, hidden_dim, hidden_dim, action_dim], learning_rate=0.0003)
        self.log_std = np.zeros(action_dim) - 0.5  # Start with smaller std
        self.action_dim = action_dim

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Sample action and return log probability."""
        mean, _ = self.mean_network.forward(obs)
        mean = np.clip(mean, -5, 5)  # Clip mean
        std = np.exp(np.clip(self.log_std, -5, 2))  # Clip log_std

        action = mean + std * np.random.randn(self.action_dim)
        action = np.clip(action, -1, 1)

        # Log probability of Gaussian
        log_prob = -0.5 * np.sum(((action - mean) / (std + 1e-8)) ** 2 + 2 * self.log_std + np.log(2 * np.pi))

        return action, log_prob

    def update(self, obs: np.ndarray, action: np.ndarray, advantage: float):
        """Policy gradient update."""
        mean, activations = self.mean_network.forward(obs)
        mean = np.clip(mean, -5, 5)
        std = np.exp(np.clip(self.log_std, -5, 2))

        # Gradient of log probability w.r.t. mean with numerical stability
        grad_mean = (action - mean) / (std ** 2 + 1e-8)
        grad_mean = np.clip(grad_mean, -10, 10)  # Clip gradients

        grads = self.mean_network.backward(activations, grad_mean)
        # Clip gradient updates
        for g in grads:
            g['w'] = np.clip(g['w'], -1, 1)
            g['b'] = np.clip(g['b'], -1, 1)
        self.mean_network.update(grads, np.clip(advantage, -10, 10))

        # Update log_std with clipping
        grad_log_std = ((action - mean) ** 2 / (std ** 2 + 1e-8) - 1)
        grad_log_std = np.clip(grad_log_std, -10, 10)
        self.log_std += 0.0003 * np.clip(advantage, -10, 10) * grad_log_std
        self.log_std = np.clip(self.log_std, -5, 2)


class ValueNetwork:
    """Value function for variance reduction."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        self.network = MLP([obs_dim, hidden_dim, hidden_dim, 1], learning_rate=0.001)

    def predict(self, obs: np.ndarray) -> float:
        """Predict state value."""
        value, _ = self.network.forward(obs)
        return value[0]

    def update(self, obs: np.ndarray, target: float):
        """Update value function."""
        value, activations = self.network.forward(obs)
        error = target - value[0]

        grads = self.network.backward(activations, np.array([error]))
        self.network.update(grads, 1.0)


# =============================================================================
# FLEXIBILITY TRACKER
# =============================================================================

class FlexibilityTracker:
    """Track flexibility metrics during training."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.obs_history = []
        self.action_history = []

    def record(self, obs: np.ndarray, action):
        """Record observation-action pair."""
        self.obs_history.append(obs.flatten())
        if isinstance(action, (int, np.integer)):
            self.action_history.append(np.array([float(action)]))
        else:
            self.action_history.append(np.array(action).flatten())

    def get_flexibility(self) -> float:
        """Calculate flexibility from history."""
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
        """Clear history."""
        self.obs_history = []
        self.action_history = []


# =============================================================================
# ENVIRONMENTS
# =============================================================================

class CoordinationEnv:
    """Multi-agent coordination environment."""

    def __init__(self, n_agents: int = 4, obs_dim: int = 10, action_dim: int = 5):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        """Reset environment."""
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.target = np.random.randn(self.obs_dim)
        self.step_count = 0
        return [self._get_obs(i) for i in range(self.n_agents)]

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """Get observation for agent."""
        obs = self.state + np.random.randn(self.obs_dim) * 0.1
        return obs

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool]:
        """Take step in environment."""
        # Aggregate actions and pad/truncate to match state dim
        action_mean = np.mean(actions, axis=0)
        if len(action_mean) < self.obs_dim:
            action_mean = np.pad(action_mean, (0, self.obs_dim - len(action_mean)))
        else:
            action_mean = action_mean[:self.obs_dim]
        self.state += action_mean * 0.1

        # Compute reward (shared)
        dist = np.linalg.norm(self.state - self.target)
        reward = -dist

        # Coordination bonus
        action_std = np.std(actions, axis=0).mean()
        coordination_bonus = -action_std * 0.1
        reward += coordination_bonus

        self.step_count += 1
        done = dist < 0.5 or self.step_count >= 200

        obs = [self._get_obs(i) for i in range(self.n_agents)]
        rewards = [reward] * self.n_agents

        return obs, rewards, done


class SingleAgentEnv:
    """Simple single-agent control task."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.target = np.zeros(self.obs_dim)
        self.step_count = 0
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take step."""
        # Map discrete action to continuous
        if action == 0:
            delta = np.array([0.1, 0, 0, 0])
        elif action == 1:
            delta = np.array([-0.1, 0, 0, 0])
        else:
            delta = np.zeros(self.obs_dim)

        self.state += delta + np.random.randn(self.obs_dim) * 0.01

        dist = np.linalg.norm(self.state - self.target)
        reward = -dist

        self.step_count += 1
        done = self.step_count >= 200

        return self.state.copy(), reward, done


# =============================================================================
# TRAINING ALGORITHMS
# =============================================================================

def train_single_agent(n_episodes: int = 1000, track_every: int = 10) -> Dict:
    """Train single agent with REINFORCE."""
    env = SingleAgentEnv()
    policy = CategoricalPolicy(env.obs_dim, env.action_dim)
    value_fn = ValueNetwork(env.obs_dim)
    tracker = FlexibilityTracker()

    training_data = {
        'episode': [],
        'reward': [],
        'flexibility': [],
        'value_loss': []
    }

    for ep in range(n_episodes):
        obs = env.reset()
        tracker.reset()

        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []

        done = False
        while not done:
            action, log_prob = policy.get_action(obs)
            tracker.record(obs, action)

            next_obs, reward, done = env.step(action)

            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)

            obs = next_obs

        # Compute returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy and value function
        for obs, action, G in zip(episode_obs, episode_actions, returns):
            value = value_fn.predict(obs)
            advantage = G - value

            policy.update(obs, action, advantage)
            value_fn.update(obs, G)

        # Track metrics
        if ep % track_every == 0:
            training_data['episode'].append(ep)
            training_data['reward'].append(sum(episode_rewards))
            training_data['flexibility'].append(tracker.get_flexibility())

    return training_data


def train_multi_agent(n_episodes: int = 1000, n_agents: int = 4, track_every: int = 10) -> Dict:
    """Train multi-agent system with independent REINFORCE."""
    env = CoordinationEnv(n_agents=n_agents)

    # Each agent has its own policy
    policies = [GaussianPolicy(env.obs_dim, env.action_dim) for _ in range(n_agents)]
    value_fns = [ValueNetwork(env.obs_dim) for _ in range(n_agents)]
    trackers = [FlexibilityTracker() for _ in range(n_agents)]

    training_data = {
        'episode': [],
        'reward': [],
        'flexibility': [],
        'coordination': []
    }

    for ep in range(n_episodes):
        obs_list = env.reset()
        for tracker in trackers:
            tracker.reset()

        episode_data = [{'obs': [], 'actions': [], 'rewards': [], 'log_probs': []}
                       for _ in range(n_agents)]

        done = False
        while not done:
            actions = []
            for i, (policy, obs) in enumerate(zip(policies, obs_list)):
                action, log_prob = policy.get_action(obs)
                trackers[i].record(obs, action)
                actions.append(action)
                episode_data[i]['obs'].append(obs)
                episode_data[i]['actions'].append(action)
                episode_data[i]['log_probs'].append(log_prob)

            obs_list, rewards, done = env.step(actions)

            for i, r in enumerate(rewards):
                episode_data[i]['rewards'].append(r)

        # Update each agent
        for i in range(n_agents):
            data = episode_data[i]

            # Compute returns
            returns = []
            G = 0
            for r in reversed(data['rewards']):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = np.array(returns)

            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Update
            for obs, action, G in zip(data['obs'], data['actions'], returns):
                value = value_fns[i].predict(obs)
                advantage = G - value

                policies[i].update(obs, action, advantage)
                value_fns[i].update(obs, G)

        # Track metrics
        if ep % track_every == 0:
            mean_reward = np.mean([sum(d['rewards']) for d in episode_data])
            mean_flex = np.mean([t.get_flexibility() for t in trackers])

            # Coordination metric
            all_actions = [d['actions'] for d in episode_data]
            if all_actions[0]:
                coord = 1 - np.mean([np.std([a[j] for a in all_actions], axis=0).mean()
                                    for j in range(len(all_actions[0]))])
            else:
                coord = 0

            training_data['episode'].append(ep)
            training_data['reward'].append(mean_reward)
            training_data['flexibility'].append(mean_flex)
            training_data['coordination'].append(coord)

    return training_data


def compare_trained_vs_random(n_test_episodes: int = 200) -> Dict:
    """Compare trained agents to random policies."""

    print("\nTraining multi-agent system (500 episodes)...")
    training_data = train_multi_agent(n_episodes=500, track_every=50)

    # Final trained performance
    trained_flex = training_data['flexibility'][-1]
    trained_reward = training_data['reward'][-1]

    # Random policy baseline
    print("Testing random policies...")
    env = CoordinationEnv(n_agents=4)
    random_results = []

    for _ in range(n_test_episodes):
        obs_list = env.reset()
        trackers = [FlexibilityTracker() for _ in range(4)]
        total_reward = 0

        for step in range(200):
            actions = [np.random.randn(5) * 0.5 for _ in range(4)]
            for i, (obs, action) in enumerate(zip(obs_list, actions)):
                trackers[i].record(obs, action)

            obs_list, rewards, done = env.step(actions)
            total_reward += rewards[0]

            if done:
                break

        mean_flex = np.mean([t.get_flexibility() for t in trackers])
        random_results.append({'flexibility': mean_flex, 'reward': total_reward})

    random_flex = np.mean([r['flexibility'] for r in random_results])
    random_reward = np.mean([r['reward'] for r in random_results])

    return {
        'training_data': training_data,
        'trained_flex': trained_flex,
        'trained_reward': trained_reward,
        'random_flex': random_flex,
        'random_reward': random_reward
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("FULL RL TRAINING SUITE WITH FLEXIBILITY TRACKING")
    print("=" * 70)

    # 1. Single-agent training
    print("\n[1/4] Training single-agent (1000 episodes)...")
    single_data = train_single_agent(n_episodes=1000, track_every=50)

    # Analyze single-agent
    flexibility = np.array(single_data['flexibility'])
    rewards = np.array(single_data['reward'])
    r_single, p_single = stats.pearsonr(flexibility, rewards)

    print(f"  Single-agent flex-reward: r = {r_single:+.3f}, p = {p_single:.2e}")

    # 2. Multi-agent training
    print("\n[2/4] Training multi-agent (1000 episodes)...")
    multi_data = train_multi_agent(n_episodes=1000, track_every=50)

    # Analyze multi-agent
    flexibility = np.array(multi_data['flexibility'])
    rewards = np.array(multi_data['reward'])
    r_multi, p_multi = stats.pearsonr(flexibility, rewards)

    print(f"  Multi-agent flex-reward: r = {r_multi:+.3f}, p = {p_multi:.2e}")

    # 3. Compare trained vs random
    print("\n[3/4] Comparing trained vs random policies...")
    comparison = compare_trained_vs_random()

    print(f"  Trained: flex = {comparison['trained_flex']:+.3f}, reward = {comparison['trained_reward']:+.3f}")
    print(f"  Random:  flex = {comparison['random_flex']:+.3f}, reward = {comparison['random_reward']:+.3f}")

    # 4. Flexibility-coordination analysis
    print("\n[4/4] Analyzing flexibility-coordination relationship...")
    coordination = np.array(multi_data['coordination'])
    flexibility = np.array(multi_data['flexibility'])
    r_coord, p_coord = stats.pearsonr(flexibility, coordination)

    print(f"  Flex-coordination: r = {r_coord:+.3f}, p = {p_coord:.2e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Condition':<25} {'r':>10} {'p':>15} {'Interpretation'}")
    print("-" * 70)

    sig_single = '***' if p_single < 0.001 else '**' if p_single < 0.01 else '*' if p_single < 0.05 else 'ns'
    sig_multi = '***' if p_multi < 0.001 else '**' if p_multi < 0.01 else '*' if p_multi < 0.05 else 'ns'
    sig_coord = '***' if p_coord < 0.001 else '**' if p_coord < 0.01 else '*' if p_coord < 0.05 else 'ns'

    print(f"{'Single-agent':<25} {r_single:>+10.3f} {p_single:>15.2e} {sig_single}")
    print(f"{'Multi-agent':<25} {r_multi:>+10.3f} {p_multi:>15.2e} {sig_multi}")
    print(f"{'Flex-Coordination':<25} {r_coord:>+10.3f} {p_coord:>15.2e} {sig_coord}")

    # Key finding
    print("\n" + "-" * 70)
    if r_multi > r_single + 0.1:
        print("✓ CONFIRMED: Flexibility-reward relationship stronger in multi-agent")
    elif r_single > r_multi + 0.1:
        print("⚠️ UNEXPECTED: Single-agent shows stronger relationship")
    else:
        print("→ Similar relationship strength in both conditions")

    # Training dynamics
    print("\n" + "-" * 70)
    print("TRAINING DYNAMICS")
    print("-" * 70)

    # Did flexibility increase during training?
    early_flex = np.mean(multi_data['flexibility'][:5])
    late_flex = np.mean(multi_data['flexibility'][-5:])
    flex_change = late_flex - early_flex

    early_reward = np.mean(multi_data['reward'][:5])
    late_reward = np.mean(multi_data['reward'][-5:])
    reward_change = late_reward - early_reward

    print(f"Flexibility: {early_flex:+.3f} → {late_flex:+.3f} (Δ = {flex_change:+.3f})")
    print(f"Reward: {early_reward:+.3f} → {late_reward:+.3f} (Δ = {reward_change:+.3f})")

    if flex_change > 0 and reward_change > 0:
        print("\n✓ Both flexibility and reward increased during training")
    elif flex_change < 0 and reward_change > 0:
        print("\n→ Reward increased but flexibility decreased (agents became more rigid)")
    else:
        print(f"\n→ Mixed pattern: flex Δ={flex_change:+.3f}, reward Δ={reward_change:+.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rl_training_results_{timestamp}.npz"

    np.savez(filename,
             single_episodes=single_data['episode'],
             single_rewards=single_data['reward'],
             single_flexibility=single_data['flexibility'],
             multi_episodes=multi_data['episode'],
             multi_rewards=multi_data['reward'],
             multi_flexibility=multi_data['flexibility'],
             multi_coordination=multi_data['coordination'],
             r_single=r_single,
             r_multi=r_multi,
             r_coord=r_coord)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")

    return {
        'single': single_data,
        'multi': multi_data,
        'comparison': comparison,
        'r_single': r_single,
        'r_multi': r_multi,
        'r_coord': r_coord
    }


if __name__ == '__main__':
    main()
