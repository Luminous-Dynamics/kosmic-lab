#!/usr/bin/env python3
"""
Longitudinal Flexibility Tracking (Paper 4)

Tracks flexibility throughout training, not just final.
Shows how developmental curriculum shapes flexibility trajectory.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 longitudinal_flexibility.py"
"""

import numpy as np
from typing import Dict, List
from scipy import stats
from datetime import datetime


class LearningAgent:
    """Agent with trackable learning trajectory."""

    def __init__(self, agent_id: int, obs_dim: int = 10, action_dim: int = 10):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_weights = np.random.randn(action_dim, obs_dim + 5) * 0.1
        self.obs_history = []
        self.action_history = []

    def observe(self, env_state: np.ndarray) -> np.ndarray:
        obs = env_state + np.random.randn(*env_state.shape) * 0.1
        self.obs_history.append(obs)
        return obs

    def act(self, obs: np.ndarray, messages: np.ndarray) -> np.ndarray:
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined)
        self.action_history.append(action)
        return action

    def get_flexibility(self) -> float:
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-50:]).flatten()
        actions = np.array(self.action_history[-50:]).flatten()
        if len(obs) < 2 or len(actions) < 2:
            return 0.0
        min_len = min(len(obs), len(actions))
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        if np.isnan(corr):
            return 0.0
        return -abs(corr) * 2.0

    def update(self, reward: float, learning_rate: float = 0.01):
        noise = np.random.randn(*self.policy_weights.shape)
        self.policy_weights += learning_rate * reward * noise

    def reset_history(self):
        """Clear history but keep weights."""
        self.obs_history = []
        self.action_history = []


def run_training_episode(agents, difficulty: float = 1.0):
    """Run one training episode."""
    n_agents = len(agents)

    A = np.ones((n_agents, n_agents))
    np.fill_diagonal(A, 0)

    state = np.random.randn(10) * 0.1 * difficulty
    target = np.random.randn(10) * difficulty

    for agent in agents:
        agent.reset_history()

    for step in range(200):
        observations = [agent.observe(state) for agent in agents]
        messages = [obs[:5] for obs in observations]

        received = []
        for i in range(n_agents):
            incoming = [messages[j] for j in range(n_agents) if A[i, j] > 0]
            if incoming:
                received.append(np.mean(incoming, axis=0))
            else:
                received.append(np.zeros(5))

        actions = [agent.act(obs, msg) for agent, obs, msg in zip(agents, observations, received)]
        action_agg = np.mean(actions, axis=0)
        state += action_agg * 0.1

        if np.linalg.norm(state - target) < 0.2:
            break

    flexibility = np.mean([agent.get_flexibility() for agent in agents])
    dist = np.linalg.norm(state - target)
    reward = -dist

    return {'flexibility': flexibility, 'reward': reward}


def run_curriculum(curriculum_type: str, n_episodes: int = 500, n_agents: int = 4) -> Dict:
    """Run training with different curricula."""

    agents = [LearningAgent(i) for i in range(n_agents)]

    trajectories = {
        'episode': [],
        'flexibility': [],
        'reward': [],
        'difficulty': []
    }

    for ep in range(n_episodes):
        # Determine difficulty based on curriculum
        if curriculum_type == 'standard':
            difficulty = 1.0
            lr = 0.01
        elif curriculum_type == 'curriculum':
            # Gradually increase difficulty
            difficulty = min(1.0, (ep + 1) / 200)
            lr = 0.01
        elif curriculum_type == 'meta':
            # Adaptive learning rate
            difficulty = 1.0
            lr = 0.01 * (1 + ep / 250)
        elif curriculum_type == 'developmental':
            # Both curriculum and meta
            difficulty = min(1.0, (ep + 1) / 200)
            lr = 0.01 * (1 + ep / 250)
        else:
            difficulty = 1.0
            lr = 0.01

        result = run_training_episode(agents, difficulty)

        # Update agents
        for agent in agents:
            agent.update(result['reward'], lr)

        trajectories['episode'].append(ep)
        trajectories['flexibility'].append(result['flexibility'])
        trajectories['reward'].append(result['reward'])
        trajectories['difficulty'].append(difficulty)

    return trajectories


def analyze_trajectory(trajectories: Dict) -> Dict:
    """Analyze learning trajectory."""
    episodes = np.array(trajectories['episode'])
    flexibility = np.array(trajectories['flexibility'])
    reward = np.array(trajectories['reward'])

    # Overall correlation
    r_overall, p_overall = stats.pearsonr(flexibility, reward)

    # Early vs late training
    mid = len(episodes) // 2
    r_early, p_early = stats.pearsonr(flexibility[:mid], reward[:mid])
    r_late, p_late = stats.pearsonr(flexibility[mid:], reward[mid:])

    # Trend analysis
    slope_flex, _, r_flex, p_flex, _ = stats.linregress(episodes, flexibility)
    slope_reward, _, r_reward, p_reward, _ = stats.linregress(episodes, reward)

    # Final performance (last 50 episodes)
    final_flex = np.mean(flexibility[-50:])
    final_reward = np.mean(reward[-50:])

    return {
        'r_overall': r_overall,
        'p_overall': p_overall,
        'r_early': r_early,
        'r_late': r_late,
        'slope_flex': slope_flex,
        'slope_reward': slope_reward,
        'final_flex': final_flex,
        'final_reward': final_reward,
        'trajectory': trajectories
    }


def main():
    print("\n" + "=" * 70)
    print("LONGITUDINAL FLEXIBILITY TRACKING (Paper 4)")
    print("=" * 70)

    curricula = ['standard', 'curriculum', 'meta', 'developmental']
    n_episodes = 500

    all_results = {}

    for curr_type in curricula:
        print(f"\nTraining with {curr_type} curriculum ({n_episodes} episodes)...")
        trajectories = run_curriculum(curr_type, n_episodes)
        analysis = analyze_trajectory(trajectories)
        all_results[curr_type] = analysis

    # Summary
    print("\n" + "=" * 70)
    print("TRAJECTORY SUMMARY")
    print("=" * 70)

    print(f"\n{'Curriculum':<15} {'r_overall':>10} {'r_early':>10} {'r_late':>10} {'Final Flex':>12}")
    print("-" * 70)

    for curr_type, data in all_results.items():
        print(f"{curr_type:<15} {data['r_overall']:>+10.3f} {data['r_early']:>+10.3f} {data['r_late']:>+10.3f} {data['final_flex']:>+12.3f}")

    # Key finding: Does relationship strengthen over training?
    print("\n" + "-" * 70)
    print("KEY FINDING: Does flex-reward relationship strengthen over training?")
    print("-" * 70)

    for curr_type, data in all_results.items():
        diff = data['r_late'] - data['r_early']
        direction = "strengthens" if diff > 0.1 else "weakens" if diff < -0.1 else "stable"
        print(f"{curr_type}: {direction} (Δr = {diff:+.3f})")

    # Compare developmental to standard
    print("\n" + "-" * 70)
    print("DEVELOPMENTAL vs STANDARD")
    print("-" * 70)

    dev = all_results['developmental']
    std = all_results['standard']

    print(f"Final flexibility: dev = {dev['final_flex']:+.3f}, std = {std['final_flex']:+.3f}")
    print(f"Final reward: dev = {dev['final_reward']:+.3f}, std = {std['final_reward']:+.3f}")
    print(f"Overall r: dev = {dev['r_overall']:+.3f}, std = {std['r_overall']:+.3f}")
    print(f"Late r: dev = {dev['r_late']:+.3f}, std = {std['r_late']:+.3f}")

    if dev['r_late'] > std['r_late']:
        print("\n✓ Developmental curriculum shows stronger late-training flex-reward relationship")
    else:
        print("\n✗ No advantage for developmental curriculum in flex-reward relationship")

    # Flexibility trajectory shape
    print("\n" + "-" * 70)
    print("FLEXIBILITY TRAJECTORY SHAPE")
    print("-" * 70)

    for curr_type, data in all_results.items():
        if data['slope_flex'] > 0.0001:
            shape = "increases"
        elif data['slope_flex'] < -0.0001:
            shape = "decreases"
        else:
            shape = "stable"
        print(f"{curr_type}: flexibility {shape} during training (slope = {data['slope_flex']:.6f})")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"longitudinal_flexibility_{timestamp}.npz"

    save_data = {}
    for curr_type, data in all_results.items():
        traj = data['trajectory']
        save_data[f'{curr_type}_episodes'] = traj['episode']
        save_data[f'{curr_type}_flexibility'] = traj['flexibility']
        save_data[f'{curr_type}_reward'] = traj['reward']

    np.savez(filename, **save_data)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
