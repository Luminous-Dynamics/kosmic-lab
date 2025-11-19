#!/usr/bin/env python3
"""
Feedback Ablation Study (Paper 1)

Tests different feedback types to establish mechanism:
- No feedback (baseline)
- Flexibility-based feedback
- Reward-based feedback
- Entropy-based feedback
- Random feedback (control)

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 feedback_ablation.py"
"""

import numpy as np
from typing import Dict, List
from scipy import stats
from datetime import datetime


class FeedbackAgent:
    """Agent that learns from various feedback types."""

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

    def get_entropy(self) -> float:
        """Action entropy as alternative metric."""
        if len(self.action_history) < 10:
            return 0.0
        actions = np.array(self.action_history[-50:])
        # Approximate entropy via variance
        return np.mean(np.var(actions, axis=0))

    def update(self, feedback: float, learning_rate: float = 0.01):
        noise = np.random.randn(*self.policy_weights.shape)
        self.policy_weights += learning_rate * feedback * noise

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


def run_episode(agents) -> Dict:
    """Run one coordination episode."""
    n_agents = len(agents)

    A = np.ones((n_agents, n_agents))
    np.fill_diagonal(A, 0)

    state = np.random.randn(10) * 0.1
    target = np.random.randn(10)

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
    entropy = np.mean([agent.get_entropy() for agent in agents])
    dist = np.linalg.norm(state - target)
    reward = -dist

    return {
        'flexibility': flexibility,
        'entropy': entropy,
        'reward': reward
    }


def run_feedback_condition(feedback_type: str, n_episodes: int = 200) -> List[Dict]:
    """Run training with specific feedback type."""
    n_agents = 4
    agents = [FeedbackAgent(i) for i in range(n_agents)]

    results = []

    for ep in range(n_episodes):
        result = run_episode(agents)

        # Apply feedback based on type
        for agent in agents:
            if feedback_type == 'none':
                # No update
                pass
            elif feedback_type == 'flexibility':
                # Reward flexibility
                flex = agent.get_flexibility()
                agent.update(flex * 0.1)
            elif feedback_type == 'reward':
                # Standard reward feedback
                agent.update(result['reward'] * 0.01)
            elif feedback_type == 'entropy':
                # Reward entropy
                ent = agent.get_entropy()
                agent.update(ent * 0.1)
            elif feedback_type == 'random':
                # Random feedback (control)
                agent.update(np.random.randn() * 0.1)
            elif feedback_type == 'combined':
                # Flexibility + reward
                flex = agent.get_flexibility()
                agent.update((flex + result['reward'] * 0.1) * 0.05)

        results.append(result)

    return results


def analyze_condition(results: List[Dict]) -> Dict:
    """Analyze results for a feedback condition."""
    flexibility = np.array([r['flexibility'] for r in results])
    entropy = np.array([r['entropy'] for r in results])
    reward = np.array([r['reward'] for r in results])

    # Correlation
    r_flex_reward, p_flex = stats.pearsonr(flexibility, reward)
    r_ent_reward, p_ent = stats.pearsonr(entropy, reward)

    # Final performance (last 50)
    final_flex = np.mean(flexibility[-50:])
    final_reward = np.mean(reward[-50:])

    # Improvement over time
    early_reward = np.mean(reward[:50])
    late_reward = np.mean(reward[-50:])
    improvement = late_reward - early_reward

    return {
        'r_flex_reward': r_flex_reward,
        'p_flex': p_flex,
        'r_ent_reward': r_ent_reward,
        'final_flex': final_flex,
        'final_reward': final_reward,
        'improvement': improvement,
        'mean_reward': reward.mean(),
        'std_reward': reward.std()
    }


def main():
    print("\n" + "=" * 70)
    print("FEEDBACK ABLATION STUDY (Paper 1)")
    print("=" * 70)

    feedback_types = ['none', 'flexibility', 'reward', 'entropy', 'random', 'combined']
    n_episodes = 200

    all_results = {}

    for fb_type in feedback_types:
        print(f"\nTesting {fb_type} feedback...")
        results = run_feedback_condition(fb_type, n_episodes)
        analysis = analyze_condition(results)
        all_results[fb_type] = analysis

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Feedback':<12} {'Mean Reward':>12} {'Final Reward':>12} {'Improvement':>12} {'Final Flex':>12}")
    print("-" * 70)

    for fb_type, data in all_results.items():
        print(f"{fb_type:<12} {data['mean_reward']:>+12.3f} {data['final_reward']:>+12.3f} {data['improvement']:>+12.3f} {data['final_flex']:>+12.3f}")

    # Statistical comparisons
    print("\n" + "-" * 70)
    print("STATISTICAL COMPARISONS (vs no feedback)")
    print("-" * 70)

    baseline = all_results['none']

    for fb_type, data in all_results.items():
        if fb_type == 'none':
            continue

        # Effect size
        pooled_std = np.sqrt((data['std_reward']**2 + baseline['std_reward']**2) / 2)
        if pooled_std > 0:
            d = (data['mean_reward'] - baseline['mean_reward']) / pooled_std
        else:
            d = 0

        direction = "better" if d > 0 else "worse"
        print(f"{fb_type}: d = {d:+.2f} ({direction} than baseline)")

    # Key finding
    print("\n" + "-" * 70)
    print("KEY FINDING: Which feedback type works best?")
    print("-" * 70)

    best = max(all_results.items(), key=lambda x: x[1]['final_reward'])
    print(f"\nBest feedback type: {best[0]}")
    print(f"  Final reward: {best[1]['final_reward']:+.3f}")
    print(f"  Final flexibility: {best[1]['final_flex']:+.3f}")
    print(f"  Improvement: {best[1]['improvement']:+.3f}")

    # Flexibility feedback specifically
    print("\n" + "-" * 70)
    print("FLEXIBILITY FEEDBACK ANALYSIS")
    print("-" * 70)

    flex_fb = all_results['flexibility']
    none_fb = all_results['none']
    reward_fb = all_results['reward']

    print(f"\nFlexibility vs None:")
    print(f"  Final reward: {flex_fb['final_reward']:+.3f} vs {none_fb['final_reward']:+.3f}")
    print(f"  Final flexibility: {flex_fb['final_flex']:+.3f} vs {none_fb['final_flex']:+.3f}")

    print(f"\nFlexibility vs Reward:")
    print(f"  Final reward: {flex_fb['final_reward']:+.3f} vs {reward_fb['final_reward']:+.3f}")
    print(f"  Final flexibility: {flex_fb['final_flex']:+.3f} vs {reward_fb['final_flex']:+.3f}")

    if flex_fb['final_reward'] > none_fb['final_reward']:
        print("\n✓ Flexibility feedback improves coordination")
    else:
        print("\n✗ Flexibility feedback does not improve coordination")

    if flex_fb['final_flex'] > none_fb['final_flex']:
        print("✓ Flexibility feedback increases flexibility")
    else:
        print("✗ Flexibility feedback does not increase flexibility")

    # Random control check
    print("\n" + "-" * 70)
    print("CONTROL CHECK: Random feedback")
    print("-" * 70)

    random_fb = all_results['random']
    if random_fb['final_reward'] < none_fb['final_reward']:
        print("✓ Random feedback hurts performance (as expected)")
    else:
        print("⚠️ Random feedback doesn't hurt - may need more episodes")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feedback_ablation_{timestamp}.npz"

    save_data = {}
    for fb_type, data in all_results.items():
        for key, value in data.items():
            save_data[f"{fb_type}_{key}"] = value

    np.savez(filename, **save_data)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
