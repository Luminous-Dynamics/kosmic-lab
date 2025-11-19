#!/usr/bin/env python3
"""
Adversarial Attack Variations (Paper 5)

Tests multiple attack types to establish thorough null result:
- Weight perturbation
- Observation noise
- Action noise
- Message corruption

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 adversarial_variations.py"
"""

import numpy as np
from typing import Dict, List
from scipy import stats
from datetime import datetime


class RobustAgent:
    """Agent for adversarial testing."""

    def __init__(self, agent_id: int, obs_dim: int = 10, action_dim: int = 10):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_weights = np.random.randn(action_dim, obs_dim + 5) * 0.1
        self.obs_history = []
        self.action_history = []

    def observe(self, env_state: np.ndarray, obs_noise: float = 0.0) -> np.ndarray:
        obs = env_state + np.random.randn(*env_state.shape) * 0.1
        # Add adversarial observation noise
        obs += np.random.randn(*env_state.shape) * obs_noise
        self.obs_history.append(obs)
        return obs

    def act(self, obs: np.ndarray, messages: np.ndarray, action_noise: float = 0.0) -> np.ndarray:
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined)
        # Add adversarial action noise
        action += np.random.randn(*action.shape) * action_noise
        action = np.clip(action, -1, 1)
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

    def perturb_weights(self, noise_level: float):
        """Apply weight perturbation attack."""
        self.policy_weights += np.random.randn(*self.policy_weights.shape) * noise_level

    def reset(self):
        self.obs_history = []
        self.action_history = []


def run_episode(agents, obs_noise: float = 0.0, action_noise: float = 0.0,
                message_noise: float = 0.0) -> Dict:
    """Run episode with various noise types."""
    n_agents = len(agents)

    A = np.ones((n_agents, n_agents))
    np.fill_diagonal(A, 0)

    state = np.random.randn(10) * 0.1
    target = np.random.randn(10)

    for agent in agents:
        agent.reset()

    for step in range(200):
        observations = [agent.observe(state, obs_noise) for agent in agents]
        messages = [obs[:5] for obs in observations]

        # Apply message corruption
        if message_noise > 0:
            messages = [m + np.random.randn(5) * message_noise for m in messages]

        received = []
        for i in range(n_agents):
            incoming = [messages[j] for j in range(n_agents) if A[i, j] > 0]
            if incoming:
                received.append(np.mean(incoming, axis=0))
            else:
                received.append(np.zeros(5))

        actions = [agent.act(obs, msg, action_noise)
                  for agent, obs, msg in zip(agents, observations, received)]
        action_agg = np.mean(actions, axis=0)
        state += action_agg * 0.1

        if np.linalg.norm(state - target) < 0.2:
            break

    flexibility = np.mean([agent.get_flexibility() for agent in agents])
    dist = np.linalg.norm(state - target)
    reward = -dist

    return {'flexibility': flexibility, 'reward': reward}


def test_attack_type(attack_type: str, noise_levels: List[float], n_episodes: int = 200) -> Dict:
    """Test a specific attack type."""
    n_agents = 4
    results = {}

    for noise in noise_levels:
        episode_results = []

        for ep in range(n_episodes):
            agents = [RobustAgent(i) for i in range(n_agents)]

            # Get clean flexibility first
            clean_result = run_episode(agents)
            clean_flex = clean_result['flexibility']

            # Apply attack
            if attack_type == 'weight':
                for agent in agents:
                    agent.perturb_weights(noise)
                result = run_episode(agents)
            elif attack_type == 'observation':
                result = run_episode(agents, obs_noise=noise)
            elif attack_type == 'action':
                result = run_episode(agents, action_noise=noise)
            elif attack_type == 'message':
                result = run_episode(agents, message_noise=noise)
            else:
                result = run_episode(agents)

            episode_results.append({
                'clean_flex': clean_flex,
                'perturbed_flex': result['flexibility'],
                'reward': result['reward']
            })

        results[noise] = episode_results

    return results


def analyze_attack_results(results: Dict) -> Dict:
    """Analyze if flexibility predicts robustness."""
    analysis = {}

    for noise, episodes in results.items():
        clean_flex = np.array([e['clean_flex'] for e in episodes])
        reward = np.array([e['reward'] for e in episodes])

        r, p = stats.pearsonr(clean_flex, reward)
        analysis[noise] = {'r': r, 'p': p, 'n': len(episodes)}

    return analysis


def main():
    print("\n" + "=" * 70)
    print("ADVERSARIAL ATTACK VARIATIONS (Paper 5)")
    print("=" * 70)

    attack_types = {
        'weight': [0.1, 0.3, 0.5, 1.0],
        'observation': [0.1, 0.3, 0.5, 1.0],
        'action': [0.1, 0.3, 0.5, 1.0],
        'message': [0.1, 0.3, 0.5, 1.0]
    }

    all_results = {}

    for attack_type, noise_levels in attack_types.items():
        print(f"\nTesting {attack_type} perturbation...")
        results = test_attack_type(attack_type, noise_levels)
        analysis = analyze_attack_results(results)
        all_results[attack_type] = analysis

        for noise, data in analysis.items():
            sig = '***' if data['p'] < 0.001 else '**' if data['p'] < 0.01 else '*' if data['p'] < 0.05 else 'ns'
            print(f"  noise={noise}: r = {data['r']:+.3f}, p = {data['p']:.2e} {sig}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Does Flexibility Predict Robustness?")
    print("=" * 70)

    print(f"\n{'Attack Type':<15} {'Noise':>8} {'r':>10} {'p':>12} {'Sig':>6}")
    print("-" * 70)

    significant_count = 0
    total_tests = 0

    for attack_type, analysis in all_results.items():
        for noise, data in analysis.items():
            sig = '***' if data['p'] < 0.001 else '**' if data['p'] < 0.01 else '*' if data['p'] < 0.05 else 'ns'
            print(f"{attack_type:<15} {noise:>8.1f} {data['r']:>+10.3f} {data['p']:>12.2e} {sig:>6}")

            total_tests += 1
            if data['p'] < 0.05:
                significant_count += 1

    # Conclusion
    print("\n" + "-" * 70)
    print("CONCLUSION")
    print("-" * 70)

    print(f"\nSignificant results: {significant_count}/{total_tests} ({100*significant_count/total_tests:.0f}%)")

    if significant_count < total_tests * 0.1:
        print("\n✓ CONFIRMED NULL RESULT: Flexibility does NOT predict adversarial robustness")
        print("  across ANY attack type (weight, observation, action, message)")
        print("\n  This is scientifically valuable - it defines the boundaries of flexibility's")
        print("  predictive power. Flexibility predicts coordination under normal conditions,")
        print("  but robustness under attack requires different mechanisms.")
    elif significant_count < total_tests * 0.3:
        print("\n⚠️ WEAK RELATIONSHIP: Some isolated significant results")
        print("  Further investigation needed to determine if pattern is reliable")
    else:
        print("\n❌ UNEXPECTED: Flexibility appears to predict robustness")
        print("  This contradicts previous findings - investigate further")

    # Attack-specific summary
    print("\n" + "-" * 70)
    print("ATTACK-SPECIFIC SUMMARY")
    print("-" * 70)

    for attack_type, analysis in all_results.items():
        mean_r = np.mean([d['r'] for d in analysis.values()])
        sig_count = sum(1 for d in analysis.values() if d['p'] < 0.05)
        print(f"{attack_type}: mean r = {mean_r:+.3f}, significant = {sig_count}/{len(analysis)}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"adversarial_variations_{timestamp}.npz"

    save_data = {}
    for attack_type, analysis in all_results.items():
        for noise, data in analysis.items():
            key = f"{attack_type}_{str(noise).replace('.', '_')}"
            save_data[f"{key}_r"] = data['r']
            save_data[f"{key}_p"] = data['p']

    np.savez(filename, **save_data)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
