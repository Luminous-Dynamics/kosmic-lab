#!/usr/bin/env python3
"""
Generalization Validation: Test flexibility finding across conditions

Tests:
1. Topologies: fully_connected, ring, star
2. Agent counts: 2, 4, 6, 8

Expected: Flexibility (-K_individual) shows positive correlation across all conditions

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 generalization_validation.py"
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from datetime import datetime


class Agent:
    """Single agent in multi-agent system."""

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

    def create_message(self, obs: np.ndarray) -> np.ndarray:
        return obs[:5]

    def get_k_index(self) -> float:
        """Compute Simple K-Index (obs-action correlation)."""
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-50:]).flatten()
        actions = np.array(self.action_history[-50:]).flatten()
        if len(obs) < 2 or len(actions) < 2:
            return 0.0
        correlation = np.corrcoef(obs, actions)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return abs(correlation) * 2.0


class CommunicationNetwork:
    """Communication topology."""

    def __init__(self, n_agents: int, topology: str = "fully_connected"):
        self.n_agents = n_agents
        self.adjacency = self._build_topology(topology)

    def _build_topology(self, topology: str) -> np.ndarray:
        A = np.zeros((self.n_agents, self.n_agents))
        if topology == "fully_connected":
            A = np.ones((self.n_agents, self.n_agents))
        elif topology == "ring":
            for i in range(self.n_agents):
                A[i, (i + 1) % self.n_agents] = 1
                A[i, (i - 1) % self.n_agents] = 1
        elif topology == "star":
            # Agent 0 is hub
            for i in range(1, self.n_agents):
                A[0, i] = 1
                A[i, 0] = 1
        np.fill_diagonal(A, 0)
        return A

    def exchange_messages(self, messages: List[np.ndarray]) -> List[np.ndarray]:
        received = []
        for i in range(self.n_agents):
            incoming = [messages[j] for j in range(self.n_agents) if self.adjacency[i, j] > 0]
            if incoming:
                received.append(np.mean(incoming, axis=0))
            else:
                received.append(np.zeros(5))
        return received


class MultiAgentEnvironment:
    """Coordination task environment."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.state = np.zeros(10)
        self.target = np.random.randn(10)

    def reset(self):
        self.state = np.random.randn(10) * 0.1
        self.target = np.random.randn(10)
        return self.state

    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, List[float], bool]:
        action_aggregate = np.mean(actions, axis=0)
        self.state += action_aggregate * 0.1

        rewards = []
        for action in actions:
            dist = np.linalg.norm(self.state - self.target)
            coord = -np.linalg.norm(action - action_aggregate)
            rewards.append(-dist + 0.5 * coord)

        done = np.linalg.norm(self.state - self.target) < 0.2
        return self.state, rewards, done


def run_condition(n_agents: int, topology: str, n_episodes: int = 200) -> Dict:
    """Run one experimental condition."""
    results = []

    for ep in range(n_episodes):
        agents = [Agent(i) for i in range(n_agents)]
        network = CommunicationNetwork(n_agents, topology)
        env = MultiAgentEnvironment(n_agents)

        state = env.reset()
        for agent in agents:
            agent.obs_history = []
            agent.action_history = []

        for step in range(200):
            observations = [agent.observe(state) for agent in agents]
            messages = [agent.create_message(obs) for agent, obs in zip(agents, observations)]
            received_messages = network.exchange_messages(messages)
            actions = [agent.act(obs, msg) for agent, obs, msg in zip(agents, observations, received_messages)]
            state, rewards, done = env.step(actions)
            if done:
                break

        # Compute metrics
        individual_k = [agent.get_k_index() for agent in agents]
        mean_individual_k = np.mean(individual_k)
        mean_reward = np.mean(rewards)

        results.append({
            'mean_individual_k': mean_individual_k,
            'mean_reward': mean_reward,
        })

    # Compute correlation
    flexibility = -np.array([r['mean_individual_k'] for r in results])
    rewards = np.array([r['mean_reward'] for r in results])
    r, p = stats.pearsonr(flexibility, rewards)

    # 95% CI
    n = len(rewards)
    z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
    se = 1 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)

    return {
        'n_agents': n_agents,
        'topology': topology,
        'n_episodes': n_episodes,
        'r': r,
        'p': p,
        'ci_low': r_lo,
        'ci_high': r_hi,
        'flexibility': flexibility,
        'rewards': rewards,
    }


def main():
    print("\n" + "=" * 70)
    print("GENERALIZATION VALIDATION")
    print("=" * 70)
    print("\nTesting flexibility finding across conditions...")
    print("=" * 70 + "\n")

    # Define conditions
    conditions = [
        # Topology variations (4 agents)
        (4, "fully_connected"),
        (4, "ring"),
        (4, "star"),
        # Agent count variations (fully connected)
        (2, "fully_connected"),
        (6, "fully_connected"),
        (8, "fully_connected"),
    ]

    all_results = []

    for n_agents, topology in conditions:
        print(f"Testing: {n_agents} agents, {topology} topology...")
        result = run_condition(n_agents, topology, n_episodes=200)
        all_results.append(result)

        sig = '***' if result['p'] < 0.001 else '**' if result['p'] < 0.01 else '*' if result['p'] < 0.05 else ''
        print(f"  r = {result['r']:+.4f}, p = {result['p']:.2e} {sig}")
        print(f"  95% CI: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
        print()

    # Summary table
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Condition':<30} {'r':>8} {'p':>12} {'95% CI':>20} {'Sig':>5}")
    print("-" * 70)

    all_positive = True
    all_significant = True

    for result in all_results:
        condition = f"{result['n_agents']} agents, {result['topology']}"
        ci = f"[{result['ci_low']:.2f}, {result['ci_high']:.2f}]"
        sig = '***' if result['p'] < 0.001 else '**' if result['p'] < 0.01 else '*' if result['p'] < 0.05 else ''
        print(f"{condition:<30} {result['r']:>+8.4f} {result['p']:>12.2e} {ci:>20} {sig:>5}")

        if result['r'] <= 0:
            all_positive = False
        if result['p'] >= 0.05:
            all_significant = False

    # Meta-analysis
    print("\n" + "-" * 70)
    print("META-ANALYSIS")
    print("-" * 70)

    all_flex = np.concatenate([r['flexibility'] for r in all_results])
    all_rewards = np.concatenate([r['rewards'] for r in all_results])
    r_meta, p_meta = stats.pearsonr(all_flex, all_rewards)

    n = len(all_rewards)
    z = 0.5 * np.log((1 + r_meta) / (1 - r_meta))
    se = 1 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)

    print(f"\nCombined (n = {n} episodes across all conditions):")
    print(f"  r = {r_meta:+.4f}, p = {p_meta:.2e}")
    print(f"  95% CI: [{r_lo:.3f}, {r_hi:.3f}]")
    print(f"  R² = {r_meta**2:.1%}")

    # Generalization assessment
    print("\n" + "=" * 70)
    print("GENERALIZATION ASSESSMENT")
    print("=" * 70)

    if all_positive and all_significant:
        print("\n✅ STRONG GENERALIZATION")
        print("   Finding holds across all tested conditions")
        print("   - All correlations positive")
        print("   - All p-values < 0.05")
    elif all_positive:
        print("\n⚠️ PARTIAL GENERALIZATION")
        print("   All correlations positive but some not significant")
    else:
        print("\n❌ LIMITED GENERALIZATION")
        print("   Finding does not hold across all conditions")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generalization_results_{timestamp}.npz"

    save_data = {
        'conditions': np.array([f"{r['n_agents']}_{r['topology']}" for r in all_results]),
        'correlations': np.array([r['r'] for r in all_results]),
        'p_values': np.array([r['p'] for r in all_results]),
        'ci_lows': np.array([r['ci_low'] for r in all_results]),
        'ci_highs': np.array([r['ci_high'] for r in all_results]),
        'meta_r': r_meta,
        'meta_p': p_meta,
        'meta_n': n,
    }
    np.savez(filename, **save_data)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")

    return all_results, r_meta


if __name__ == '__main__':
    main()
