#!/usr/bin/env python3
"""
Original Conditions Replication

Replicates the EXACT original experimental setup that produced r = +0.74.

Key features from original:
1. Communication network with message passing
2. obs_dim=10, action_dim=10
3. 200 steps per episode
4. Multiple topologies and agent counts
5. Meta-analysis across conditions

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u original_conditions_replication.py"
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
        min_len = min(len(obs), len(actions))
        correlation = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
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

    # Compute correlation (flexibility = -K)
    flexibility = -np.array([r['mean_individual_k'] for r in results])
    rewards = np.array([r['mean_reward'] for r in results])
    r, p = stats.pearsonr(flexibility, rewards)

    return {
        'n_agents': n_agents,
        'topology': topology,
        'n_episodes': n_episodes,
        'r': r,
        'p': p,
        'flexibility': flexibility,
        'rewards': rewards,
    }


def main():
    print("\n" + "=" * 70)
    print("ORIGINAL CONDITIONS REPLICATION")
    print("=" * 70)
    print("\nReplicating exact conditions that produced r = +0.74")
    print("=" * 70 + "\n")

    # Original conditions
    conditions = [
        (4, "fully_connected"),
        (4, "ring"),
        (4, "star"),
        (2, "fully_connected"),
        (6, "fully_connected"),
        (8, "fully_connected"),
    ]

    all_results = []

    for n_agents, topology in conditions:
        print(f"Testing: {n_agents} agents, {topology}...")
        result = run_condition(n_agents, topology, n_episodes=200)
        all_results.append(result)

        sig = '***' if result['p'] < 0.001 else '**' if result['p'] < 0.01 else '*' if result['p'] < 0.05 else ''
        print(f"  r = {result['r']:+.4f}, p = {result['p']:.4f} {sig}")

    # Meta-analysis (combining all conditions - this is what produced r = +0.74)
    print("\n" + "=" * 70)
    print("META-ANALYSIS (COMBINED)")
    print("=" * 70)

    all_flex = np.concatenate([r['flexibility'] for r in all_results])
    all_rewards = np.concatenate([r['rewards'] for r in all_results])
    r_meta, p_meta = stats.pearsonr(all_flex, all_rewards)

    n = len(all_rewards)

    # Bootstrap CI
    boot_r = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        boot_r.append(stats.pearsonr(all_flex[idx], all_rewards[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    sig = '***' if p_meta < 0.001 else '**' if p_meta < 0.01 else '*' if p_meta < 0.05 else ''
    print(f"\nCombined (n = {n}):")
    print(f"  r = {r_meta:+.4f}, p = {p_meta:.2e} {sig}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # Compare to original
    print("\n" + "-" * 70)
    print("COMPARISON TO ORIGINAL")
    print("-" * 70)

    original_r = 0.74
    diff = r_meta - original_r

    if abs(diff) < 0.1:
        print(f"\n✓ REPLICATED: r = {r_meta:+.3f} (original: {original_r})")
        print("  Effect successfully recovered!")
    elif r_meta > 0.5:
        print(f"\n⚠️ PARTIAL: r = {r_meta:+.3f} (original: {original_r})")
        print("  Strong effect but not as large as original")
    elif r_meta > 0.3:
        print(f"\n⚠️ WEAKER: r = {r_meta:+.3f} (original: {original_r})")
        print("  Moderate effect, smaller than original")
    elif r_meta > 0:
        print(f"\n→ MUCH WEAKER: r = {r_meta:+.3f} (original: {original_r})")
        print("  Effect present but much smaller")
    else:
        print(f"\n✗ NOT REPLICATED: r = {r_meta:+.3f} (original: {original_r})")
        print("  No positive correlation found")

    # Per-condition summary
    print("\n" + "-" * 70)
    print("PER-CONDITION RESULTS")
    print("-" * 70)

    print(f"\n{'Condition':<25} {'r':>10} {'p':>12}")
    print("-" * 50)
    for result in all_results:
        condition = f"{result['n_agents']} agents, {result['topology']}"
        print(f"{condition:<25} {result['r']:>+10.3f} {result['p']:>12.4f}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)

    if r_meta > 0.5:
        print("\nThe original r = +0.74 was due to:")
        print("  1. Communication network with message passing")
        print("  2. Combining multiple conditions (Simpson's paradox)")
        print("  3. Longer episodes (200 steps)")
        print("\nOur null result was because we lacked these features.")
    else:
        print("\nEven with original conditions, effect is smaller.")
        print("Possible reasons:")
        print("  1. Stochastic variation")
        print("  2. Original had additional features")
        print("  3. Effect is genuinely unstable")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"original_replication_{timestamp}.npz"
    np.savez(filename,
             all_flex=all_flex, all_rewards=all_rewards,
             r_meta=r_meta, p_meta=p_meta,
             per_condition_r=[r['r'] for r in all_results])
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
