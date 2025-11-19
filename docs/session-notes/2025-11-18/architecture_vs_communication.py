#!/usr/bin/env python3
"""
Architecture vs Communication Test

Tests whether the effect is due to:
A) Communication (message passing)
B) Architecture (policy structure)
C) Episode length (200 vs 50 steps)
D) Obs/action dimensions (10/10 vs 6/3)

Uses EXACT original architecture with controlled variations.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u architecture_vs_communication.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class OriginalAgent:
    """Exact original architecture."""

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

    def get_flexibility(self) -> float:
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-50:]).flatten()
        actions = np.array(self.action_history[-50:]).flatten()
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        if np.isnan(corr):
            return 0.0
        return -abs(corr) * 2.0


class NullResultAgent:
    """Architecture from definitive_validation.py that produced r ≈ 0."""

    def __init__(self, obs_dim: int = 6, action_dim: int = 3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
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


class CommunicationNetwork:
    def __init__(self, n_agents: int, enabled: bool = True):
        self.n_agents = n_agents
        self.enabled = enabled
        self.adjacency = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def exchange_messages(self, messages):
        received = []
        for i in range(self.n_agents):
            if self.enabled:
                incoming = [messages[j] for j in range(self.n_agents) if self.adjacency[i, j] > 0]
                if incoming:
                    received.append(np.mean(incoming, axis=0))
                else:
                    received.append(np.zeros(5))
            else:
                received.append(np.zeros(5))  # No messages
        return received


class OriginalEnvironment:
    """Original environment with 10D state."""

    def __init__(self, n_agents: int):
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


class NullResultEnvironment:
    """Environment from null-result tests with 6D state."""

    def __init__(self, n_agents: int = 4):
        self.n_agents = n_agents
        self.adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def reset(self):
        self.state = np.random.randn(6) * 0.1
        self.target = np.random.randn(6)
        self.steps = 0
        return [self.state + np.random.randn(6) * 0.2 for _ in range(self.n_agents)]

    def step(self, actions):
        # Pad actions if needed
        padded_actions = []
        for a in actions:
            if len(a) < 6:
                a = np.pad(a, (0, 6 - len(a)))
            padded_actions.append(a[:6])

        action_mean = np.mean(padded_actions, axis=0)
        self.state += action_mean * 0.1
        dist = np.linalg.norm(self.state - self.target)

        alignments = []
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                alignments.append(np.dot(padded_actions[i], padded_actions[j]))
        coord = np.mean(alignments) if alignments else 0

        reward = -dist * 0.5 + coord * 0.5
        self.steps += 1
        done = dist < 0.3 or self.steps >= 50

        return [self.state + np.random.randn(6) * 0.2 for _ in range(self.n_agents)], reward, done


def run_original_architecture(n_agents: int, comm_enabled: bool, n_steps: int, n_episodes: int = 100):
    """Run original architecture with controlled communication."""
    results = []

    for _ in range(n_episodes):
        agents = [OriginalAgent(i) for i in range(n_agents)]
        network = CommunicationNetwork(n_agents, enabled=comm_enabled)
        env = OriginalEnvironment(n_agents)

        state = env.reset()
        for agent in agents:
            agent.obs_history = []
            agent.action_history = []

        total_reward = 0
        for step in range(n_steps):
            observations = [agent.observe(state) for agent in agents]
            messages = [agent.create_message(obs) for agent, obs in zip(agents, observations)]
            received = network.exchange_messages(messages)
            actions = [agent.act(obs, msg) for agent, obs, msg in zip(agents, observations, received)]
            state, reward, done = env.step(actions)
            total_reward += reward
            if done:
                break

        flex = np.mean([agent.get_flexibility() for agent in agents])
        results.append((flex, total_reward))

    flex_arr = np.array([x[0] for x in results])
    rew_arr = np.array([x[1] for x in results])
    r, p = stats.pearsonr(flex_arr, rew_arr)
    return r, p, np.mean(rew_arr)


def run_null_architecture(n_episodes: int = 100):
    """Run null-result architecture."""
    results = []

    for _ in range(n_episodes):
        env = NullResultEnvironment()
        policies = [NullResultAgent() for _ in range(4)]

        obs_list = env.reset()
        for p in policies:
            p.obs_history = []
            p.action_history = []
        total = 0

        for step in range(50):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            obs_list, rew, done = env.step(actions)
            total += rew
            if done:
                break

        flex = np.mean([p.get_flexibility() for p in policies])
        results.append((flex, total))

    flex_arr = np.array([x[0] for x in results])
    rew_arr = np.array([x[1] for x in results])
    r, p = stats.pearsonr(flex_arr, rew_arr)
    return r, p, np.mean(rew_arr)


def main():
    print("\n" + "=" * 70)
    print("ARCHITECTURE VS COMMUNICATION TEST")
    print("=" * 70)

    n_episodes = 200

    # Test 1: Original architecture WITH communication
    print("\n" + "-" * 70)
    print("TEST 1: ORIGINAL ARCHITECTURE + COMMUNICATION")
    print("-" * 70)
    print("(4 agents, 200 steps, 10D obs/action, messages ON)")

    r1, p1, rew1 = run_original_architecture(4, comm_enabled=True, n_steps=200, n_episodes=n_episodes)
    sig = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else ''
    print(f"r = {r1:+.3f}, p = {p1:.4f} {sig}")

    # Test 2: Original architecture WITHOUT communication
    print("\n" + "-" * 70)
    print("TEST 2: ORIGINAL ARCHITECTURE + NO COMMUNICATION")
    print("-" * 70)
    print("(4 agents, 200 steps, 10D obs/action, messages OFF)")

    r2, p2, rew2 = run_original_architecture(4, comm_enabled=False, n_steps=200, n_episodes=n_episodes)
    sig = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
    print(f"r = {r2:+.3f}, p = {p2:.4f} {sig}")

    # Test 3: Original architecture with shorter episodes
    print("\n" + "-" * 70)
    print("TEST 3: ORIGINAL ARCHITECTURE + SHORT EPISODES")
    print("-" * 70)
    print("(4 agents, 50 steps, 10D obs/action, messages ON)")

    r3, p3, rew3 = run_original_architecture(4, comm_enabled=True, n_steps=50, n_episodes=n_episodes)
    sig = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else ''
    print(f"r = {r3:+.3f}, p = {p3:.4f} {sig}")

    # Test 4: Null-result architecture (6D, 50 steps, no messages)
    print("\n" + "-" * 70)
    print("TEST 4: NULL-RESULT ARCHITECTURE")
    print("-" * 70)
    print("(4 agents, 50 steps, 6D obs/3D action, no messages)")

    r4, p4, rew4 = run_null_architecture(n_episodes=n_episodes)
    sig = '***' if p4 < 0.001 else '**' if p4 < 0.01 else '*' if p4 < 0.05 else ''
    print(f"r = {r4:+.3f}, p = {p4:.4f} {sig}")

    # Test 5: Original architecture with 2 agents (strongest condition)
    print("\n" + "-" * 70)
    print("TEST 5: ORIGINAL + 2 AGENTS (Strongest Condition)")
    print("-" * 70)
    print("(2 agents, 200 steps, 10D obs/action, messages ON)")

    r5, p5, rew5 = run_original_architecture(2, comm_enabled=True, n_steps=200, n_episodes=n_episodes)
    sig = '***' if p5 < 0.001 else '**' if p5 < 0.01 else '*' if p5 < 0.05 else ''
    print(f"r = {r5:+.3f}, p = {p5:.4f} {sig}")

    # Test 6: 2 agents WITHOUT communication
    print("\n" + "-" * 70)
    print("TEST 6: 2 AGENTS + NO COMMUNICATION")
    print("-" * 70)
    print("(2 agents, 200 steps, 10D obs/action, messages OFF)")

    r6, p6, rew6 = run_original_architecture(2, comm_enabled=False, n_steps=200, n_episodes=n_episodes)
    sig = '***' if p6 < 0.001 else '**' if p6 < 0.01 else '*' if p6 < 0.05 else ''
    print(f"r = {r6:+.3f}, p = {p6:.4f} {sig}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nComparison Table:")
    print(f"{'Condition':<45} {'r':>8} {'p':>10}")
    print("-" * 65)
    print(f"{'Original + Comm (baseline)':<45} {r1:>+8.3f} {p1:>10.4f}")
    print(f"{'Original + No Comm':<45} {r2:>+8.3f} {p2:>10.4f}")
    print(f"{'Original + Short (50 steps)':<45} {r3:>+8.3f} {p3:>10.4f}")
    print(f"{'Null-result arch (6D/3D, 50 steps)':<45} {r4:>+8.3f} {p4:>10.4f}")
    print(f"{'2 agents + Comm':<45} {r5:>+8.3f} {p5:>10.4f}")
    print(f"{'2 agents + No Comm':<45} {r6:>+8.3f} {p6:>10.4f}")

    # Determine key factor
    print("\n" + "-" * 70)
    print("KEY FACTOR IDENTIFICATION")
    print("-" * 70)

    comm_effect = r1 - r2
    length_effect = r1 - r3
    arch_effect = r1 - r4
    agents_comm = r5 - r6

    print(f"\nCommunication effect (4 agents):   Δr = {comm_effect:+.3f}")
    print(f"Episode length effect:             Δr = {length_effect:+.3f}")
    print(f"Architecture effect:               Δr = {arch_effect:+.3f}")
    print(f"Communication effect (2 agents):   Δr = {agents_comm:+.3f}")

    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    if abs(comm_effect) > 0.15:
        print("\n✓ COMMUNICATION is a key factor")
        print(f"  With comm: r = {r1:+.3f}, Without: r = {r2:+.3f}")
    elif abs(comm_effect) < 0.05:
        print("\n✗ Communication is NOT the key factor")
        print(f"  Effect appears in both conditions")
    else:
        print("\n? Communication has a small effect")
        print(f"  Δr = {comm_effect:+.3f}")

    if abs(arch_effect) > 0.3:
        print("\n✓ ARCHITECTURE is a key factor")
        print(f"  Original: r = {r1:+.3f}, Null-result: r = {r4:+.3f}")
    else:
        print("\n? Architecture has moderate effect")
        print(f"  Δr = {arch_effect:+.3f}")

    if abs(length_effect) > 0.15:
        print("\n✓ EPISODE LENGTH is a key factor")
        print(f"  200 steps: r = {r1:+.3f}, 50 steps: r = {r3:+.3f}")
    else:
        print("\n? Episode length has small effect")
        print(f"  Δr = {length_effect:+.3f}")

    # Final interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if r2 > 0.3 and r4 < 0.1:
        print("\nThe null result was due to ARCHITECTURE, not communication.")
        print("The original architecture produces the effect even without messages.")
        print("The 6D/3D architecture lacks the structure for flexibility to matter.")
    elif r2 < 0.1 and r1 > 0.3:
        print("\nThe null result was due to MISSING COMMUNICATION.")
        print("Communication enables flexibility to predict performance.")
    else:
        print("\nMultiple factors contribute to the effect.")
        print("Both architecture and experimental conditions matter.")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"architecture_vs_communication_{timestamp}.npz"
    np.savez(filename,
             r_orig_comm=r1, r_orig_no_comm=r2, r_short=r3, r_null=r4,
             r_2ag_comm=r5, r_2ag_no_comm=r6)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
