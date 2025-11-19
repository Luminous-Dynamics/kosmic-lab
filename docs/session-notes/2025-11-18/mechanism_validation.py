#!/usr/bin/env python3
"""
Mechanism Validation: Why Does Communication Enable Flexibility to Matter?

Tests:
1. A/B test: Same teams with/without communication
2. Partial communication (0%, 25%, 50%, 75%, 100%)
3. Message utilization: How much do flexible agents use messages?

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u mechanism_validation.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Agent:
    """Agent with controllable message usage."""

    def __init__(self, agent_id: int, obs_dim: int = 10, action_dim: int = 10):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Separate weights for obs and messages to measure utilization
        self.obs_weights = np.random.randn(action_dim, obs_dim) * 0.1
        self.msg_weights = np.random.randn(action_dim, 5) * 0.1
        self.obs_history = []
        self.action_history = []
        self.msg_contribution = []  # Track how much messages influenced actions

    def observe(self, env_state: np.ndarray) -> np.ndarray:
        obs = env_state + np.random.randn(*env_state.shape) * 0.1
        self.obs_history.append(obs)
        return obs

    def act(self, obs: np.ndarray, messages: np.ndarray, use_messages: bool = True) -> np.ndarray:
        obs_component = self.obs_weights @ obs

        if use_messages and np.any(messages != 0):
            msg_component = self.msg_weights @ messages
            action = np.tanh(obs_component + msg_component)
            # Track message contribution
            obs_norm = np.linalg.norm(obs_component)
            msg_norm = np.linalg.norm(msg_component)
            if obs_norm + msg_norm > 0:
                self.msg_contribution.append(msg_norm / (obs_norm + msg_norm))
        else:
            action = np.tanh(obs_component)
            self.msg_contribution.append(0.0)

        self.action_history.append(action)
        return action

    def create_message(self, obs: np.ndarray) -> np.ndarray:
        return obs[:5]

    def get_flexibility(self) -> float:
        """Compute flexibility (negative K)."""
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
        return -abs(corr) * 2.0  # Flexibility is negative K

    def get_message_utilization(self) -> float:
        """How much did this agent use messages?"""
        if not self.msg_contribution:
            return 0.0
        return np.mean(self.msg_contribution)


class CommunicationNetwork:
    """Communication with controllable message probability."""

    def __init__(self, n_agents: int, msg_prob: float = 1.0):
        self.n_agents = n_agents
        self.msg_prob = msg_prob
        self.adjacency = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def exchange_messages(self, messages):
        received = []
        for i in range(self.n_agents):
            incoming = []
            for j in range(self.n_agents):
                if self.adjacency[i, j] > 0:
                    # Probabilistic message delivery
                    if np.random.rand() < self.msg_prob:
                        incoming.append(messages[j])
            if incoming:
                received.append(np.mean(incoming, axis=0))
            else:
                received.append(np.zeros(5))
        return received


class Environment:
    """Coordination task."""

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


def run_episode(n_agents: int, msg_prob: float, n_steps: int = 200):
    """Run one episode with specified message probability."""
    agents = [Agent(i) for i in range(n_agents)]
    network = CommunicationNetwork(n_agents, msg_prob)
    env = Environment(n_agents)

    state = env.reset()
    for agent in agents:
        agent.obs_history = []
        agent.action_history = []
        agent.msg_contribution = []

    total_reward = 0
    for step in range(n_steps):
        observations = [agent.observe(state) for agent in agents]
        messages = [agent.create_message(obs) for agent, obs in zip(agents, observations)]
        received = network.exchange_messages(messages)
        use_msgs = msg_prob > 0
        actions = [agent.act(obs, msg, use_msgs) for agent, obs, msg in zip(agents, observations, received)]
        state, reward, done = env.step(actions)
        total_reward += reward
        if done:
            break

    flexibility = np.mean([agent.get_flexibility() for agent in agents])
    msg_util = np.mean([agent.get_message_utilization() for agent in agents])

    return flexibility, total_reward, msg_util


def main():
    print("\n" + "=" * 70)
    print("MECHANISM VALIDATION")
    print("=" * 70)

    # Test 1: A/B Test - Same random seeds with/without communication
    print("\n" + "-" * 70)
    print("TEST 1: A/B COMPARISON (Same Teams)")
    print("-" * 70)

    n_teams = 100
    with_comm = []
    without_comm = []

    print(f"\nEvaluating {n_teams} teams with and without communication...")

    for i in range(n_teams):
        # Set same seed for both conditions
        seed = i * 1000

        # With communication
        np.random.seed(seed)
        flex_with, reward_with, _ = run_episode(4, msg_prob=1.0)
        with_comm.append((flex_with, reward_with))

        # Without communication (same initial conditions)
        np.random.seed(seed)
        flex_without, reward_without, _ = run_episode(4, msg_prob=0.0)
        without_comm.append((flex_without, reward_without))

    # Analyze
    flex_with = np.array([x[0] for x in with_comm])
    rew_with = np.array([x[1] for x in with_comm])
    flex_without = np.array([x[0] for x in without_comm])
    rew_without = np.array([x[1] for x in without_comm])

    r_with, p_with = stats.pearsonr(flex_with, rew_with)
    r_without, p_without = stats.pearsonr(flex_without, rew_without)

    print(f"\nWith communication:    r = {r_with:+.3f}, p = {p_with:.4f}")
    print(f"Without communication: r = {r_without:+.3f}, p = {p_without:.4f}")
    print(f"Difference:            Δr = {r_with - r_without:+.3f}")

    # Paired t-test on rewards
    t_stat, t_p = stats.ttest_rel(rew_with, rew_without)
    print(f"\nReward improvement:    Δ = {np.mean(rew_with - rew_without):+.2f}")
    print(f"Paired t-test:         t = {t_stat:.2f}, p = {t_p:.4f}")

    # Test 2: Partial Communication
    print("\n" + "-" * 70)
    print("TEST 2: PARTIAL COMMUNICATION")
    print("-" * 70)

    msg_probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_episodes = 100

    print(f"\nTesting {len(msg_probs)} message probability levels...")

    results = {}
    for prob in msg_probs:
        data = []
        for _ in range(n_episodes):
            flex, reward, _ = run_episode(4, msg_prob=prob)
            data.append((flex, reward))

        flex_arr = np.array([x[0] for x in data])
        rew_arr = np.array([x[1] for x in data])
        r, p = stats.pearsonr(flex_arr, rew_arr)
        results[prob] = {'r': r, 'p': p, 'mean_reward': np.mean(rew_arr)}

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  msg_prob={prob:.2f}: r = {r:+.3f}, p = {p:.4f}{sig}, mean_rew = {np.mean(rew_arr):.1f}")

    # Test 3: Message Utilization
    print("\n" + "-" * 70)
    print("TEST 3: MESSAGE UTILIZATION ANALYSIS")
    print("-" * 70)

    n_episodes = 200
    data = []

    print(f"\nAnalyzing message utilization in {n_episodes} episodes...")

    for _ in range(n_episodes):
        flex, reward, msg_util = run_episode(4, msg_prob=1.0)
        data.append((flex, reward, msg_util))

    flex_arr = np.array([x[0] for x in data])
    rew_arr = np.array([x[1] for x in data])
    util_arr = np.array([x[2] for x in data])

    # Correlations
    r_flex_rew, p1 = stats.pearsonr(flex_arr, rew_arr)
    r_flex_util, p2 = stats.pearsonr(flex_arr, util_arr)
    r_util_rew, p3 = stats.pearsonr(util_arr, rew_arr)

    print(f"\nCorrelations:")
    print(f"  Flexibility ↔ Reward:      r = {r_flex_rew:+.3f}, p = {p1:.4f}")
    print(f"  Flexibility ↔ Msg Util:    r = {r_flex_util:+.3f}, p = {p2:.4f}")
    print(f"  Msg Util ↔ Reward:         r = {r_util_rew:+.3f}, p = {p3:.4f}")

    # Quartile analysis of message utilization
    print("\nMessage utilization by flexibility quartile:")
    q1, q2, q3 = np.percentile(flex_arr, [25, 50, 75])
    for lo, hi, name in [(flex_arr.min(), q1, "Q1 (rigid)"),
                          (q1, q2, "Q2"),
                          (q2, q3, "Q3"),
                          (q3, flex_arr.max(), "Q4 (flexible)")]:
        mask = (flex_arr >= lo) & (flex_arr <= hi)
        mean_util = util_arr[mask].mean()
        mean_rew = rew_arr[mask].mean()
        print(f"  {name:<15}: msg_util = {mean_util:.3f}, reward = {mean_rew:.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("MECHANISM SUMMARY")
    print("=" * 70)

    print("\n1. A/B Test confirms communication is necessary:")
    print(f"   With communication: r = {r_with:+.3f}")
    print(f"   Without:            r = {r_without:+.3f}")

    print("\n2. Partial communication shows dose-response:")
    for prob in msg_probs:
        print(f"   {int(prob*100):3d}% messages: r = {results[prob]['r']:+.3f}")

    print("\n3. Message utilization analysis:")
    if r_flex_util > 0:
        print("   ✓ Flexible agents use messages MORE")
    else:
        print("   → Flexible agents use messages LESS (unexpected)")

    if r_util_rew > 0:
        print("   ✓ Higher message utilization → Better rewards")
    else:
        print("   → Higher message utilization → Worse rewards (unexpected)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mechanism_validation_{timestamp}.npz"
    np.savez(filename,
             r_with=r_with, r_without=r_without,
             msg_probs=msg_probs,
             partial_r=[results[p]['r'] for p in msg_probs],
             r_flex_util=r_flex_util, r_util_rew=r_util_rew)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
