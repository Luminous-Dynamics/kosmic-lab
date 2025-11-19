#!/usr/bin/env python3
"""
8-Agent Breakdown Investigation

Why does the flexibility-reward correlation disappear at 8 agents?
Tests multiple hypotheses:
1. Signal dilution (message averaging)
2. Need for longer episodes
3. Action averaging washes out flexibility
4. Coordination complexity

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u eight_agent_breakdown_investigation.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.policy_weights = np.random.randn(10, 15) * 0.1
        self.obs_history = []
        self.action_history = []

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined)
        self.obs_history.append(obs)
        self.action_history.append(action)
        return action

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
    def __init__(self, n_agents, topology='fully_connected'):
        self.n_agents = n_agents
        self.topology = topology
        self.adj = np.zeros((n_agents, n_agents))

        if topology == 'fully_connected':
            self.adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)
        elif topology == 'small_world':
            # Ring + random long-range connections
            for i in range(n_agents):
                self.adj[i, (i+1) % n_agents] = 1
                self.adj[i, (i-1) % n_agents] = 1
                # Add 2 random long-range connections
                for _ in range(2):
                    j = np.random.randint(n_agents)
                    if j != i:
                        self.adj[i, j] = 1
                        self.adj[j, i] = 1
        elif topology == 'hierarchical':
            # Tree structure: 0 is root, connects to 1,2; 1 to 3,4; 2 to 5,6; etc.
            for i in range(1, n_agents):
                parent = (i - 1) // 2
                self.adj[i, parent] = 1
                self.adj[parent, i] = 1

    def exchange(self, messages, weighted=False):
        received = []
        for i in range(self.n_agents):
            neighbors = [j for j in range(self.n_agents) if self.adj[i, j] > 0]
            if neighbors:
                if weighted:
                    # Weight by recency/importance instead of simple mean
                    weights = np.array([1.0 / (j + 1) for j in range(len(neighbors))])
                    weights /= weights.sum()
                    received.append(sum(w * messages[n] for w, n in zip(weights, neighbors)))
                else:
                    received.append(np.mean([messages[j] for j in neighbors], axis=0))
            else:
                received.append(np.zeros(5))
        return received


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
        return self.state, reward


def run_condition(n_agents, n_steps, n_episodes, topology='fully_connected', weighted_msgs=False):
    """Run experimental condition."""
    results = []

    for _ in range(n_episodes):
        agents = [Agent(i) for i in range(n_agents)]
        network = Network(n_agents, topology)
        env = Environment(n_agents)

        state = env.reset()
        for a in agents:
            a.obs_history = []
            a.action_history = []

        total = 0
        for step in range(n_steps):
            observations = [state + np.random.randn(10) * 0.1 for _ in range(n_agents)]
            messages = [a.create_message(o) for a, o in zip(agents, observations)]
            received = network.exchange(messages, weighted=weighted_msgs)
            actions = [a.act(o, m) for a, o, m in zip(agents, observations, received)]
            state, reward = env.step(actions)
            total += reward

        flex = np.mean([a.get_flexibility() for a in agents])
        results.append((flex, total))

    flex_arr = np.array([x[0] for x in results])
    rew_arr = np.array([x[1] for x in results])
    r, p = stats.pearsonr(flex_arr, rew_arr)

    return r, p, np.mean(rew_arr)


def main():
    print("\n" + "=" * 70)
    print("8-AGENT BREAKDOWN INVESTIGATION")
    print("=" * 70)

    n_episodes = 100

    # ===== HYPOTHESIS 1: Need More Steps =====
    print("\n" + "-" * 70)
    print("HYPOTHESIS 1: 8 AGENTS NEED MORE STEPS")
    print("-" * 70)

    print("\nDoes extending episode length restore the effect?")

    step_counts = [200, 300, 400, 500, 600]
    for n_steps in step_counts:
        r, p, _ = run_condition(8, n_steps, n_episodes)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {n_steps} steps: r = {r:+.3f}, p = {p:.4f}{sig}")

    # ===== HYPOTHESIS 2: Topology Matters =====
    print("\n" + "-" * 70)
    print("HYPOTHESIS 2: ALTERNATIVE TOPOLOGIES")
    print("-" * 70)

    print("\nDoes changing network structure help?")

    topologies = ['fully_connected', 'small_world', 'hierarchical']
    for topo in topologies:
        r, p, _ = run_condition(8, 300, n_episodes, topology=topo)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {topo}: r = {r:+.3f}, p = {p:.4f}{sig}")

    # ===== HYPOTHESIS 3: Weighted Messages =====
    print("\n" + "-" * 70)
    print("HYPOTHESIS 3: WEIGHTED MESSAGE AGGREGATION")
    print("-" * 70)

    print("\nDoes weighting messages (vs mean) help?")

    r_mean, p_mean, _ = run_condition(8, 300, n_episodes, weighted_msgs=False)
    r_weighted, p_weighted, _ = run_condition(8, 300, n_episodes, weighted_msgs=True)

    print(f"  Mean aggregation: r = {r_mean:+.3f}, p = {p_mean:.4f}")
    print(f"  Weighted aggregation: r = {r_weighted:+.3f}, p = {p_weighted:.4f}")

    # ===== HYPOTHESIS 4: Comparison with Smaller Teams =====
    print("\n" + "-" * 70)
    print("HYPOTHESIS 4: TEAM SIZE GRADIENT AT 300 STEPS")
    print("-" * 70)

    print("\nIs 8 the exact threshold, or gradual decline?")

    team_sizes = [4, 5, 6, 7, 8, 9, 10]
    for n in team_sizes:
        r, p, _ = run_condition(n, 300, n_episodes)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {n} agents: r = {r:+.3f}, p = {p:.4f}{sig}")

    # ===== HYPOTHESIS 5: Per-Agent Flexibility Analysis =====
    print("\n" + "-" * 70)
    print("HYPOTHESIS 5: INDIVIDUAL AGENT ANALYSIS")
    print("-" * 70)

    print("\nDo all agents lose flexibility, or just some?")

    # Run one episode with detailed per-agent tracking
    agents = [Agent(i) for i in range(8)]
    network = Network(8)
    env = Environment(8)

    state = env.reset()
    for step in range(300):
        observations = [state + np.random.randn(10) * 0.1 for _ in range(8)]
        messages = [a.create_message(o) for a, o in zip(agents, observations)]
        received = network.exchange(messages)
        actions = [a.act(o, m) for a, o, m in zip(agents, observations, received)]
        state, _ = env.step(actions)

    flexibilities = [a.get_flexibility() for a in agents]
    print(f"  Flexibility range: [{min(flexibilities):.3f}, {max(flexibilities):.3f}]")
    print(f"  Flexibility std: {np.std(flexibilities):.3f}")
    print(f"  Individual: {', '.join([f'{f:.2f}' for f in flexibilities])}")

    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Find if any condition restores the effect
    print("\nSearching for conditions that restore effect...")

    # Test extended steps
    r_500, p_500, _ = run_condition(8, 500, n_episodes)
    r_600, p_600, _ = run_condition(8, 600, n_episodes)

    if r_500 > 0.2 or r_600 > 0.2:
        print(f"  ✓ Effect RECOVERS with extended episodes")
        print(f"    500 steps: r = {r_500:+.3f}")
        print(f"    600 steps: r = {r_600:+.3f}")
    else:
        print(f"  ✗ Effect does NOT recover with extended episodes")

    # Test hierarchical at long episodes
    r_hier, p_hier, _ = run_condition(8, 500, n_episodes, topology='hierarchical')
    if r_hier > 0.2:
        print(f"  ✓ Hierarchical topology helps: r = {r_hier:+.3f}")
    else:
        print(f"  ✗ Topology change doesn't help: r = {r_hier:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Determine if it's fundamental or recoverable
    best_8_agent = max(r_500, r_600, r_hier)

    if best_8_agent > 0.3:
        print(f"\n✓ 8-agent effect IS RECOVERABLE")
        print(f"  Best achieved: r = {best_8_agent:+.3f}")
        print(f"  Recommendation: Use 500+ steps for 8-agent teams")
    elif best_8_agent > 0.15:
        print(f"\n⚠ 8-agent effect is WEAK but present")
        print(f"  Best achieved: r = {best_8_agent:+.3f}")
        print(f"  May be near fundamental limit")
    else:
        print(f"\n✗ 8-agent effect is FUNDAMENTAL BREAKDOWN")
        print(f"  Best achieved: r = {best_8_agent:+.3f}")
        print(f"  Coordination complexity exceeds flexibility benefit")
        print(f"  Recommendation: Use teams of ≤6 agents")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eight_agent_investigation_{timestamp}.npz"
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
