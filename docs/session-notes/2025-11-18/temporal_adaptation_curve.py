#!/usr/bin/env python3
"""
Temporal Adaptation Curve

When does the flexibility-reward effect emerge during an episode?
Measures correlation at each timestep to find the "adaptation window".

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u temporal_adaptation_curve.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Agent:
    def __init__(self, agent_id):
        self.policy_weights = np.random.randn(10, 15) * 0.1
        self.obs_history = []
        self.action_history = []
        self.cumulative_rewards = []

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined)
        self.obs_history.append(obs)
        self.action_history.append(action)
        return action

    def store_cumulative_reward(self, total):
        self.cumulative_rewards.append(total)

    def get_flexibility_at_step(self, step):
        if step < 10:
            return 0.0
        obs = np.array(self.obs_history[:step]).flatten()
        actions = np.array(self.action_history[:step]).flatten()
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
        return self.state, reward


def main():
    print("\n" + "=" * 70)
    print("TEMPORAL ADAPTATION CURVE")
    print("=" * 70)

    n_episodes = 150
    max_steps = 300
    checkpoints = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300]

    print(f"\nRunning {n_episodes} episodes, measuring at {len(checkpoints)} checkpoints...")

    # Store results: for each checkpoint, list of (flex, reward) tuples
    results = {t: [] for t in checkpoints}

    for ep in range(n_episodes):
        agents = [Agent(i) for i in range(4)]
        network = Network(4)
        env = Environment(4)

        state = env.reset()
        cumulative = 0

        for step in range(max_steps):
            observations = [state + np.random.randn(10) * 0.1 for _ in range(4)]
            messages = [a.create_message(o) for a, o in zip(agents, observations)]
            received = network.exchange(messages)
            actions = [a.act(o, m) for a, o, m in zip(agents, observations, received)]
            state, reward = env.step(actions)
            cumulative += reward

            # Record at checkpoints
            if (step + 1) in checkpoints:
                flex = np.mean([a.get_flexibility_at_step(step + 1) for a in agents])
                results[step + 1].append((flex, cumulative))

        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{n_episodes} complete")

    # Analyze
    print("\n" + "=" * 70)
    print("RESULTS: CORRELATION AT EACH TIMESTEP")
    print("=" * 70)

    print(f"\n{'Step':<10} {'r':>10} {'p':>12} {'Interpretation':<20}")
    print("-" * 55)

    correlations = []
    for t in checkpoints:
        data = results[t]
        flex = np.array([x[0] for x in data])
        rew = np.array([x[1] for x in data])
        r, p = stats.pearsonr(flex, rew)
        correlations.append(r)

        if r > 0.4:
            interp = "Strong"
        elif r > 0.2:
            interp = "Moderate"
        elif r > 0.1:
            interp = "Weak"
        else:
            interp = "None"

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{t:<10} {r:>+10.3f} {p:>10.4f}{sig:>2} {interp:<20}")

    # Find emergence point
    print("\n" + "-" * 70)
    print("ADAPTATION WINDOW ANALYSIS")
    print("-" * 70)

    # Find first significant correlation
    emergence = None
    for i, t in enumerate(checkpoints):
        if correlations[i] > 0.15:
            emergence = t
            break

    if emergence:
        print(f"\nEffect emerges at: ~{emergence} steps")
    else:
        print("\nEffect does not clearly emerge")

    # Find saturation
    max_r = max(correlations)
    saturation_idx = correlations.index(max_r)
    saturation = checkpoints[saturation_idx]
    print(f"Peak correlation: r = {max_r:+.3f} at {saturation} steps")

    # Growth rate
    if len(correlations) >= 2:
        early = np.mean(correlations[:3])
        late = np.mean(correlations[-3:])
        growth = late - early
        print(f"Growth: early r = {early:+.3f}, late r = {late:+.3f}, Δ = {growth:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if emergence and emergence <= 100:
        print(f"\n✓ Effect emerges EARLY (by step {emergence})")
    elif emergence and emergence <= 150:
        print(f"\n✓ Effect emerges GRADUALLY (by step {emergence})")
    else:
        print(f"\n→ Effect emerges LATE or not clearly")

    print(f"\nAdaptation window: {emergence or 'N/A'} to {saturation} steps")
    print(f"Peak correlation: r = {max_r:+.3f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"temporal_adaptation_curve_{timestamp}.npz"
    np.savez(filename, checkpoints=checkpoints, correlations=correlations)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
