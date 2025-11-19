#!/usr/bin/env python3
"""
Track J1 Extended: MPE with Longer Episodes

Test if the effect appears in MPE when given sufficient steps (150, 200, 300).

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_j1_extended_episodes.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.policy_weights = np.random.randn(4, 10) * 0.1
        self.obs_history = []
        self.action_history = []

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined + np.random.randn(4) * 0.3)
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
        return obs[:4]


class Network:
    def __init__(self, n_agents):
        self.n_agents = n_agents

    def exchange(self, messages):
        avg = np.mean(messages, axis=0)
        return [avg for _ in range(self.n_agents)]


class SimpleSpreadEnv:
    def __init__(self, n_agents=4):
        self.n_agents = n_agents
        self.agents_pos = np.zeros((n_agents, 2))
        self.agents_vel = np.zeros((n_agents, 2))
        self.landmarks = np.zeros((n_agents, 2))

    def reset(self):
        self.agents_pos = np.random.randn(self.n_agents, 2) * 0.5
        self.agents_vel = np.zeros((self.n_agents, 2))
        self.landmarks = np.random.randn(self.n_agents, 2)
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            own = np.concatenate([self.agents_pos[i], self.agents_vel[i]])
            dists = np.linalg.norm(self.landmarks - self.agents_pos[i], axis=1)
            nearest = np.argmin(dists)
            rel_landmark = self.landmarks[nearest] - self.agents_pos[i]
            obs.append(np.concatenate([own, rel_landmark]))
        return obs

    def step(self, actions):
        for i in range(self.n_agents):
            self.agents_vel[i] = actions[i][:2] * 0.5
            self.agents_pos[i] += self.agents_vel[i] * 0.1

        reward = 0
        for landmark in self.landmarks:
            dists = np.linalg.norm(self.agents_pos - landmark, axis=1)
            reward -= np.min(dists)

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self.agents_pos[i] - self.agents_pos[j])
                if dist < 0.2:
                    reward -= 0.5

        return self._get_obs(), reward


def run_condition(n_agents, n_steps, n_episodes=100):
    results = []

    for _ in range(n_episodes):
        agents = [Agent(i) for i in range(n_agents)]
        network = Network(n_agents)
        env = SimpleSpreadEnv(n_agents)

        observations = env.reset()
        for a in agents:
            a.obs_history = []
            a.action_history = []

        total_reward = 0
        for step in range(n_steps):
            noisy_obs = [o + np.random.randn(6) * 0.05 for o in observations]
            messages = [a.create_message(o) for a, o in zip(agents, noisy_obs)]
            received = network.exchange(messages)
            actions = [a.act(o, m) for a, o, m in zip(agents, noisy_obs, received)]
            observations, reward = env.step(actions)
            total_reward += reward

        flex = np.mean([a.get_flexibility() for a in agents])
        results.append((flex, total_reward))

    flex_arr = np.array([x[0] for x in results])
    rew_arr = np.array([x[1] for x in results])
    r, p = stats.pearsonr(flex_arr, rew_arr)

    return r, p


def main():
    print("\n" + "=" * 70)
    print("TRACK J1 EXTENDED: MPE WITH LONGER EPISODES")
    print("=" * 70)

    np.random.seed(42)

    step_counts = [100, 150, 200, 300]
    n_agents = 4

    print(f"\nTesting Simple Spread with {n_agents} agents...")

    results = {}

    for n_steps in step_counts:
        r, p = run_condition(n_agents, n_steps, n_episodes=100)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        results[n_steps] = {'r': r, 'p': p}
        print(f"  {n_steps} steps: r = {r:+.3f}, p = {p:.4f}{sig}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Dose-response in MPE
    steps = list(results.keys())
    rs = [results[s]['r'] for s in steps]
    r_dose, p_dose = stats.pearsonr(steps, rs)

    print(f"\nDose-response (steps ↔ r): r = {r_dose:+.3f}, p = {p_dose:.4f}")

    # Find threshold
    threshold = None
    for s in step_counts:
        if results[s]['r'] > 0.15 and results[s]['p'] < 0.05:
            threshold = s
            break

    if threshold:
        print(f"Effect emerges at: {threshold} steps")
    else:
        print("Effect does not reach significance")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    max_r = max(rs)
    best_steps = step_counts[rs.index(max_r)]

    if max_r > 0.3:
        print(f"\n✓ VALIDATED: Effect appears in MPE at {best_steps} steps")
        print(f"  r = {max_r:+.3f}")
        print(f"  Confirms scaling law: MPE needs longer episodes")
    elif max_r > 0.15:
        print(f"\n⚠ PARTIAL: Weak effect at {best_steps} steps (r = {max_r:+.3f})")
    else:
        print(f"\n✗ Effect still not present in MPE even at 300 steps")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_j1_extended_{timestamp}.npz"
    np.savez(filename, results=results)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
