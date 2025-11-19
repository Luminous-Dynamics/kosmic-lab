#!/usr/bin/env python3
"""
Track J1: MPE-Style Validation

Validate flexibility-reward correlation in MPE-style environments.
Implements simplified versions of:
1. Simple Spread - agents must cover landmarks
2. Cooperative Navigation - agents must reach goals without collision

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_j1_mpe_validation.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.policy_weights = np.random.randn(4, 10) * 0.1  # 4D action, 10D input (6 obs + 4 msg)
        self.obs_history = []
        self.action_history = []
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)

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
        # All agents receive average of all messages
        avg = np.mean(messages, axis=0)
        return [avg for _ in range(self.n_agents)]


class SimpleSpreadEnv:
    """
    MPE Simple Spread: Agents must spread to cover landmarks.
    Reward based on minimum distance from each landmark to nearest agent.
    """

    def __init__(self, n_agents=4):
        self.n_agents = n_agents
        self.n_landmarks = n_agents
        self.agents_pos = np.zeros((n_agents, 2))
        self.agents_vel = np.zeros((n_agents, 2))
        self.landmarks = np.zeros((n_agents, 2))

    def reset(self):
        # Random positions
        self.agents_pos = np.random.randn(self.n_agents, 2) * 0.5
        self.agents_vel = np.zeros((self.n_agents, 2))
        self.landmarks = np.random.randn(self.n_landmarks, 2)
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            # Own position and velocity
            own = np.concatenate([self.agents_pos[i], self.agents_vel[i]])
            # Relative position to nearest landmark
            dists = np.linalg.norm(self.landmarks - self.agents_pos[i], axis=1)
            nearest = np.argmin(dists)
            rel_landmark = self.landmarks[nearest] - self.agents_pos[i]
            obs.append(np.concatenate([own, rel_landmark]))
        return obs

    def step(self, actions):
        # Actions: [vel_x, vel_y, 0, 0] (last 2 unused)
        for i in range(self.n_agents):
            self.agents_vel[i] = actions[i][:2] * 0.5
            self.agents_pos[i] += self.agents_vel[i] * 0.1

        # Reward: negative min distance from each landmark to any agent
        reward = 0
        for landmark in self.landmarks:
            dists = np.linalg.norm(self.agents_pos - landmark, axis=1)
            reward -= np.min(dists)

        # Collision penalty
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self.agents_pos[i] - self.agents_pos[j])
                if dist < 0.2:
                    reward -= 0.5

        return self._get_obs(), reward


class CooperativeNavEnv:
    """
    MPE Cooperative Navigation: Each agent has a goal, must reach without collision.
    """

    def __init__(self, n_agents=4):
        self.n_agents = n_agents
        self.agents_pos = np.zeros((n_agents, 2))
        self.agents_vel = np.zeros((n_agents, 2))
        self.goals = np.zeros((n_agents, 2))

    def reset(self):
        self.agents_pos = np.random.randn(self.n_agents, 2) * 0.5
        self.agents_vel = np.zeros((self.n_agents, 2))
        self.goals = np.random.randn(self.n_agents, 2)
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            own = np.concatenate([self.agents_pos[i], self.agents_vel[i]])
            rel_goal = self.goals[i] - self.agents_pos[i]
            obs.append(np.concatenate([own, rel_goal]))
        return obs

    def step(self, actions):
        for i in range(self.n_agents):
            self.agents_vel[i] = actions[i][:2] * 0.5
            self.agents_pos[i] += self.agents_vel[i] * 0.1

        # Reward: negative distance to own goal
        reward = 0
        for i in range(self.n_agents):
            dist = np.linalg.norm(self.agents_pos[i] - self.goals[i])
            reward -= dist

        # Collision penalty (stronger here)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self.agents_pos[i] - self.agents_pos[j])
                if dist < 0.2:
                    reward -= 1.0

        return self._get_obs(), reward


def run_condition(env_class, n_agents=4, n_episodes=100, n_steps=100):
    """Run experiment on given environment."""
    results = []

    for _ in range(n_episodes):
        agents = [Agent(i) for i in range(n_agents)]
        network = Network(n_agents)
        env = env_class(n_agents)

        observations = env.reset()
        for a in agents:
            a.obs_history = []
            a.action_history = []

        total_reward = 0
        for step in range(n_steps):
            # Add observation noise
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

    return r, p, np.mean(rew_arr), np.std(rew_arr)


def main():
    print("\n" + "=" * 70)
    print("TRACK J1: MPE-STYLE VALIDATION")
    print("=" * 70)

    np.random.seed(42)
    n_episodes = 150

    environments = {
        'Simple Spread': SimpleSpreadEnv,
        'Cooperative Nav': CooperativeNavEnv,
    }

    print(f"\nTesting {n_episodes} episodes per environment...")

    results = {}

    for name, env_class in environments.items():
        print(f"\n{name}:")

        # Test different team sizes
        for n_agents in [2, 4, 6]:
            r, p, mean_rew, std_rew = run_condition(
                env_class, n_agents=n_agents, n_episodes=n_episodes
            )
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

            key = f"{name}_{n_agents}"
            results[key] = {'r': r, 'p': p, 'mean': mean_rew, 'std': std_rew}

            print(f"  {n_agents} agents: r = {r:+.3f}, p = {p:.4f}{sig}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compare to original coordination task
    original_r = 0.698  # From Paper 3

    all_rs = [results[k]['r'] for k in results]
    mean_r = np.mean(all_rs)

    print(f"\nOriginal task: r = {original_r:+.3f}")
    print(f"MPE tasks mean: r = {mean_r:+.3f}")

    # Per-environment summary
    print("\nPer-Environment Summary:")
    for env_name in environments:
        env_rs = [results[k]['r'] for k in results if env_name in k]
        env_mean = np.mean(env_rs)
        print(f"  {env_name}: mean r = {env_mean:+.3f}")

    # Team size effect
    print("\nTeam Size Effect (across environments):")
    for n in [2, 4, 6]:
        size_rs = [results[k]['r'] for k in results if f"_{n}" in k]
        size_mean = np.mean(size_rs)
        print(f"  {n} agents: mean r = {size_mean:+.3f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    sig_count = sum(1 for k in results if results[k]['p'] < 0.05)
    total = len(results)

    print(f"\nSignificant correlations: {sig_count}/{total}")

    if mean_r > 0.3:
        print(f"\n✓ VALIDATED: Flexibility-reward effect generalizes to MPE")
        print(f"  Mean correlation: r = {mean_r:+.3f}")
        if mean_r > original_r * 0.7:
            print(f"  Effect size comparable to original task")
        else:
            print(f"  Effect size smaller than original (may need longer episodes)")
    elif mean_r > 0.15:
        print(f"\n⚠ PARTIAL: Effect present but weaker in MPE")
        print(f"  Mean r = {mean_r:+.3f} vs original r = {original_r:+.3f}")
    else:
        print(f"\n✗ NOT VALIDATED: Effect does not generalize to MPE")
        print(f"  Mean r = {mean_r:+.3f}")

    # Specific findings
    best_key = max(results.keys(), key=lambda k: results[k]['r'])
    worst_key = min(results.keys(), key=lambda k: results[k]['r'])

    print(f"\nBest: {best_key} (r = {results[best_key]['r']:+.3f})")
    print(f"Worst: {worst_key} (r = {results[worst_key]['r']:+.3f})")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_j1_mpe_{timestamp}.npz"
    np.savez(filename, results=results)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
