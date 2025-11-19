#!/usr/bin/env python3
"""
Episode Length Gradient Test

Tests dose-response relationship between episode length and flexibility-reward correlation.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u episode_length_gradient.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Agent:
    def __init__(self, agent_id: int, obs_dim: int = 10, action_dim: int = 10):
        self.id = agent_id
        self.policy_weights = np.random.randn(action_dim, obs_dim + 5) * 0.1
        self.obs_history = []
        self.action_history = []

    def observe(self, env_state):
        obs = env_state + np.random.randn(*env_state.shape) * 0.1
        self.obs_history.append(obs)
        return obs

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined)
        self.action_history.append(action)
        return action

    def create_message(self, obs):
        return obs[:5]

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
        done = dist < 0.2
        return self.state, reward, done


def run_condition(n_steps: int, n_episodes: int = 150, n_seeds: int = 3):
    """Run with multiple seeds for CI."""
    all_r = []

    for seed in range(n_seeds):
        np.random.seed(seed * 1000)
        results = []

        for _ in range(n_episodes):
            agents = [Agent(i) for i in range(4)]
            network = Network(4)
            env = Environment(4)

            state = env.reset()
            for agent in agents:
                agent.obs_history = []
                agent.action_history = []

            total = 0
            for step in range(n_steps):
                observations = [agent.observe(state) for agent in agents]
                messages = [agent.create_message(obs) for agent, obs in zip(agents, observations)]
                received = network.exchange(messages)
                actions = [agent.act(obs, msg) for agent, obs, msg in zip(agents, observations, received)]
                state, reward, done = env.step(actions)
                total += reward
                if done:
                    break

            flex = np.mean([agent.get_flexibility() for agent in agents])
            results.append((flex, total))

        flex_arr = np.array([x[0] for x in results])
        rew_arr = np.array([x[1] for x in results])
        r, _ = stats.pearsonr(flex_arr, rew_arr)
        all_r.append(r)

    return np.mean(all_r), np.std(all_r), all_r


def main():
    print("\n" + "=" * 70)
    print("EPISODE LENGTH GRADIENT TEST")
    print("=" * 70)

    # Test gradient
    step_counts = [25, 50, 75, 100, 150, 200, 250, 300]

    print(f"\nTesting {len(step_counts)} episode lengths with 3 seeds each...")

    results = {}
    for n_steps in step_counts:
        mean_r, std_r, all_r = run_condition(n_steps)
        results[n_steps] = {'mean': mean_r, 'std': std_r, 'all': all_r}

        sig = '***' if mean_r > 0.3 else '**' if mean_r > 0.2 else '*' if mean_r > 0.1 else ''
        print(f"  {n_steps:3d} steps: r = {mean_r:+.3f} ± {std_r:.3f} {sig}")

    # Analysis
    print("\n" + "=" * 70)
    print("DOSE-RESPONSE ANALYSIS")
    print("=" * 70)

    steps = np.array(step_counts)
    means = np.array([results[s]['mean'] for s in step_counts])

    # Correlation between steps and r
    r_dose, p_dose = stats.pearsonr(steps, means)
    print(f"\nSteps ↔ Correlation: r = {r_dose:+.3f}, p = {p_dose:.4f}")

    # Find threshold
    threshold_r = 0.2
    above_threshold = [s for s in step_counts if results[s]['mean'] > threshold_r]
    if above_threshold:
        print(f"Threshold for r > {threshold_r}: ≥{min(above_threshold)} steps")

    # Saturation point
    max_r = max(means)
    saturation = [s for s in step_counts if results[s]['mean'] > 0.9 * max_r]
    if saturation:
        print(f"Saturation point (90% of max): ≥{min(saturation)} steps")

    # Effect sizes
    print("\n" + "-" * 70)
    print("EFFECT SIZE BY EPISODE LENGTH")
    print("-" * 70)

    print(f"\n{'Steps':<10} {'r':>10} {'95% CI':>20} {'Interpretation':<20}")
    print("-" * 60)

    for n_steps in step_counts:
        mean_r = results[n_steps]['mean']
        std_r = results[n_steps]['std']
        ci_low = mean_r - 1.96 * std_r
        ci_high = mean_r + 1.96 * std_r

        if mean_r > 0.4:
            interp = "Strong"
        elif mean_r > 0.2:
            interp = "Moderate"
        elif mean_r > 0.1:
            interp = "Weak"
        else:
            interp = "None"

        print(f"{n_steps:<10} {mean_r:>+10.3f} [{ci_low:>+7.3f}, {ci_high:>+7.3f}]  {interp:<20}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if r_dose > 0.8:
        print("\n✓ STRONG dose-response relationship confirmed")
        print(f"  Episode length strongly predicts correlation strength (r = {r_dose:+.2f})")
    elif r_dose > 0.5:
        print("\n✓ MODERATE dose-response relationship")
        print(f"  Episode length moderately predicts correlation (r = {r_dose:+.2f})")
    else:
        print("\n? WEAK dose-response relationship")
        print(f"  Relationship is variable (r = {r_dose:+.2f})")

    # Practical recommendation
    print("\nPractical recommendation:")
    if above_threshold:
        print(f"  Minimum {min(above_threshold)} steps for meaningful effect")
    if saturation:
        print(f"  Optimal ≥{min(saturation)} steps (diminishing returns beyond)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_length_gradient_{timestamp}.npz"
    np.savez(filename,
             steps=step_counts,
             means=means,
             stds=[results[s]['std'] for s in step_counts],
             r_dose=r_dose)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
