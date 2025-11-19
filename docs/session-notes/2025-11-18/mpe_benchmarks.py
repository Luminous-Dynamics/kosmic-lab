#!/usr/bin/env python3
"""
MPE/PettingZoo Benchmarks

Tests flexibility in standard multi-agent benchmarks for publication credibility.
Environments: Simple Spread, Simple Reference, Cooperative Navigation

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy python3Packages.pettingzoo --run "python3 mpe_benchmarks.py"
"""

import numpy as np
from typing import Dict, List
from scipy import stats
from datetime import datetime

# Try to import PettingZoo
try:
    from pettingzoo.mpe import simple_spread_v3, simple_reference_v3, simple_tag_v3
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    print("WARNING: PettingZoo not available. Using simulated MPE environments.")


class Agent:
    """Agent for MPE environments."""

    def __init__(self, agent_id: str, obs_dim: int, action_dim: int):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_weights = np.random.randn(action_dim, obs_dim) * 0.1
        self.obs_history = []
        self.action_history = []
        self.learning_rate = 0.01

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Select continuous action."""
        action = np.tanh(self.policy_weights @ obs)
        self.obs_history.append(obs.flatten())
        self.action_history.append(action.flatten())
        return action

    def get_flexibility(self) -> float:
        """Calculate flexibility from history."""
        if len(self.obs_history) < 10:
            return 0.0

        obs = np.concatenate(self.obs_history[-50:])
        actions = np.concatenate(self.action_history[-50:])

        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0

        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        if np.isnan(corr):
            return 0.0

        return -abs(corr) * 2.0

    def update(self, reward: float):
        """Simple policy update."""
        noise = np.random.randn(*self.policy_weights.shape)
        self.policy_weights += self.learning_rate * reward * noise

    def reset(self):
        """Clear history for new episode."""
        self.obs_history = []
        self.action_history = []


def run_simple_spread(n_episodes: int = 200) -> Dict:
    """Run Simple Spread benchmark."""
    n_agents = 3
    obs_dim = 18  # Typical for simple_spread
    action_dim = 5  # 5 continuous actions

    results = []

    for ep in range(n_episodes):
        if PETTINGZOO_AVAILABLE:
            env = simple_spread_v3.parallel_env()
            observations, _ = env.reset()
            agents = {name: Agent(name, obs_dim, action_dim)
                     for name in env.agents}
        else:
            # Simulated environment
            agent_names = [f"agent_{i}" for i in range(n_agents)]
            agents = {name: Agent(name, obs_dim, action_dim)
                     for name in agent_names}
            observations = {name: np.random.randn(obs_dim) * 0.1
                          for name in agent_names}

        total_rewards = {name: 0.0 for name in agents}

        for step in range(25):  # MPE typically runs 25 steps
            actions = {}
            for name, agent in agents.items():
                obs = observations.get(name, np.zeros(obs_dim))
                if len(obs) != obs_dim:
                    obs = np.zeros(obs_dim)
                actions[name] = agent.act(obs)

            if PETTINGZOO_AVAILABLE:
                observations, rewards, terminations, truncations, _ = env.step(actions)
                done = all(terminations.values()) or all(truncations.values())
            else:
                # Simulated dynamics
                positions = np.array([obs[:2] for obs in observations.values()])
                target = np.zeros(2)
                distances = np.linalg.norm(positions - target, axis=1)
                base_reward = -np.mean(distances)

                observations = {name: np.random.randn(obs_dim) * 0.1
                              for name in agents}
                rewards = {name: base_reward + np.random.randn() * 0.1
                          for name in agents}
                done = False

            for name, reward in rewards.items():
                total_rewards[name] += reward
                agents[name].update(reward)

            if done:
                break

        if PETTINGZOO_AVAILABLE:
            env.close()

        # Aggregate metrics
        mean_flex = np.mean([agent.get_flexibility() for agent in agents.values()])
        mean_reward = np.mean(list(total_rewards.values()))

        results.append({
            'flexibility': mean_flex,
            'reward': mean_reward
        })

    return {'env_name': 'simple_spread', 'results': results}


def run_simple_reference(n_episodes: int = 200) -> Dict:
    """Run Simple Reference benchmark (communication task)."""
    n_agents = 2
    obs_dim = 21
    action_dim = 5

    results = []

    for ep in range(n_episodes):
        if PETTINGZOO_AVAILABLE:
            env = simple_reference_v3.parallel_env()
            observations, _ = env.reset()
            agents = {name: Agent(name, obs_dim, action_dim)
                     for name in env.agents}
        else:
            agent_names = ['agent_0', 'speaker_0']
            agents = {name: Agent(name, obs_dim, action_dim)
                     for name in agent_names}
            observations = {name: np.random.randn(obs_dim) * 0.1
                          for name in agent_names}

        total_rewards = {name: 0.0 for name in agents}

        for step in range(25):
            actions = {}
            for name, agent in agents.items():
                obs = observations.get(name, np.zeros(obs_dim))
                if len(obs) != obs_dim:
                    obs = np.zeros(obs_dim)
                actions[name] = agent.act(obs)

            if PETTINGZOO_AVAILABLE:
                observations, rewards, terminations, truncations, _ = env.step(actions)
                done = all(terminations.values()) or all(truncations.values())
            else:
                # Communication success depends on coordination
                actions_arr = np.array(list(actions.values()))
                coordination = 1 - np.std(actions_arr)
                base_reward = coordination * 0.5 - 0.5

                observations = {name: np.random.randn(obs_dim) * 0.1
                              for name in agents}
                rewards = {name: base_reward + np.random.randn() * 0.1
                          for name in agents}
                done = False

            for name, reward in rewards.items():
                total_rewards[name] += reward
                agents[name].update(reward)

            if done:
                break

        if PETTINGZOO_AVAILABLE:
            env.close()

        mean_flex = np.mean([agent.get_flexibility() for agent in agents.values()])
        mean_reward = np.mean(list(total_rewards.values()))

        results.append({
            'flexibility': mean_flex,
            'reward': mean_reward
        })

    return {'env_name': 'simple_reference', 'results': results}


def run_cooperative_navigation(n_episodes: int = 200) -> Dict:
    """Run Cooperative Navigation (variant of spread with more agents)."""
    n_agents = 4
    obs_dim = 24
    action_dim = 5

    results = []

    for ep in range(n_episodes):
        agent_names = [f"agent_{i}" for i in range(n_agents)]
        agents = {name: Agent(name, obs_dim, action_dim)
                 for name in agent_names}

        # Target positions for each agent
        targets = np.random.randn(n_agents, 2)
        positions = np.random.randn(n_agents, 2)

        observations = {}
        for i, name in enumerate(agent_names):
            # Obs: own pos, target, other agent positions
            obs = np.zeros(obs_dim)
            obs[:2] = positions[i]
            obs[2:4] = targets[i]
            for j, other in enumerate(agent_names):
                if i != j:
                    obs[4 + j*2:6 + j*2] = positions[j]
            observations[name] = obs

        total_rewards = {name: 0.0 for name in agents}

        for step in range(25):
            actions = {}
            for name, agent in agents.items():
                obs = observations[name]
                actions[name] = agent.act(obs)

            # Update positions based on actions
            for i, name in enumerate(agent_names):
                positions[i] += actions[name][:2] * 0.1

            # Reward: negative distance to targets + collision penalty
            distances = np.linalg.norm(positions - targets, axis=1)
            collision_penalty = 0
            for i in range(n_agents):
                for j in range(i+1, n_agents):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 0.5:
                        collision_penalty += 0.1

            base_reward = -np.mean(distances) - collision_penalty

            # Update observations
            for i, name in enumerate(agent_names):
                obs = np.zeros(obs_dim)
                obs[:2] = positions[i]
                obs[2:4] = targets[i]
                for j, other in enumerate(agent_names):
                    if i != j:
                        obs[4 + j*2:6 + j*2] = positions[j]
                observations[name] = obs

            rewards = {name: base_reward + np.random.randn() * 0.1
                      for name in agents}

            for name, reward in rewards.items():
                total_rewards[name] += reward
                agents[name].update(reward)

        mean_flex = np.mean([agent.get_flexibility() for agent in agents.values()])
        mean_reward = np.mean(list(total_rewards.values()))

        results.append({
            'flexibility': mean_flex,
            'reward': mean_reward
        })

    return {'env_name': 'cooperative_navigation', 'results': results}


def analyze_results(experiment: Dict) -> Dict:
    """Analyze flexibility-reward correlation."""
    results = experiment['results']

    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])

    r, p = stats.pearsonr(flexibility, rewards)
    rho, _ = stats.spearmanr(flexibility, rewards)

    # Effect size
    median_flex = np.median(flexibility)
    high_flex = rewards[flexibility > median_flex]
    low_flex = rewards[flexibility <= median_flex]

    if len(high_flex) > 1 and len(low_flex) > 1:
        pooled_std = np.sqrt((high_flex.std()**2 + low_flex.std()**2) / 2)
        d = (high_flex.mean() - low_flex.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        d = 0

    return {
        'r': r, 'p': p, 'rho': rho, 'd': d,
        'n': len(results),
        'mean_flex': flexibility.mean(),
        'mean_reward': rewards.mean()
    }


def main():
    print("\n" + "=" * 70)
    print("MPE/PETTINGZOO BENCHMARKS")
    print("=" * 70)

    if not PETTINGZOO_AVAILABLE:
        print("\n⚠️  PettingZoo not available - using simulated MPE environments")
        print("    Install with: pip install pettingzoo[mpe]")

    # Run all benchmarks
    benchmarks = [
        ("Simple Spread", run_simple_spread),
        ("Simple Reference", run_simple_reference),
        ("Cooperative Navigation", run_cooperative_navigation),
    ]

    all_results = {}

    for name, run_func in benchmarks:
        print(f"\nRunning {name}...")
        experiment = run_func(n_episodes=200)
        analysis = analyze_results(experiment)
        all_results[experiment['env_name']] = {**experiment, **analysis}

        sig = '***' if analysis['p'] < 0.001 else '**' if analysis['p'] < 0.01 else '*' if analysis['p'] < 0.05 else ''
        print(f"  r = {analysis['r']:+.3f}, p = {analysis['p']:.2e} {sig}")

    # Summary
    print("\n" + "=" * 70)
    print("MPE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Environment':<25} {'r':>8} {'p':>12} {'d':>8} {'Sig':>6}")
    print("-" * 70)

    for env_name, data in all_results.items():
        sig = '***' if data['p'] < 0.001 else '**' if data['p'] < 0.01 else '*' if data['p'] < 0.05 else 'ns'
        print(f"{env_name:<25} {data['r']:>+8.3f} {data['p']:>12.2e} {data['d']:>+8.2f} {sig:>6}")

    # Meta-analysis
    print("\n" + "-" * 70)
    all_r = [data['r'] for data in all_results.values()]
    mean_r = np.mean(all_r)
    print(f"Mean r across MPE benchmarks: {mean_r:+.3f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mpe_benchmarks_{timestamp}.npz"

    np.savez(filename,
             environments=list(all_results.keys()),
             correlations=[data['r'] for data in all_results.values()],
             p_values=[data['p'] for data in all_results.values()],
             effect_sizes=[data['d'] for data in all_results.values()],
             pettingzoo_available=PETTINGZOO_AVAILABLE)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")

    return all_results


if __name__ == '__main__':
    main()
