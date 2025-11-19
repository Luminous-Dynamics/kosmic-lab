#!/usr/bin/env python3
"""
Track D: Corrected K-Index Validation

Validates that corrected K formulation (flexibility × coordination) shows
positive correlation with performance, as discovered in experimental analysis.

Expected result: r ≈ +0.58, p < 0.001

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 track_d_corrected_k_validation.py"
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from pathlib import Path
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


class CorrectedKValidation:
    """Track D runner with corrected K metrics."""

    def __init__(self, n_agents: int = 4, n_episodes: int = 200, topology: str = "fully_connected"):
        self.n_agents = n_agents
        self.n_episodes = n_episodes
        self.topology = topology
        self.results = []

    def run_episode(self, agents, network, env) -> Dict:
        """Run one episode with corrected K computation."""
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

        # Compute K metrics
        individual_k = [agent.get_k_index() for agent in agents]
        mean_individual_k = np.mean(individual_k)

        # Collective K
        all_obs, all_actions = [], []
        for agent in agents:
            if agent.obs_history:
                all_obs.extend(agent.obs_history[-30:])
                all_actions.extend(agent.action_history[-30:])

        if len(all_obs) >= 10:
            all_obs = np.array(all_obs).flatten()
            all_actions = np.array(all_actions).flatten()
            correlation = np.corrcoef(all_obs, all_actions)[0, 1]
            collective_k = abs(correlation) * 2.5 if not np.isnan(correlation) else 0.0
        else:
            collective_k = 0.0

        mean_reward = np.mean(rewards)

        # Corrected K metrics
        epsilon = 0.001
        flexibility = 1.0 / (mean_individual_k + epsilon)
        coordination_ratio = collective_k / (mean_individual_k + epsilon)
        corrected_k = flexibility * coordination_ratio

        return {
            'collective_k': collective_k,
            'mean_individual_k': mean_individual_k,
            'mean_reward': mean_reward,
            'flexibility': flexibility,
            'coordination_ratio': coordination_ratio,
            'corrected_k': corrected_k,
        }

    def run(self):
        """Run all episodes."""
        print(f"Running {self.n_episodes} episodes with {self.n_agents} agents...")

        for ep in range(self.n_episodes):
            # Create fresh agents each episode (no learning across episodes)
            agents = [Agent(i) for i in range(self.n_agents)]
            network = CommunicationNetwork(self.n_agents, self.topology)
            env = MultiAgentEnvironment(self.n_agents)

            result = self.run_episode(agents, network, env)
            self.results.append(result)

            if (ep + 1) % 50 == 0:
                print(f"  Episode {ep + 1}/{self.n_episodes}: "
                      f"Corrected K={result['corrected_k']:.3f}, "
                      f"Reward={result['mean_reward']:.3f}")

    def analyze_results(self):
        """Analyze correlations between K metrics and performance."""
        rewards = np.array([r['mean_reward'] for r in self.results])

        metrics = {
            'Corrected K (flex*coord)': np.array([r['corrected_k'] for r in self.results]),
            'Flexibility (1/K_ind)': np.array([r['flexibility'] for r in self.results]),
            'Coordination (K_coll/K_ind)': np.array([r['coordination_ratio'] for r in self.results]),
            '-K_individual': -np.array([r['mean_individual_k'] for r in self.results]),
            'Original Collective K': np.array([r['collective_k'] for r in self.results]),
            'Original Individual K': np.array([r['mean_individual_k'] for r in self.results]),
        }

        print("\n" + "=" * 70)
        print("CORRECTED K VALIDATION RESULTS")
        print("=" * 70)
        print(f"\nEpisodes: {len(rewards)}")
        print(f"Reward: mean={rewards.mean():.3f}, std={rewards.std():.3f}")
        print("\n" + "-" * 70)
        print("Correlations with Performance:")
        print("-" * 70)

        results = []
        for name, values in metrics.items():
            r, p = stats.pearsonr(values, rewards)
            results.append((name, r, p))

        results.sort(key=lambda x: x[1], reverse=True)

        for name, r, p in results:
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {name:30s}: r = {r:+.4f}, p = {p:.6f} {sig}")

        # Key result
        corrected_k = metrics['Corrected K (flex*coord)']
        r_main, p_main = stats.pearsonr(corrected_k, rewards)

        print("\n" + "=" * 70)
        print("KEY RESULT")
        print("=" * 70)
        print(f"\nCorrected K vs Reward: r = {r_main:+.4f}, p = {p_main:.6f}")

        # 95% CI
        n = len(rewards)
        z = 0.5 * np.log((1 + r_main) / (1 - r_main))
        se = 1 / np.sqrt(n - 3)
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        print(f"95% CI: [{r_lower:.3f}, {r_upper:.3f}]")
        print(f"Variance explained: {r_main**2 * 100:.1f}%")

        if r_main > 0 and p_main < 0.05:
            print("\n✅ SUCCESS: Corrected K shows significant POSITIVE correlation!")
        elif r_main < 0 and p_main < 0.05:
            print("\n⚠️ Correlation still negative - may need different formulation")
        else:
            print("\n❌ Not significant")

        return r_main, p_main

    def save_results(self, output_path: str):
        """Save results to NPZ file."""
        np.savez(output_path,
                 collective_k=np.array([r['collective_k'] for r in self.results]),
                 mean_individual_k=np.array([r['mean_individual_k'] for r in self.results]),
                 mean_reward=np.array([r['mean_reward'] for r in self.results]),
                 flexibility=np.array([r['flexibility'] for r in self.results]),
                 coordination_ratio=np.array([r['coordination_ratio'] for r in self.results]),
                 corrected_k=np.array([r['corrected_k'] for r in self.results]))
        print(f"\nResults saved: {output_path}")


def main():
    print("\n" + "=" * 70)
    print("TRACK D: CORRECTED K-INDEX VALIDATION")
    print("=" * 70)
    print("\nHypothesis: Corrected K (flexibility * coordination) shows r ≈ +0.58")
    print("=" * 70 + "\n")

    # Run experiment
    experiment = CorrectedKValidation(n_agents=4, n_episodes=200)
    experiment.run()
    r, p = experiment.analyze_results()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.save_results(f"track_d_corrected_k_{timestamp}.npz")

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
