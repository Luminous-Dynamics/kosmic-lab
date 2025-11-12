#!/usr/bin/env python3
"""
Track D: Multi-Agent Coordination Experiments

Scientific Question: Does collective intelligence emerge from K-Index optimization?
"""

import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Agent:
    """Single agent in multi-agent system."""
    
    def __init__(self, agent_id: int, obs_dim: int = 10, action_dim: int = 10, capacity: str = "medium"):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_weights = np.random.randn(action_dim, obs_dim + 5) * 0.1
        
        self.obs_history = []
        self.action_history = []
        
    def observe(self, env_state: np.ndarray) -> np.ndarray:
        """Get local observation."""
        obs = env_state + np.random.randn(*env_state.shape) * 0.1
        self.obs_history.append(obs)
        return obs
        
    def act(self, obs: np.ndarray, messages: np.ndarray) -> np.ndarray:
        """Select action."""
        combined = np.concatenate([obs, messages])
        action = np.tanh(self.policy_weights @ combined)
        self.action_history.append(action)
        return action
        
    def create_message(self, obs: np.ndarray) -> np.ndarray:
        """Create message to broadcast."""
        return obs[:5]
        
    def get_k_index(self) -> float:
        """Compute K-Index from history."""
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-50:]).flatten()
        actions = np.array(self.action_history[-50:]).flatten()
        if len(obs) < 2 or len(actions) < 2:
            return 0.0
        correlation = np.corrcoef(obs, actions)[0, 1]
        return abs(correlation) * 2.0


class CommunicationNetwork:
    """Communication topology."""
    
    def __init__(self, n_agents: int, topology: str = "fully_connected", comm_cost: float = 0.1):
        self.n_agents = n_agents
        self.comm_cost = comm_cost
        self.adjacency = self._build_topology(topology)
        
    def _build_topology(self, topology: str) -> np.ndarray:
        """Build adjacency matrix."""
        A = np.zeros((self.n_agents, self.n_agents))
        if topology == "fully_connected":
            A = np.ones((self.n_agents, self.n_agents))
        elif topology == "ring":
            for i in range(self.n_agents):
                A[i, (i+1) % self.n_agents] = 1
                A[i, (i-1) % self.n_agents] = 1
        np.fill_diagonal(A, 0)
        return A
        
    def exchange_messages(self, messages: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """Exchange messages."""
        received = []
        total_cost = 0.0
        for i in range(self.n_agents):
            incoming = [messages[j] for j in range(self.n_agents) if self.adjacency[i, j] > 0]
            if incoming:
                received.append(np.mean(incoming, axis=0))
                total_cost += len(incoming) * self.comm_cost
            else:
                received.append(np.zeros(5))
        return received, total_cost


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
        """Environment step."""
        action_aggregate = np.mean(actions, axis=0)
        self.state += action_aggregate * 0.1
        
        # Rewards
        rewards = []
        for action in actions:
            dist = np.linalg.norm(self.state - self.target)
            coord = -np.linalg.norm(action - action_aggregate)
            rewards.append(-dist + 0.5 * coord)
            
        done = np.linalg.norm(self.state - self.target) < 0.2
        return self.state, rewards, done


class MultiAgentCoordination:
    """Track D runner."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_agents = config['experiment']['n_agents']
        self.episodes = config['experiment']['episodes']
        self.agents = [Agent(i) for i in range(self.n_agents)]
        self.network = CommunicationNetwork(
            self.n_agents,
            config['parameters'].get('network_topology', 'fully_connected'),
            config['parameters'].get('communication_cost', 0.1)
        )
        self.env = MultiAgentEnvironment(self.n_agents)
        self.results = []
        
    def run_episode(self, episode_idx: int) -> Dict:
        """Run one episode."""
        state = self.env.reset()
        for agent in self.agents:
            agent.obs_history = []
            agent.action_history = []
            
        for step in range(200):
            observations = [agent.observe(state) for agent in self.agents]
            messages = [agent.create_message(obs) for agent, obs in zip(self.agents, observations)]
            received_messages, comm_cost = self.network.exchange_messages(messages)
            actions = [agent.act(obs, msg) for agent, obs, msg in zip(self.agents, observations, received_messages)]
            state, rewards, done = self.env.step(actions)
            if done:
                break
                
        individual_k = [agent.get_k_index() for agent in self.agents]
        collective_k = self._compute_collective_k()
        
        return {
            'episode': episode_idx,
            'collective_k': collective_k,
            'mean_individual_k': np.mean(individual_k),
            'individual_k': individual_k,
            'mean_reward': np.mean(rewards)
        }
        
    def _compute_collective_k(self) -> float:
        """Compute collective K-Index."""
        all_obs, all_actions = [], []
        for agent in self.agents:
            if agent.obs_history:
                all_obs.extend(agent.obs_history[-30:])
                all_actions.extend(agent.action_history[-30:])
        if len(all_obs) < 10:
            return 0.0
        all_obs = np.array(all_obs).flatten()
        all_actions = np.array(all_actions).flatten()
        if len(all_obs) < 2 or len(all_actions) < 2:
            return 0.0
        correlation = np.corrcoef(all_obs, all_actions)[0, 1]
        return abs(correlation) * 2.5
        
    def run(self) -> List[Dict]:
        """Run all episodes."""
        logger.info(f"Starting Track D: {self.episodes} episodes, {self.n_agents} agents")
        for ep in range(self.episodes):
            result = self.run_episode(ep)
            self.results.append(result)
            if (ep + 1) % 10 == 0:
                logger.info(f"Episode {ep+1}: Collective K={result['collective_k']:.3f}, Individual K={result['mean_individual_k']:.3f}")
        return self.results
        
    def save_results(self, output_dir: Path):
        """Save results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"track_d_{timestamp}.npz"
        np.savez(filename,
            collective_k=np.array([r['collective_k'] for r in self.results]),
            mean_individual_k=np.array([r['mean_individual_k'] for r in self.results]),
            mean_reward=np.array([r['mean_reward'] for r in self.results]))
        logger.info(f"Saved: {filename}")
        return filename


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Track D: Multi-Agent Coordination")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='logs/track_d')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    experiment = MultiAgentCoordination(config)
    experiment.run()
    experiment.save_results(Path(args.output_dir))
    
    logger.info(f"\nCollective K: {np.mean([r['collective_k'] for r in experiment.results]):.3f}")
    logger.info(f"Individual K: {np.mean([r['mean_individual_k'] for r in experiment.results]):.3f}")


if __name__ == '__main__':
    main()
