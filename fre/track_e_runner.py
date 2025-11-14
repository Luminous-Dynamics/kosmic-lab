#!/usr/bin/env python3
"""
🌱 Track E: Developmental Learning Runner

Tests how learning systems develop coherence over extended time periods.
Compares different learning paradigms: standard RL, curriculum learning,
meta-learning, and full developmental (curriculum + meta).

Scientific Question: How does K-Index evolve during learning, and which
learning paradigms maximize final coherence?
"""

import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class LearningPhase:
    """Represents a phase in the learning schedule."""

    name: str
    lr: float
    exploration: float
    start_episode: int
    end_episode: int


class DevelopmentalEnvironment:
    """Environment with progressive task complexity."""

    def __init__(self, obs_dim: int, action_dim: int, task_complexity: str, max_steps: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.task_complexity = task_complexity
        self.max_steps = max_steps
        self.current_episode = 0
        self.difficulty_level = 1.0

    def reset(self):
        """Reset environment for new episode."""
        # Progressive difficulty
        if self.task_complexity == "progressive":
            self.difficulty_level = 1.0 + (self.current_episode / 50.0) * 2.0

        self.state = np.random.randn(self.obs_dim) * self.difficulty_level
        self.step_count = 0
        return self.state.copy()

    def step(self, action: np.ndarray):
        """Execute action and return next state, reward, done."""
        # Compute reward (higher difficulty = harder to get reward)
        action_quality = np.dot(action, self.state[: self.action_dim]) / (
            self.action_dim * self.difficulty_level
        )
        reward = np.tanh(action_quality)

        # Update state with some dynamics
        self.state = 0.9 * self.state + 0.1 * np.random.randn(self.obs_dim)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.state.copy(), reward, done

    def get_difficulty(self):
        """Return current difficulty level."""
        return self.difficulty_level


class LearningAgent:
    """Agent that learns over time with various paradigms."""

    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # Initialize simple policy network (linear for speed)
        self.weights = np.random.randn(obs_dim, action_dim) * 0.1
        self.bias = np.zeros(action_dim)

        # Learning parameters
        self.learning_rate = 0.001
        self.exploration_rate = 0.3

        # Memory for K-Index computation
        self.obs_history = []
        self.action_history = []

        # Meta-learning parameters (if applicable)
        self.meta_weights = None
        if config.get("meta_learning", False):
            self.meta_weights = self.weights.copy()

        # Curriculum stage
        self.curriculum_stage = 0

    def set_learning_phase(self, phase: LearningPhase):
        """Update learning parameters based on phase."""
        self.learning_rate = phase.lr
        self.exploration_rate = phase.exploration

    def act(self, observation: np.ndarray, explore: bool = True):
        """Select action based on current policy."""
        # Forward pass
        logits = observation @ self.weights + self.bias

        # Add exploration noise
        if explore:
            noise = np.random.randn(self.action_dim) * self.exploration_rate
            action = np.tanh(logits + noise)
        else:
            action = np.tanh(logits)

        # Record for K-Index
        self.obs_history.append(observation.copy())
        self.action_history.append(action.copy())

        return action

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
    ):
        """Update policy based on experience (simplified TD learning)."""
        # Compute TD target
        next_value = np.mean(next_observation @ self.weights + self.bias)
        current_value = np.mean(observation @ self.weights + self.bias)
        td_error = reward + 0.99 * next_value - current_value

        # Gradient update (simplified)
        gradient = np.outer(observation, action) * td_error
        self.weights += self.learning_rate * gradient
        self.bias += self.learning_rate * action * td_error

    def meta_update(self):
        """Meta-learning update (MAML-inspired)."""
        if self.meta_weights is not None:
            # Simple meta-update: blend current weights with meta-weights
            self.meta_weights = 0.95 * self.meta_weights + 0.05 * self.weights
            # Occasionally reset to meta-weights for fast adaptation
            if np.random.rand() < 0.1:
                self.weights = self.meta_weights.copy()

    def advance_curriculum_stage(self):
        """Move to next curriculum stage."""
        self.curriculum_stage += 1

    def get_k_index(self):
        """Compute K-Index from observation-action history."""
        if len(self.obs_history) < 10:
            return 0.0

        # Use last 100 samples
        recent_obs = np.array(self.obs_history[-100:])
        recent_actions = np.array(self.action_history[-100:])

        # Compute norms (compresses dimensionality)
        obs_norms = np.linalg.norm(recent_obs, axis=1)
        action_norms = np.linalg.norm(recent_actions, axis=1)

        if len(obs_norms) < 2 or len(action_norms) < 2:
            return 0.0

        # K-Index as correlation between observation magnitude and action magnitude
        try:
            correlation = np.corrcoef(obs_norms, action_norms)[0, 1]
            k_index = abs(correlation) * 2.0
        except:
            # If correlation fails (e.g., constant values), return 0
            k_index = 0.0

        return k_index


def run_track_e(config_path: str = "fre/configs/track_e_developmental.yaml"):
    """Execute Track E developmental learning experiments."""

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract parameters
    n_episodes = config["experiment"]["n_episodes"]
    env_config = config["environment"]
    agent_config = config["agent"]
    learning_conditions = config["learning_conditions"]

    print(f"\n🌱 Starting Track E: Developmental Learning")
    print(f"Total episodes per condition: {n_episodes}")
    print(f"Learning conditions: {len(learning_conditions)}")
    print(f"Output: {output_dir}\n")

    # Create learning phase schedule
    phases = []
    for phase_config in agent_config["learning_rate_schedule"]:
        phase_name = phase_config["phase"]
        if phase_name == "early":
            start, end = 0, 15
        elif phase_name == "middle":
            start, end = 16, 35
        else:  # late
            start, end = 36, 50

        phases.append(
            LearningPhase(
                name=phase_name,
                lr=phase_config["lr"],
                exploration=phase_config["exploration"],
                start_episode=start,
                end_episode=end,
            )
        )

    # Run each learning condition
    all_results = []

    for condition in learning_conditions:
        condition_name = condition["name"]
        print(f"\n[Condition: {condition_name}]")
        print(f"Algorithm: {condition['algorithm']}")
        print(f"Meta-learning: {condition['meta_learning']}")
        print(f"Curriculum: {condition['curriculum']}")

        # Initialize environment and agent
        env = DevelopmentalEnvironment(
            obs_dim=env_config["obs_dim"],
            action_dim=env_config["action_dim"],
            task_complexity=env_config["task_complexity"],
            max_steps=env_config["max_steps"],
        )

        agent = LearningAgent(
            obs_dim=env_config["obs_dim"], action_dim=env_config["action_dim"], config=condition
        )

        # Track metrics
        k_indices = []
        rewards = []
        difficulties = []
        learning_rates = []
        exploration_rates = []

        # Run episodes
        for episode in range(n_episodes):
            env.current_episode = episode

            # Update learning phase
            current_phase = phases[0]
            for phase in phases:
                if phase.start_episode <= episode <= phase.end_episode:
                    current_phase = phase
                    break
            agent.set_learning_phase(current_phase)

            # Curriculum advancement
            if condition["curriculum"] and episode % 17 == 0 and episode > 0:
                agent.advance_curriculum_stage()

            # Run episode
            state = env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(env_config["max_steps"]):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            # Meta-learning update
            if condition["meta_learning"]:
                agent.meta_update()

            # Compute metrics
            k_index = agent.get_k_index()
            difficulty = env.get_difficulty()

            k_indices.append(k_index)
            rewards.append(episode_reward / episode_steps)  # Normalized
            difficulties.append(difficulty)
            learning_rates.append(agent.learning_rate)
            exploration_rates.append(agent.exploration_rate)

            # Progress
            if (episode + 1) % 10 == 0:
                print(
                    f"  Episode {episode+1}/{n_episodes}: K={k_index:.3f}, R={episode_reward/episode_steps:.3f}, Diff={difficulty:.2f}"
                )

        # Store results
        condition_results = {
            "condition": condition_name,
            "k_indices": np.array(k_indices),
            "rewards": np.array(rewards),
            "difficulties": np.array(difficulties),
            "learning_rates": np.array(learning_rates),
            "exploration_rates": np.array(exploration_rates),
            "final_k": k_indices[-1],
            "max_k": max(k_indices),
            "mean_k": np.mean(k_indices),
            "k_growth_rate": (k_indices[-1] - k_indices[0]) / n_episodes,
        }
        all_results.append(condition_results)

        print(
            f"  ✅ Final K: {k_indices[-1]:.3f}, Max K: {max(k_indices):.3f}, Growth: {condition_results['k_growth_rate']:.4f}"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"track_e_{timestamp}.npz"
    np.savez(
        results_file,
        **{
            result["condition"]: {
                "k_indices": result["k_indices"],
                "rewards": result["rewards"],
                "difficulties": result["difficulties"],
                "learning_rates": result["learning_rates"],
                "exploration_rates": result["exploration_rates"],
            }
            for result in all_results
        },
    )

    print(f"\n✅ Track E complete!")
    print(f"Results saved to: {results_file}")

    # Summary
    print(f"\n📊 Summary:")
    for result in all_results:
        print(f"\n{result['condition']}:")
        print(f"  Final K: {result['final_k']:.3f}")
        print(f"  Max K: {result['max_k']:.3f}")
        print(f"  Mean K: {result['mean_k']:.3f}")
        print(f"  K Growth Rate: {result['k_growth_rate']:.4f}")

    # Determine winner
    best_condition = max(all_results, key=lambda x: x["final_k"])
    print(
        f"\n🏆 Best condition: {best_condition['condition']} (Final K = {best_condition['final_k']:.3f})"
    )

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track E: Developmental Learning")
    parser.add_argument(
        "--config",
        type=str,
        default="fre/configs/track_e_developmental.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    results = run_track_e(args.config)
