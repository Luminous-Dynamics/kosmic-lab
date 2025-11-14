#!/usr/bin/env python3
"""
🌊 Track D: Parameter Sweep Runner

Systematically tests multi-agent coordination across parameter space to find
conditions where collective intelligence emerges.

Scientific Question: Under what conditions does Collective K > Individual K?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime
from track_d_runner import Agent, CommunicationNetwork, MultiAgentEnvironment
import itertools


def run_parameter_sweep(config_path="configs/track_d_sweep.yaml"):
    """Execute parameter sweep across all conditions."""

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sweep_params = config["parameter_sweep"]
    fixed_params = config["fixed"]
    base_episodes = config["experiment"]["base_episodes"]

    # Create output directory
    output_dir = Path(config.get("output_dir", "logs/track_d/parameter_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all parameter combinations
    comm_costs = sweep_params["communication_costs"]
    topologies = sweep_params["network_topologies"]
    capacities = sweep_params["agent_capacities"]
    n_agents_list = sweep_params["n_agents"]

    total_conditions = len(comm_costs) * len(topologies) * len(capacities) * len(n_agents_list)
    print(f"\n🌊 Starting Track D Parameter Sweep")
    print(f"Total conditions: {total_conditions}")
    print(f"Episodes per condition: {base_episodes}")
    print(f"Total episodes: {total_conditions * base_episodes}\n")

    results = []
    condition_idx = 0

    # Sweep through all combinations
    for comm_cost, topology, capacity, n_agents in itertools.product(
        comm_costs, topologies, capacities, n_agents_list
    ):
        condition_idx += 1
        print(
            f"[{condition_idx}/{total_conditions}] Running: cost={comm_cost}, topology={topology}, capacity={capacity}, n_agents={n_agents}"
        )

        # Run episodes for this condition
        collective_ks = []
        individual_ks = []
        rewards = []

        for episode in range(base_episodes):
            # Initialize agents
            agents = [Agent(i, capacity=capacity) for i in range(n_agents)]

            # Create communication network
            network = CommunicationNetwork(n_agents, topology, comm_cost)

            # Create environment
            env = MultiAgentEnvironment(n_agents)

            # Run episode
            state = env.reset()
            episode_rewards = []

            for step in range(fixed_params["max_steps"]):
                # Agents observe
                observations = [agent.observe(state) for agent in agents]

                # Communication
                messages = [agent.create_message(observations[i]) for i, agent in enumerate(agents)]

                # Receive messages - CORRECTED METHOD NAME
                received_messages, comm_cost_incurred = network.exchange_messages(messages)

                # Agents act
                actions = [
                    agent.act(observations[i], received_messages[i])
                    for i, agent in enumerate(agents)
                ]

                # Environment step
                state, step_rewards, done = env.step(actions)
                episode_rewards.extend(step_rewards)

                if done:
                    break

            # Compute K-Indices
            individual_k_values = [agent.get_k_index() for agent in agents]
            collective_k = compute_collective_k(agents)

            collective_ks.append(collective_k)
            individual_ks.append(np.mean(individual_k_values))
            rewards.append(np.mean(episode_rewards))

        # Aggregate results for this condition
        mean_collective_k = np.mean(collective_ks)
        mean_individual_k = np.mean(individual_ks)
        emergence_ratio = mean_collective_k / mean_individual_k if mean_individual_k > 0 else 0

        result = {
            "communication_cost": comm_cost,
            "network_topology": topology,
            "agent_capacity": capacity,
            "n_agents": n_agents,
            "mean_collective_k": mean_collective_k,
            "std_collective_k": np.std(collective_ks),
            "mean_individual_k": mean_individual_k,
            "std_individual_k": np.std(individual_ks),
            "emergence_ratio": emergence_ratio,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "n_episodes": base_episodes,
        }

        results.append(result)

        print(f"  → Collective K: {mean_collective_k:.3f} ± {np.std(collective_ks):.3f}")
        print(f"  → Individual K: {mean_individual_k:.3f} ± {np.std(individual_ks):.3f}")
        print(f"  → Emergence Ratio: {emergence_ratio:.3f}")
        print()

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"parameter_sweep_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Parameter sweep complete!")
    print(f"Results saved to: {csv_path}")
    print(f"\n📊 Summary Statistics:")
    print(f"Best emergence ratio: {df['emergence_ratio'].max():.3f}")
    print(
        f"  (at: cost={df.loc[df['emergence_ratio'].idxmax(), 'communication_cost']}, "
        f"topology={df.loc[df['emergence_ratio'].idxmax(), 'network_topology']})"
    )
    print(f"Highest collective K: {df['mean_collective_k'].max():.3f}")
    print(
        f"  (at: cost={df.loc[df['mean_collective_k'].idxmax(), 'communication_cost']}, "
        f"topology={df.loc[df['mean_collective_k'].idxmax(), 'network_topology']})"
    )

    return df


def compute_collective_k(agents):
    """Compute collective K-Index from joint observations and actions."""
    all_obs = []
    all_actions = []

    for agent in agents:
        if len(agent.obs_history) >= 10:
            all_obs.append(np.array(agent.obs_history[-50:]))
            all_actions.append(np.array(agent.action_history[-50:]))

    if len(all_obs) == 0:
        return 0.0

    # Concatenate across agents
    joint_obs = np.concatenate([obs.flatten() for obs in all_obs])
    joint_actions = np.concatenate([act.flatten() for act in all_actions])

    if len(joint_obs) < 2 or len(joint_actions) < 2:
        return 0.0

    # Collective K = correlation between joint observations and joint actions
    correlation = np.corrcoef(joint_obs, joint_actions)[0, 1]
    return abs(correlation) * 2.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track D Parameter Sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/track_d_sweep.yaml",
        help="Path to sweep configuration file",
    )
    args = parser.parse_args()

    results_df = run_parameter_sweep(args.config)
