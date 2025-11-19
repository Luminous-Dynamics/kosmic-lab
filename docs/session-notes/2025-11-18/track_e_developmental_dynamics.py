#!/usr/bin/env python3
"""
Track E: Developmental Dynamics of Flexibility

How does flexibility develop during learning?
Does early flexibility predict final coordination?
Is there an optimal flexibility level?

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u track_e_developmental_dynamics.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class DevelopingAgent:
    """Agent that learns via REINFORCE with flexibility tracking."""

    def __init__(self, agent_id):
        self.id = agent_id
        self.policy_weights = np.random.randn(10, 15) * 0.1
        self.obs_history = []
        self.action_history = []
        self.log_probs = []
        self.rewards = []
        self.flexibility_trajectory = []

    def act(self, obs, messages):
        combined = np.concatenate([obs, messages])
        mean = self.policy_weights @ combined
        # Add noise for exploration
        action = mean + np.random.randn(10) * 0.3
        action = np.tanh(action)

        # Compute log prob (simplified Gaussian)
        log_prob = -0.5 * np.sum((action - np.tanh(mean))**2)

        self.obs_history.append(obs)
        self.action_history.append(action)
        self.log_probs.append(log_prob)

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

    def update(self, learning_rate=0.01):
        """REINFORCE update."""
        if not self.rewards:
            return

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update weights (simplified gradient)
        for i, (obs, action, log_prob, ret) in enumerate(zip(
            self.obs_history, self.action_history, self.log_probs, returns
        )):
            if i < len(self.obs_history) - 1:
                # Approximate gradient
                gradient = np.outer(action, np.concatenate([obs, np.zeros(5)]))
                self.policy_weights += learning_rate * ret * gradient[:, :15]

        # Record flexibility before clearing
        self.flexibility_trajectory.append(self.get_flexibility())

        # Clear episode data
        self.obs_history = []
        self.action_history = []
        self.log_probs = []
        self.rewards = []

    def create_message(self, obs):
        return obs[:5]


class Network:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.adj = np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def exchange(self, messages):
        return [np.mean([messages[j] for j in range(self.n_agents)
                        if self.adj[i,j] > 0], axis=0)
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


def run_training(n_agents=4, n_episodes=200, n_steps=200, learning_rate=0.01):
    """Train agents and track flexibility development."""
    agents = [DevelopingAgent(i) for i in range(n_agents)]
    network = Network(n_agents)
    env = Environment(n_agents)

    episode_rewards = []
    episode_flexibilities = []

    for ep in range(n_episodes):
        state = env.reset()

        # Clear histories for new episode
        for a in agents:
            a.obs_history = []
            a.action_history = []
            a.log_probs = []
            a.rewards = []

        total_reward = 0
        for step in range(n_steps):
            observations = [state + np.random.randn(10) * 0.1 for _ in range(n_agents)]
            messages = [a.create_message(o) for a, o in zip(agents, observations)]
            received = network.exchange(messages)
            actions = [a.act(o, m) for a, o, m in zip(agents, observations, received)]
            state, reward = env.step(actions)

            for a in agents:
                a.rewards.append(reward)
            total_reward += reward

        # Update all agents
        for a in agents:
            a.update(learning_rate)

        # Record episode metrics
        episode_rewards.append(total_reward)
        mean_flex = np.mean([a.flexibility_trajectory[-1] if a.flexibility_trajectory else 0
                            for a in agents])
        episode_flexibilities.append(mean_flex)

    return episode_rewards, episode_flexibilities, agents


def main():
    print("\n" + "=" * 70)
    print("TRACK E: DEVELOPMENTAL DYNAMICS OF FLEXIBILITY")
    print("=" * 70)

    np.random.seed(42)

    # ===== EXPERIMENT 1: Flexibility Development Curve =====
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: FLEXIBILITY DEVELOPMENT DURING LEARNING")
    print("-" * 70)

    n_runs = 5
    n_episodes = 150

    all_rewards = []
    all_flexibilities = []

    print(f"\nTraining {n_runs} independent runs...")

    for run in range(n_runs):
        np.random.seed(run * 100)
        rewards, flexibilities, _ = run_training(n_episodes=n_episodes)
        all_rewards.append(rewards)
        all_flexibilities.append(flexibilities)
        print(f"  Run {run + 1}/{n_runs} complete")

    # Average across runs
    mean_rewards = np.mean(all_rewards, axis=0)
    mean_flex = np.mean(all_flexibilities, axis=0)

    # Analyze development phases
    early = slice(0, 30)
    mid = slice(50, 80)
    late = slice(120, 150)

    print("\nFlexibility Development Phases:")
    print(f"  Early (0-30):    {np.mean(mean_flex[early]):+.3f}")
    print(f"  Mid (50-80):     {np.mean(mean_flex[mid]):+.3f}")
    print(f"  Late (120-150):  {np.mean(mean_flex[late]):+.3f}")

    # ===== EXPERIMENT 2: Early Flexibility Predicts Final Performance =====
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: EARLY FLEXIBILITY → FINAL PERFORMANCE")
    print("-" * 70)

    n_teams = 30
    early_flex_scores = []
    final_rewards = []

    print(f"\nTraining {n_teams} independent teams...")

    for team in range(n_teams):
        np.random.seed(team * 1000)
        rewards, flexibilities, _ = run_training(n_episodes=100)

        # Early flexibility (first 20 episodes)
        early_flex = np.mean(flexibilities[:20])
        # Final performance (last 20 episodes)
        final_perf = np.mean(rewards[-20:])

        early_flex_scores.append(early_flex)
        final_rewards.append(final_perf)

    r, p = stats.pearsonr(early_flex_scores, final_rewards)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

    print(f"\nEarly flexibility → Final reward: r = {r:+.3f}, p = {p:.4f}{sig}")

    if r > 0.3:
        print("  → Early flexibility PREDICTS final coordination success")
    elif r > 0.15:
        print("  → Early flexibility weakly predicts success")
    else:
        print("  → Early flexibility does NOT predict success")

    # ===== EXPERIMENT 3: Flexibility-Performance Correlation Over Development =====
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: FLEX-REWARD CORRELATION OVER DEVELOPMENT")
    print("-" * 70)

    checkpoints = [10, 30, 50, 70, 100, 130]

    print("\nCorrelation at each training stage:")

    # Train many teams and measure correlation at each checkpoint
    n_teams = 50
    checkpoint_correlations = []

    for checkpoint in checkpoints:
        flex_at_checkpoint = []
        reward_at_checkpoint = []

        for team in range(n_teams):
            np.random.seed(team * 500)
            rewards, flexibilities, _ = run_training(n_episodes=checkpoint + 10)

            # Use data around checkpoint
            flex_at_checkpoint.append(np.mean(flexibilities[checkpoint-5:checkpoint+5]))
            reward_at_checkpoint.append(np.mean(rewards[checkpoint-5:checkpoint+5]))

        r, p = stats.pearsonr(flex_at_checkpoint, reward_at_checkpoint)
        checkpoint_correlations.append(r)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  Episode {checkpoint:3d}: r = {r:+.3f}, p = {p:.4f}{sig}")

    # ===== EXPERIMENT 4: Optimal Flexibility Level =====
    print("\n" + "-" * 70)
    print("EXPERIMENT 4: OPTIMAL FLEXIBILITY LEVEL")
    print("-" * 70)

    # Train to convergence and look at final flex-reward relationship
    n_teams = 40
    final_flexibilities = []
    final_performances = []

    print(f"\nTraining {n_teams} teams to convergence...")

    for team in range(n_teams):
        np.random.seed(team * 2000)
        rewards, flexibilities, _ = run_training(n_episodes=200, learning_rate=0.005)

        final_flex = np.mean(flexibilities[-30:])
        final_perf = np.mean(rewards[-30:])

        final_flexibilities.append(final_flex)
        final_performances.append(final_perf)

    # Check for quadratic relationship (too much flexibility might hurt)
    flex_arr = np.array(final_flexibilities)
    perf_arr = np.array(final_performances)

    # Linear fit
    r_linear, p_linear = stats.pearsonr(flex_arr, perf_arr)

    # Check for optimum by binning
    sorted_idx = np.argsort(flex_arr)
    thirds = np.array_split(sorted_idx, 3)

    low_flex_perf = np.mean(perf_arr[thirds[0]])
    mid_flex_perf = np.mean(perf_arr[thirds[1]])
    high_flex_perf = np.mean(perf_arr[thirds[2]])

    print(f"\nPerformance by flexibility tertile:")
    print(f"  Low flexibility:  {low_flex_perf:.1f}")
    print(f"  Mid flexibility:  {mid_flex_perf:.1f}")
    print(f"  High flexibility: {high_flex_perf:.1f}")

    if mid_flex_perf > low_flex_perf and mid_flex_perf > high_flex_perf:
        print("\n  → INVERTED-U: Optimal flexibility is MODERATE")
    elif high_flex_perf > mid_flex_perf > low_flex_perf:
        print("\n  → MONOTONIC: More flexibility is better")
    else:
        print("\n  → No clear optimum pattern")

    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("ANALYSIS: DEVELOPMENTAL DYNAMICS")
    print("=" * 70)

    # Flexibility trajectory
    flex_change = np.mean(mean_flex[late]) - np.mean(mean_flex[early])
    print(f"\n1. Flexibility Development: Δ = {flex_change:+.3f}")
    if flex_change > 0.1:
        print("   → Flexibility INCREASES during learning")
    elif flex_change < -0.1:
        print("   → Flexibility DECREASES during learning")
    else:
        print("   → Flexibility stays relatively STABLE")

    # Predictive value
    print(f"\n2. Early Flexibility Prediction: r = {r:+.3f}")
    if r > 0.3:
        print("   → Early flexibility is a STRONG predictor")
    elif r > 0.15:
        print("   → Early flexibility is a MODERATE predictor")
    else:
        print("   → Early flexibility is NOT predictive")

    # Correlation development
    corr_change = checkpoint_correlations[-1] - checkpoint_correlations[0]
    print(f"\n3. Flex-Reward Correlation Development: Δ = {corr_change:+.3f}")
    if corr_change > 0.2:
        print("   → Correlation STRENGTHENS during learning")
    elif corr_change < -0.2:
        print("   → Correlation WEAKENS during learning")
    else:
        print("   → Correlation remains STABLE")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print("\n1. Developmental Trajectory:")
    print(f"   Early: {np.mean(mean_flex[early]):+.3f} → Late: {np.mean(mean_flex[late]):+.3f}")

    print(f"\n2. Predictive Power: r = {r:+.3f}")

    print("\n3. Optimal Level:")
    if mid_flex_perf > max(low_flex_perf, high_flex_perf):
        print("   → Moderate flexibility is optimal (inverted-U)")
    else:
        print("   → More flexibility generally better (monotonic)")

    print("\n4. Correlation Evolution:")
    print(f"   Episode 10: r = {checkpoint_correlations[0]:+.3f}")
    print(f"   Episode 130: r = {checkpoint_correlations[-1]:+.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track_e_developmental_{timestamp}.npz"
    np.savez(filename,
             mean_rewards=mean_rewards,
             mean_flex=mean_flex,
             early_flex_scores=early_flex_scores,
             final_rewards=final_rewards,
             checkpoint_correlations=checkpoint_correlations,
             final_flexibilities=final_flexibilities,
             final_performances=final_performances)
    print(f"\nSaved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
