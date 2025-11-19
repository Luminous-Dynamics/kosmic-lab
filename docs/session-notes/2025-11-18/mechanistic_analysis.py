#!/usr/bin/env python3
"""
Mechanistic Analysis: WHY Flexibility Predicts Coordination

Moves from correlation to explanation by analyzing:
1. Adaptation speed - how quickly agents match partner behavior
2. Response diversity - variety of actions to similar observations
3. Partner modeling - evidence of tracking partner states

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 mechanistic_analysis.py"
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from datetime import datetime


class AdaptiveAgent:
    """Agent with measurable adaptation mechanisms."""

    def __init__(self, agent_id: int, obs_dim: int = 10, action_dim: int = 10):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_weights = np.random.randn(action_dim, obs_dim + 5) * 0.1

        # Tracking for mechanistic analysis
        self.obs_history = []
        self.action_history = []
        self.partner_action_history = []
        self.adaptation_times = []

    def observe(self, env_state: np.ndarray) -> np.ndarray:
        obs = env_state + np.random.randn(*env_state.shape) * 0.1
        self.obs_history.append(obs)
        return obs

    def act(self, obs: np.ndarray, partner_action: np.ndarray) -> np.ndarray:
        # Combine observation and partner's last action
        combined = np.concatenate([obs, partner_action])
        action = np.tanh(self.policy_weights @ combined)
        self.action_history.append(action)
        self.partner_action_history.append(partner_action.copy())
        return action

    def get_flexibility(self) -> float:
        if len(self.obs_history) < 10:
            return 0.0
        obs = np.array(self.obs_history[-50:]).flatten()
        actions = np.array(self.action_history[-50:]).flatten()
        if len(obs) < 2 or len(actions) < 2:
            return 0.0
        min_len = min(len(obs), len(actions))
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        if np.isnan(corr):
            return 0.0
        return -abs(corr) * 2.0

    def get_adaptation_speed(self) -> float:
        """Measure how quickly agent matches partner's action changes."""
        if len(self.action_history) < 10 or len(self.partner_action_history) < 10:
            return 0.0

        actions = np.array(self.action_history[-50:])
        partner_actions = np.array(self.partner_action_history[-50:])

        # Cross-correlation to find lag
        if len(actions) < 3:
            return 0.0

        # Compute action changes
        my_changes = np.diff(actions, axis=0)
        partner_changes = np.diff(partner_actions, axis=0)

        # Correlation at different lags
        best_lag = 0
        best_corr = 0

        for lag in range(min(5, len(my_changes))):
            try:
                if lag == 0:
                    a = my_changes.flatten()
                    b = partner_changes.flatten()
                else:
                    a = my_changes[lag:].flatten()
                    b = partner_changes[:-lag].flatten()

                min_len = min(len(a), len(b))
                if min_len < 2:
                    continue
                c = np.corrcoef(a[:min_len], b[:min_len])[0, 1]

                if not np.isnan(c) and abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag
            except:
                continue

        # Adaptation speed = inverse of lag (faster = higher)
        return 1.0 / (best_lag + 1)

    def get_response_diversity(self) -> float:
        """Measure variety of actions to similar observations."""
        if len(self.obs_history) < 20:
            return 0.0

        obs = np.array(self.obs_history[-50:])
        actions = np.array(self.action_history[-50:])

        # Find similar observations
        diversities = []
        for i in range(len(obs)):
            # Find observations similar to this one
            distances = np.linalg.norm(obs - obs[i], axis=1)
            similar_idx = np.where(distances < np.percentile(distances, 20))[0]

            if len(similar_idx) > 1:
                # Measure action diversity for similar observations
                similar_actions = actions[similar_idx]
                diversity = np.std(similar_actions)
                diversities.append(diversity)

        return np.mean(diversities) if diversities else 0.0

    def get_partner_modeling_score(self) -> float:
        """Measure evidence of tracking partner state."""
        if len(self.action_history) < 10:
            return 0.0

        actions = np.array(self.action_history[-50:])
        partner_actions = np.array(self.partner_action_history[-50:])

        # Mutual information approximation
        # High MI = actions depend on partner's actions
        if len(actions) < 2:
            return 0.0

        # Simple correlation as MI proxy
        a = actions.flatten()
        b = partner_actions.flatten()
        min_len = min(len(a), len(b))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(a[:min_len], b[:min_len])[0, 1]
        if np.isnan(corr):
            return 0.0

        return abs(corr)

    def reset(self):
        self.obs_history = []
        self.action_history = []
        self.partner_action_history = []


def run_mechanistic_episode(agents: List[AdaptiveAgent], topology="fully_connected") -> Dict:
    """Run episode with mechanistic tracking."""
    n_agents = len(agents)

    # Adjacency
    A = np.ones((n_agents, n_agents))
    np.fill_diagonal(A, 0)

    state = np.random.randn(10) * 0.1
    target = np.random.randn(10)

    for agent in agents:
        agent.reset()

    # Initialize partner actions
    prev_actions = [np.zeros(10) for _ in range(n_agents)]

    for step in range(200):
        observations = [agent.observe(state) for agent in agents]

        # Each agent sees aggregate of neighbors' previous actions
        new_actions = []
        for i, agent in enumerate(agents):
            neighbor_actions = [prev_actions[j] for j in range(n_agents) if A[i, j] > 0]
            if neighbor_actions:
                partner_action = np.mean(neighbor_actions, axis=0)[:5]
            else:
                partner_action = np.zeros(5)

            action = agent.act(observations[i], partner_action)
            new_actions.append(action)

        prev_actions = new_actions
        action_agg = np.mean(new_actions, axis=0)
        state += action_agg * 0.1

        if np.linalg.norm(state - target) < 0.2:
            break

    # Collect mechanistic metrics
    flexibility = np.mean([agent.get_flexibility() for agent in agents])
    adaptation_speed = np.mean([agent.get_adaptation_speed() for agent in agents])
    response_diversity = np.mean([agent.get_response_diversity() for agent in agents])
    partner_modeling = np.mean([agent.get_partner_modeling_score() for agent in agents])

    dist = np.linalg.norm(state - target)
    reward = -dist

    return {
        'flexibility': flexibility,
        'adaptation_speed': adaptation_speed,
        'response_diversity': response_diversity,
        'partner_modeling': partner_modeling,
        'reward': reward
    }


def main():
    print("\n" + "=" * 70)
    print("MECHANISTIC ANALYSIS: WHY FLEXIBILITY PREDICTS COORDINATION")
    print("=" * 70)

    n_episodes = 200
    n_agents = 4

    results = []

    print("\nRunning mechanistic analysis episodes...")
    for ep in range(n_episodes):
        agents = [AdaptiveAgent(i) for i in range(n_agents)]
        result = run_mechanistic_episode(agents)
        results.append(result)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")

    # Extract arrays
    flexibility = np.array([r['flexibility'] for r in results])
    adaptation_speed = np.array([r['adaptation_speed'] for r in results])
    response_diversity = np.array([r['response_diversity'] for r in results])
    partner_modeling = np.array([r['partner_modeling'] for r in results])
    reward = np.array([r['reward'] for r in results])

    # Analysis
    print("\n" + "=" * 70)
    print("MECHANISTIC CORRELATIONS")
    print("=" * 70)

    metrics = {
        'Flexibility': flexibility,
        'Adaptation Speed': adaptation_speed,
        'Response Diversity': response_diversity,
        'Partner Modeling': partner_modeling
    }

    print(f"\n{'Metric':<25} {'→ Reward r':>12} {'p':>12} {'Sig':>6}")
    print("-" * 70)

    correlations = {}
    for name, values in metrics.items():
        r, p = stats.pearsonr(values, reward)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"{name:<25} {r:>+12.3f} {p:>12.2e} {sig:>6}")
        correlations[name] = {'r': r, 'p': p}

    # Inter-metric correlations
    print("\n" + "-" * 70)
    print("Inter-Metric Correlations (understanding mechanisms)")
    print("-" * 70)

    print(f"\n{'Metric Pair':<40} {'r':>8} {'p':>12}")
    print("-" * 70)

    pairs = [
        ('Flexibility', 'Adaptation Speed'),
        ('Flexibility', 'Response Diversity'),
        ('Flexibility', 'Partner Modeling'),
        ('Adaptation Speed', 'Response Diversity'),
    ]

    for m1, m2 in pairs:
        r, p = stats.pearsonr(metrics[m1], metrics[m2])
        print(f"{m1} ↔ {m2:<20} {r:>+8.3f} {p:>12.2e}")

    # Mediation analysis (simplified)
    print("\n" + "=" * 70)
    print("MEDIATION ANALYSIS")
    print("=" * 70)

    # Does adaptation speed mediate flexibility → reward?
    # Step 1: Flexibility → Reward (total effect)
    r_total, _ = stats.pearsonr(flexibility, reward)

    # Step 2: Flexibility → Adaptation Speed
    r_flex_adapt, _ = stats.pearsonr(flexibility, adaptation_speed)

    # Step 3: Adaptation Speed → Reward (controlling for flexibility)
    # Simple approximation: partial correlation
    resid_adapt = adaptation_speed - np.mean(adaptation_speed)
    resid_reward = reward - np.mean(reward)
    r_adapt_reward, _ = stats.pearsonr(resid_adapt, resid_reward)

    # Indirect effect (Sobel approximation)
    indirect = r_flex_adapt * r_adapt_reward

    print(f"\nFlexibility → Reward (total):     r = {r_total:+.3f}")
    print(f"Flexibility → Adaptation Speed:   r = {r_flex_adapt:+.3f}")
    print(f"Adaptation Speed → Reward:        r = {r_adapt_reward:+.3f}")
    print(f"Indirect effect (via adaptation): r = {indirect:+.3f}")
    print(f"Direct effect remaining:          r = {r_total - indirect:+.3f}")

    if abs(indirect) > 0.1:
        print("\n✓ Adaptation speed partially mediates flexibility → reward")
    else:
        print("\n✗ Weak mediation through adaptation speed")

    # Summary
    print("\n" + "=" * 70)
    print("MECHANISTIC SUMMARY")
    print("=" * 70)

    best_mechanism = max(correlations.items(), key=lambda x: abs(x[1]['r']))
    print(f"\nBest predictor: {best_mechanism[0]} (r = {best_mechanism[1]['r']:+.3f})")

    print("\nInterpretation:")
    if correlations['Adaptation Speed']['r'] > 0.3:
        print("  • Flexible agents adapt faster to partner behavior changes")
    if correlations['Response Diversity']['r'] > 0.3:
        print("  • Flexible agents show more varied responses to similar situations")
    if correlations['Partner Modeling']['r'] > 0.3:
        print("  • Flexible agents better track and respond to partner actions")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mechanistic_analysis_{timestamp}.npz"

    np.savez(filename,
             flexibility=flexibility,
             adaptation_speed=adaptation_speed,
             response_diversity=response_diversity,
             partner_modeling=partner_modeling,
             reward=reward)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
