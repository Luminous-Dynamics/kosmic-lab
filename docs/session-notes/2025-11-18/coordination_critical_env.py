#!/usr/bin/env python3
"""
Coordination-Critical Environment

Tests flexibility in an environment where coordination explicitly requires
adapting to partner actions (not just aligning with them).

Hypothesis: Flexibility only predicts performance when the task requires it.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy --run "python3 -u coordination_critical_env.py"
"""

import numpy as np
from scipy import stats
from datetime import datetime


class Policy:
    def __init__(self, obs_dim, action_dim):
        self.weights = np.random.randn(action_dim, obs_dim) * 0.3
        self.log_std = np.random.randn(action_dim) * 0.5 - 1
        self.obs_history = []
        self.action_history = []

    def get_action(self, obs):
        mean = np.tanh(self.weights @ obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        action = mean + std * np.random.randn(len(mean))
        action = np.clip(action, -1, 1)
        self.obs_history.append(obs.flatten())
        self.action_history.append(action.flatten())
        return action

    def get_flexibility(self):
        if len(self.obs_history) < 20:
            return 0.0
        obs = np.concatenate(self.obs_history[-50:])
        actions = np.concatenate(self.action_history[-50:])
        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0
        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        return abs(corr) * 2.0 if not np.isnan(corr) else 0.0

    def reset_history(self):
        self.obs_history = []
        self.action_history = []


class CoordinationCriticalEnv:
    """
    Environment where agents must adapt to partner actions.

    Key features:
    - Each agent observes partner's previous actions
    - Reward requires complementary (not identical) actions
    - Rigid policies that ignore partners will fail
    """

    def __init__(self):
        self.n_agents = 4
        self.obs_dim = 9  # 6 state + 3 partner action
        self.action_dim = 3

    def reset(self):
        self.state = np.random.randn(6) * 0.1
        self.target = np.random.randn(6)
        self.steps = 0
        self.prev_actions = [np.zeros(3) for _ in range(4)]

        # Each agent sees state + one partner's previous action
        obs = []
        for i in range(4):
            partner = (i + 1) % 4
            agent_obs = np.concatenate([
                self.state + np.random.randn(6) * 0.1,
                self.prev_actions[partner]
            ])
            obs.append(agent_obs)
        return obs

    def step(self, actions):
        # Complementary coordination: pairs must sum to target
        # Agent 0+1 should sum to target[:3], Agent 2+3 should sum to target[3:]

        pair1_sum = actions[0] + actions[1]
        pair2_sum = actions[2] + actions[3]

        target_short = self.target[:3]
        target_short2 = self.target[3:6] if len(self.target) >= 6 else self.target[:3]

        # Reward for complementary coordination
        error1 = np.linalg.norm(pair1_sum - target_short)
        error2 = np.linalg.norm(pair2_sum - target_short2)

        # Also reward for using partner information
        responsiveness = 0
        for i in range(4):
            partner = (i + 1) % 4
            # Reward if action correlates with partner's previous action
            partner_influence = np.abs(np.dot(actions[i], self.prev_actions[partner]))
            responsiveness += partner_influence

        reward = -error1 - error2 + responsiveness * 0.1

        self.prev_actions = [a.copy() for a in actions]
        self.steps += 1
        done = self.steps >= 50

        # Next observations
        obs = []
        for i in range(4):
            partner = (i + 1) % 4
            agent_obs = np.concatenate([
                self.state + np.random.randn(6) * 0.1,
                self.prev_actions[partner]
            ])
            obs.append(agent_obs)

        return obs, [reward] * 4, done


class SimpleAlignmentEnv:
    """Control: environment where simple alignment works."""

    def __init__(self):
        self.n_agents = 4
        self.obs_dim = 6
        self.action_dim = 3

    def reset(self):
        self.state = np.random.randn(6) * 0.1
        self.target = np.random.randn(6)
        self.steps = 0
        return [self.state + np.random.randn(6) * 0.1 for _ in range(4)]

    def step(self, actions):
        # Simple: reward for all agents taking same action
        mean_action = np.mean(actions, axis=0)
        alignment = -np.std([np.linalg.norm(a - mean_action) for a in actions])

        # Also task reward
        if len(mean_action) < 6:
            mean_action = np.pad(mean_action, (0, 6 - len(mean_action)))
        self.state += mean_action * 0.1
        dist = -np.linalg.norm(self.state - self.target)

        reward = dist + alignment
        self.steps += 1
        done = self.steps >= 50

        return [self.state + np.random.randn(6) * 0.1 for _ in range(4)], [reward] * 4, done


def eval_in_env(env_class, n_teams=200):
    """Evaluate random policies in an environment."""
    results = []

    for _ in range(n_teams):
        env = env_class()
        obs_dim = env.obs_dim if hasattr(env, 'obs_dim') else 6
        policies = [Policy(obs_dim, 3) for _ in range(4)]

        obs_list = env.reset()
        for p in policies:
            p.reset_history()
        total = 0

        for step in range(50):
            actions = [p.get_action(obs) for p, obs in zip(policies, obs_list)]
            obs_list, rews, done = env.step(actions)
            total += rews[0]
            if done:
                break

        flex = np.mean([p.get_flexibility() for p in policies])
        results.append({'flex': flex, 'reward': total})

    return results


def main():
    print("\n" + "=" * 60)
    print("COORDINATION-CRITICAL ENVIRONMENT TEST")
    print("=" * 60)

    n_teams = 300

    # ================================================================
    # Test 1: Coordination-Critical Environment
    # ================================================================
    print("\n" + "-" * 60)
    print("Environment 1: Coordination-Critical (n=300)")
    print("-" * 60)
    print("(Agents must adapt to partner actions)")

    results_coord = eval_in_env(CoordinationCriticalEnv, n_teams)
    flex_coord = np.array([x['flex'] for x in results_coord])
    reward_coord = np.array([x['reward'] for x in results_coord])
    r_coord, p_coord = stats.pearsonr(flex_coord, reward_coord)

    # Effect size
    median = np.median(flex_coord)
    high = reward_coord[flex_coord > median]
    low = reward_coord[flex_coord <= median]
    d_coord = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    sig = '***' if p_coord < 0.001 else '**' if p_coord < 0.01 else '*' if p_coord < 0.05 else 'ns'
    print(f"\n  r = {r_coord:+.3f}, p = {p_coord:.4f} {sig}")
    print(f"  d = {d_coord:+.3f}")

    # ================================================================
    # Test 2: Simple Alignment Environment (Control)
    # ================================================================
    print("\n" + "-" * 60)
    print("Environment 2: Simple Alignment (n=300)")
    print("-" * 60)
    print("(Agents just need to match each other)")

    results_simple = eval_in_env(SimpleAlignmentEnv, n_teams)
    flex_simple = np.array([x['flex'] for x in results_simple])
    reward_simple = np.array([x['reward'] for x in results_simple])
    r_simple, p_simple = stats.pearsonr(flex_simple, reward_simple)

    median = np.median(flex_simple)
    high = reward_simple[flex_simple > median]
    low = reward_simple[flex_simple <= median]
    d_simple = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

    sig = '***' if p_simple < 0.001 else '**' if p_simple < 0.01 else '*' if p_simple < 0.05 else 'ns'
    print(f"\n  r = {r_simple:+.3f}, p = {p_simple:.4f} {sig}")
    print(f"  d = {d_simple:+.3f}")

    # ================================================================
    # Comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\n{'Environment':<25} {'r':>10} {'d':>10} {'p':>12}")
    print("-" * 60)
    print(f"{'Coordination-Critical':<25} {r_coord:>+10.3f} {d_coord:>+10.3f} {p_coord:>12.4f}")
    print(f"{'Simple Alignment':<25} {r_simple:>+10.3f} {d_simple:>+10.3f} {p_simple:>12.4f}")

    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    if r_coord > r_simple + 0.1 and p_coord < 0.05:
        print("✓ Flexibility predicts performance in coordination-critical tasks")
        print("  But not in simple alignment tasks")
        print("  → Flexibility is task-specific")
    elif p_coord < 0.05 and p_simple < 0.05:
        print("✓ Flexibility predicts performance in both environments")
        print("  → Flexibility is generally beneficial")
    elif p_coord >= 0.05 and p_simple >= 0.05:
        print("✗ Flexibility does not predict performance in either environment")
        print("  → Need to reconsider the metric or hypothesis")
    else:
        print("→ Mixed results - need further investigation")

    # Quartile analysis for coordination-critical
    print("\nQuartile Analysis (Coordination-Critical):")
    q1, q2, q3 = np.percentile(flex_coord, [25, 50, 75])
    for lo, hi, name in [
        (flex_coord.min(), q1, "Q1"),
        (q1, q2, "Q2"),
        (q2, q3, "Q3"),
        (q3, flex_coord.max(), "Q4")
    ]:
        mask = (flex_coord >= lo) & (flex_coord <= hi)
        mean_r = reward_coord[mask].mean()
        print(f"  {name}: {mean_r:.1f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coordination_critical_{timestamp}.npz"
    np.savez(filename,
             flex_coord=flex_coord, reward_coord=reward_coord,
             flex_simple=flex_simple, reward_simple=reward_simple,
             r_coord=r_coord, r_simple=r_simple)
    print(f"\nSaved: {filename}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
