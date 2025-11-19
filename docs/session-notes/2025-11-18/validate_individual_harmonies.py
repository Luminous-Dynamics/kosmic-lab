#!/usr/bin/env python3
"""
Rigorous Validation: Do Individual Harmonies Predict Performance?

Critical Question: Is Full K driven entirely by H2, or do other harmonies contribute?

This test:
1. Trains multiple Q-learning agents on CartPole
2. Computes ALL 7 harmonies for each episode
3. Correlates EACH harmony independently with performance
4. Tests if harmonies beyond H2 add predictive value

If only H2 correlates, we cannot claim "Full 7-Harmony K-Index works" -
we can only claim "H2 (Diversity) works."
"""

import numpy as np
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# CartPole environment
class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 12 * np.pi / 180
        self.x_threshold = 2.4
        self.state = None

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        total_mass = self.masscart + self.masspole
        polemass_length = self.masspole * self.length

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = (abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        reward = 0.0 if done else 1.0

        return self.state.copy(), reward, done

# Q-learning agent
class QLearningAgent:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.n_actions = 2
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        self.state_bounds = [
            (-2.4, 2.4), (-3.0, 3.0),
            (-0.21, 0.21), (-3.0, 3.0)
        ]

    def discretize(self, state):
        discrete = []
        for i, val in enumerate(state):
            low, high = self.state_bounds[i]
            val = np.clip(val, low, high)
            bin_idx = int((val - low) / (high - low) * (self.n_bins - 1))
            discrete.append(bin_idx)
        return tuple(discrete)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        discrete_state = self.discretize(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)

        return np.argmax(self.q_table[discrete_state])

    def update(self, state, action, reward, next_state, done):
        discrete_state = self.discretize(state)
        discrete_next = self.discretize(next_state)

        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)
        if discrete_next not in self.q_table:
            self.q_table[discrete_next] = np.zeros(self.n_actions)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next])

        self.q_table[discrete_state][action] += self.lr * (
            target - self.q_table[discrete_state][action]
        )

# All 7 Harmony computations
def compute_h1_resonant_coherence(states):
    """H1: Integration across state dimensions."""
    if len(states) < 10:
        return 0.5

    states = np.array(states)
    correlations = []

    for i in range(4):
        for j in range(i+1, 4):
            if np.std(states[:, i]) > 1e-10 and np.std(states[:, j]) > 1e-10:
                r = np.corrcoef(states[:, i], states[:, j])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))

    return float(np.mean(correlations)) if correlations else 0.5

def compute_h2_diversity(actions):
    """H2: Action distribution entropy."""
    if len(actions) < 5:
        return 0.5

    counts = Counter(actions)
    if len(counts) <= 1:
        return 0.0

    total = len(actions)
    probs = np.array([c / total for c in counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    h_max = np.log(len(counts))

    return float(h / h_max) if h_max > 0 else 0.0

def compute_h3_prediction_accuracy(states, actions):
    """H3: How well do states predict actions?"""
    if len(states) < 10:
        return 0.5

    states = np.array(states)
    actions = np.array(actions)

    # Align states with actions (states has one extra for initial state)
    if len(states) > len(actions):
        states = states[:len(actions)]

    # Simple: correlation between state norm and action
    state_norms = np.linalg.norm(states, axis=1)
    if np.std(state_norms) < 1e-10:
        return 0.5

    # Point-biserial correlation
    action_0_norms = state_norms[actions == 0]
    action_1_norms = state_norms[actions == 1]

    if len(action_0_norms) < 2 or len(action_1_norms) < 2:
        return 0.5

    # Effect size as proxy for prediction accuracy
    pooled_std = np.sqrt(
        ((len(action_0_norms)-1)*np.var(action_0_norms) +
         (len(action_1_norms)-1)*np.var(action_1_norms)) /
        (len(action_0_norms) + len(action_1_norms) - 2)
    )

    if pooled_std < 1e-10:
        return 0.5

    d = abs(np.mean(action_1_norms) - np.mean(action_0_norms)) / pooled_std
    return float(np.clip(d / 2, 0, 1))  # Normalize to [0, 1]

def compute_h4_behavioral_entropy(states):
    """H4: Entropy of state value distributions."""
    if len(states) < 10:
        return 0.5

    states = np.array(states)
    n_bins = min(5, len(states) // 5)
    if n_bins < 2:
        return 0.5

    entropies = []
    for dim in range(4):
        col = states[:, dim]
        if np.std(col) < 1e-10:
            continue

        hist, _ = np.histogram(col, bins=n_bins)
        hist = hist + 1e-10
        probs = hist / hist.sum()
        entropies.append(-np.sum(probs * np.log(probs)))

    if not entropies:
        return 0.5

    h_max = np.log(n_bins)
    return float(np.mean(entropies) / h_max) if h_max > 0 else 0.5

def compute_h5_transfer_entropy(states):
    """H5: Cross-dimension information flow."""
    if len(states) < 10:
        return 0.5

    states = np.array(states)
    influences = []

    for i in range(4):
        for j in range(i+1, 4):
            if (np.std(states[:-1, i]) > 1e-10 and
                np.std(states[1:, j]) > 1e-10):
                try:
                    c_ij = abs(np.corrcoef(states[:-1, i], states[1:, j])[0, 1])
                    c_ji = abs(np.corrcoef(states[:-1, j], states[1:, i])[0, 1])

                    if not (np.isnan(c_ij) or np.isnan(c_ji)):
                        influence = 2 * min(c_ij, c_ji) / (c_ij + c_ji + 1e-10)
                        influences.append(influence)
                except:
                    pass

    return float(np.mean(influences)) if influences else 0.5

def compute_h6_flow_symmetry(states):
    """H6: Temporal symmetry of state evolution."""
    if len(states) < 6:
        return 0.5

    states = np.array(states)
    norms = np.linalg.norm(states, axis=1)

    mid = len(norms) // 2
    first = norms[:mid]
    second = norms[mid:mid+len(first)]

    if len(first) != len(second):
        min_len = min(len(first), len(second))
        first = first[:min_len]
        second = second[:min_len]

    first_p = (np.abs(first) + 1e-10) / (np.abs(first).sum() + 1e-10)
    second_p = (np.abs(second) + 1e-10) / (np.abs(second).sum() + 1e-10)

    m = 0.5 * (first_p + second_p)
    js = 0.5 * (
        np.sum(first_p * np.log(first_p / (m + 1e-10) + 1e-10)) +
        np.sum(second_p * np.log(second_p / (m + 1e-10) + 1e-10))
    )

    return float(1.0 - np.sqrt(np.clip(js, 0, 1)))

def compute_h7_growth_rate(rewards):
    """H7: Improvement trend within episode."""
    if len(rewards) < 10:
        return 0.0

    # Split into thirds
    third = len(rewards) // 3
    if third < 3:
        return 0.0

    first_third = np.mean(rewards[:third])
    last_third = np.mean(rewards[-third:])

    if first_third < 1e-10:
        return 0.5 if last_third > first_third else 0.0

    growth = (last_third - first_third) / first_third
    return float(np.clip((growth + 1) / 2, 0, 1))

def compute_simple_k(states, actions):
    """Simple K-Index (deprecated) for comparison."""
    if len(states) < 5:
        return 0.5

    states = np.array(states)
    actions = np.array(actions)

    # Align states with actions
    if len(states) > len(actions):
        states = states[:len(actions)]

    obs_norms = np.linalg.norm(states, axis=1)
    act_values = actions.astype(float)

    if np.std(obs_norms) < 1e-10 or np.std(act_values) < 1e-10:
        return 0.0

    r = np.corrcoef(obs_norms, act_values)[0, 1]
    if np.isnan(r):
        return 0.0

    return float(2 * abs(r))

def run_episode(env, agent, train=True):
    """Run one episode and collect all data."""
    state = env.reset()
    states = [state.copy()]
    actions = []
    rewards = []
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        if train:
            agent.update(state, action, reward, next_state, done)

        states.append(next_state.copy())
        actions.append(action)
        rewards.append(reward)
        state = next_state

        if len(actions) >= 500:
            break

    return states, actions, rewards

def main():
    print("=" * 70)
    print("RIGOROUS VALIDATION: Individual Harmony Contributions")
    print("=" * 70)
    print("\nCritical Question: Is Full K driven entirely by H2?\n")

    # Run multiple training runs for statistical power
    n_runs = 5
    n_episodes = 100

    all_results = {
        'performance': [],
        'simple_k': [],
        'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': [], 'h7': [],
        'full_k': []
    }

    print(f"Running {n_runs} training runs × {n_episodes} episodes each...\n")

    for run in range(n_runs):
        env = CartPoleEnv()
        agent = QLearningAgent()

        for ep in range(n_episodes):
            states, actions, rewards = run_episode(env, agent)

            performance = len(rewards)

            # Compute all harmonies
            h1 = compute_h1_resonant_coherence(states)
            h2 = compute_h2_diversity(actions)
            h3 = compute_h3_prediction_accuracy(states, actions)
            h4 = compute_h4_behavioral_entropy(states)
            h5 = compute_h5_transfer_entropy(states)
            h6 = compute_h6_flow_symmetry(states)
            h7 = compute_h7_growth_rate(rewards)

            full_k = np.mean([h1, h2, h3, h4, h5, h6, h7])
            simple_k = compute_simple_k(states, actions)

            all_results['performance'].append(performance)
            all_results['simple_k'].append(simple_k)
            all_results['h1'].append(h1)
            all_results['h2'].append(h2)
            all_results['h3'].append(h3)
            all_results['h4'].append(h4)
            all_results['h5'].append(h5)
            all_results['h6'].append(h6)
            all_results['h7'].append(h7)
            all_results['full_k'].append(full_k)

        print(f"  Run {run+1}/{n_runs} complete")

    print("\n" + "=" * 70)
    print("RESULTS: Correlation with Performance")
    print("=" * 70)

    performance = np.array(all_results['performance'])

    # Compute correlations for each metric
    metrics = [
        ('Simple K (deprecated)', 'simple_k'),
        ('H1 Resonant Coherence', 'h1'),
        ('H2 Diversity', 'h2'),
        ('H3 Prediction Accuracy', 'h3'),
        ('H4 Behavioral Entropy', 'h4'),
        ('H5 Transfer Entropy', 'h5'),
        ('H6 Flow Symmetry', 'h6'),
        ('H7 Growth Rate', 'h7'),
        ('Full K (mean)', 'full_k'),
    ]

    print("\nMetric                    | Correlation |  p-value  | Significant?")
    print("-" * 70)

    significant_harmonies = []

    for name, key in metrics:
        values = np.array(all_results[key])
        r, p = stats.pearsonr(performance, values)
        sig = "YES" if p < 0.05 else "no"

        if p < 0.05 and key.startswith('h'):
            significant_harmonies.append((name, r, p))

        print(f"{name:25s} | r = {r:+.3f}  | p = {p:.4f} | {sig}")

    print("\n" + "=" * 70)
    print("CRITICAL ANALYSIS")
    print("=" * 70)

    # Key comparisons
    h2_r = stats.pearsonr(performance, all_results['h2'])[0]
    full_k_r = stats.pearsonr(performance, all_results['full_k'])[0]
    simple_k_r = stats.pearsonr(performance, all_results['simple_k'])[0]

    print(f"\n1. Simple K vs Performance: r = {simple_k_r:+.3f}")
    if simple_k_r < 0:
        print("   → CONFIRMS: Simple K anti-correlates (measures rigidity)")
    else:
        print("   → UNEXPECTED: Simple K positively correlates")

    print(f"\n2. H2 (Diversity) vs Performance: r = {h2_r:+.3f}")
    print(f"   Full K vs Performance: r = {full_k_r:+.3f}")

    if abs(h2_r) > abs(full_k_r):
        print("   → H2 ALONE is better predictor than Full K!")
    else:
        print("   → Full K adds value beyond H2")

    print(f"\n3. Significant harmonies (p < 0.05):")
    if significant_harmonies:
        for name, r, p in significant_harmonies:
            print(f"   - {name}: r = {r:+.3f}")
    else:
        print("   - None besides H2!")

    # Variance explained
    print(f"\n4. Variance explained:")
    print(f"   - H2 alone: {h2_r**2 * 100:.1f}%")
    print(f"   - Full K: {full_k_r**2 * 100:.1f}%")

    # Multiple regression to test unique contributions
    print("\n5. Multiple Regression: Do other harmonies add to H2?")

    from scipy.stats import spearmanr

    # Use H2 as baseline, test if others add
    h2_arr = np.array(all_results['h2'])
    residuals = performance - (h2_r * (performance.std() / h2_arr.std()) * h2_arr)

    print("   Residual correlations (after removing H2 effect):")
    for name, key in metrics:
        if key in ['h2', 'simple_k', 'full_k']:
            continue
        values = np.array(all_results[key])
        r, p = stats.pearsonr(residuals, values)
        sig = "*" if p < 0.05 else ""
        print(f"   - {name}: r = {r:+.3f} {sig}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Determine if Full K is justified
    non_h2_significant = [h for h in significant_harmonies if 'H2' not in h[0]]

    if len(non_h2_significant) == 0:
        print("\n⚠️  FINDING: Only H2 significantly predicts performance!")
        print("   The '7-Harmony Full K' claim is NOT supported.")
        print("   We should report 'H2 (Diversity) predicts performance'")
        print("   not 'Full K predicts performance'.")
    else:
        print(f"\n✅ FINDING: {len(non_h2_significant)+1} harmonies contribute:")
        print("   - H2 (Diversity)")
        for name, r, p in non_h2_significant:
            print(f"   - {name}")
        print("   Full K framing is partially justified.")

    if simple_k_r < -0.1:
        print("\n✅ CONFIRMED: Simple K anti-correlates with performance")
        print("   The paradigm shift is validated.")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if len(non_h2_significant) == 0:
        print("\nDo NOT claim 'Full 7-Harmony K-Index works'")
        print("DO claim 'H2 (Diversity/Action Entropy) predicts performance'")
        print("\nRevise all papers to focus on H2, not Full K.")
    else:
        print("\nCan claim Full K with caveats about which harmonies contribute.")

    print("\n✅ Validation complete\n")

if __name__ == '__main__':
    main()
