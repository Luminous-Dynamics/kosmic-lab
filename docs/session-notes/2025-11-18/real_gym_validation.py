#!/usr/bin/env python3
"""
Real Gym Environments Validation

Tests flexibility metric in actual Gym environments with proper RL training.
Validates that patterns from simulated environments hold in real ones.

Run with:
    nix-shell -p python3 python3Packages.numpy python3Packages.scipy python3Packages.gymnasium --run "python3 real_gym_validation.py"
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from datetime import datetime

# Try to import gymnasium
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("WARNING: gymnasium not available. Using simulated environments.")


class FlexibilityTracker:
    """Track flexibility during episodes."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.obs_history = []
        self.action_history = []

    def record(self, obs: np.ndarray, action):
        """Record observation-action pair."""
        self.obs_history.append(obs.flatten())
        if isinstance(action, (int, np.integer)):
            self.action_history.append(np.array([action]))
        else:
            self.action_history.append(np.array(action).flatten())

    def get_flexibility(self) -> float:
        """Calculate flexibility from recent history."""
        if len(self.obs_history) < 10:
            return 0.0

        obs = np.concatenate(self.obs_history[-self.window_size:])
        actions = np.concatenate(self.action_history[-self.window_size:])

        min_len = min(len(obs), len(actions))
        if min_len < 2:
            return 0.0

        corr = np.corrcoef(obs[:min_len], actions[:min_len])[0, 1]
        if np.isnan(corr):
            return 0.0

        k_individual = abs(corr) * 2.0
        return -k_individual  # Flexibility = -K

    def reset(self):
        """Clear history."""
        self.obs_history = []
        self.action_history = []


class SimplePolicy:
    """Simple learnable policy for testing."""

    def __init__(self, obs_dim: int, action_dim: int, discrete: bool = True):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.weights = np.random.randn(action_dim, obs_dim) * 0.1
        self.learning_rate = 0.01

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Select action based on observation."""
        logits = self.weights @ obs
        if self.discrete:
            # Softmax for discrete actions
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            return np.random.choice(self.action_dim, p=probs)
        else:
            return np.tanh(logits)

    def update(self, obs: np.ndarray, action, reward: float):
        """Simple policy gradient update."""
        if self.discrete:
            action_vec = np.zeros(self.action_dim)
            action_vec[action] = 1
        else:
            action_vec = action

        # Reinforce-style update
        grad = np.outer(action_vec, obs)
        self.weights += self.learning_rate * reward * grad


def run_gym_experiment(env_name: str, n_episodes: int = 200) -> Dict:
    """Run experiment on a Gym environment."""

    if GYM_AVAILABLE:
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'n'):
            action_dim = env.action_space.n
            discrete = True
        else:
            action_dim = env.action_space.shape[0]
            discrete = False
    else:
        # Simulated environment
        env = None
        if env_name == "CartPole-v1":
            obs_dim, action_dim, discrete = 4, 2, True
        elif env_name == "LunarLander-v2":
            obs_dim, action_dim, discrete = 8, 4, True
        elif env_name == "Acrobot-v1":
            obs_dim, action_dim, discrete = 6, 3, True
        elif env_name == "Pendulum-v1":
            obs_dim, action_dim, discrete = 3, 1, False
        else:
            obs_dim, action_dim, discrete = 4, 2, True

    results = []

    for ep in range(n_episodes):
        policy = SimplePolicy(obs_dim, action_dim, discrete)
        tracker = FlexibilityTracker()

        if GYM_AVAILABLE:
            obs, _ = env.reset()
        else:
            obs = np.random.randn(obs_dim) * 0.1

        total_reward = 0
        max_steps = 500 if "LunarLander" in env_name else 200

        for step in range(max_steps):
            action = policy.act(obs)
            tracker.record(obs, action)

            if GYM_AVAILABLE:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                # Simulated dynamics
                next_obs = obs + np.random.randn(obs_dim) * 0.1
                reward = -np.linalg.norm(obs) * 0.1 + np.random.randn() * 0.1
                done = step > 100 and np.random.random() < 0.05

            policy.update(obs, action, reward)
            total_reward += reward
            obs = next_obs

            if done:
                break

        flexibility = tracker.get_flexibility()
        results.append({
            'flexibility': flexibility,
            'reward': total_reward,
            'steps': step + 1
        })

    if GYM_AVAILABLE:
        env.close()

    return {
        'env_name': env_name,
        'results': results,
        'gym_available': GYM_AVAILABLE
    }


def analyze_results(experiment_results: Dict) -> Dict:
    """Analyze flexibility-reward correlation."""
    results = experiment_results['results']

    flexibility = np.array([r['flexibility'] for r in results])
    rewards = np.array([r['reward'] for r in results])

    # Pearson correlation
    r, p = stats.pearsonr(flexibility, rewards)

    # Spearman for robustness
    rho, p_spearman = stats.spearmanr(flexibility, rewards)

    # Effect size (split by median)
    median_flex = np.median(flexibility)
    high_flex = rewards[flexibility > median_flex]
    low_flex = rewards[flexibility <= median_flex]

    if len(high_flex) > 1 and len(low_flex) > 1:
        pooled_std = np.sqrt((high_flex.std()**2 + low_flex.std()**2) / 2)
        if pooled_std > 0:
            d = (high_flex.mean() - low_flex.mean()) / pooled_std
        else:
            d = 0
    else:
        d = 0

    return {
        'r': r,
        'p': p,
        'rho': rho,
        'd': d,
        'n': len(results),
        'mean_flex': flexibility.mean(),
        'mean_reward': rewards.mean()
    }


def main():
    print("\n" + "=" * 70)
    print("REAL GYM ENVIRONMENTS VALIDATION")
    print("=" * 70)

    if not GYM_AVAILABLE:
        print("\n⚠️  Gymnasium not available - using simulated environments")
        print("    Install with: nix-shell -p python3Packages.gymnasium")

    environments = [
        "CartPole-v1",
        "LunarLander-v2",
        "Acrobot-v1",
        "Pendulum-v1"
    ]

    all_results = {}

    for env_name in environments:
        print(f"\nTesting {env_name}...")
        experiment = run_gym_experiment(env_name, n_episodes=200)
        analysis = analyze_results(experiment)
        all_results[env_name] = {**experiment, **analysis}

        sig = '***' if analysis['p'] < 0.001 else '**' if analysis['p'] < 0.01 else '*' if analysis['p'] < 0.05 else ''
        print(f"  r = {analysis['r']:+.3f}, p = {analysis['p']:.2e} {sig}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Environment':<20} {'r':>8} {'p':>12} {'d':>8} {'Sig':>6}")
    print("-" * 70)

    for env_name, data in all_results.items():
        sig = '***' if data['p'] < 0.001 else '**' if data['p'] < 0.01 else '*' if data['p'] < 0.05 else 'ns'
        print(f"{env_name:<20} {data['r']:>+8.3f} {data['p']:>12.2e} {data['d']:>+8.2f} {sig:>6}")

    # Meta-analysis
    print("\n" + "-" * 70)
    all_r = [data['r'] for data in all_results.values()]
    mean_r = np.mean(all_r)
    print(f"Mean r across environments: {mean_r:+.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"real_gym_validation_{timestamp}.npz"

    np.savez(filename,
             environments=list(all_results.keys()),
             correlations=[data['r'] for data in all_results.values()],
             p_values=[data['p'] for data in all_results.values()],
             effect_sizes=[data['d'] for data in all_results.values()],
             gym_available=GYM_AVAILABLE)

    print(f"\nResults saved: {filename}")
    print("=" * 70 + "\n")

    return all_results


if __name__ == '__main__':
    main()
