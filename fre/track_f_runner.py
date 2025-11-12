#!/usr/bin/env python3
"""
Track F: Adversarial Robustness Testing (CORRECTED FGSM)
Tests K-Index stability under various adversarial perturbations.

CRITICAL FIX: Uses proper gradient-based FGSM from fre/attacks/fgsm.py
Formula: x' = x + ε × sign(∇_x L(x,y))
"""

import numpy as np
import yaml
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ✅ PATCH 1: Add PyTorch and Phase 1 imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Phase 1 modules
from fre.attacks.fgsm import fgsm_observation, sanity_check_loss_increases
from fre.metrics.k_index import k_index, k_index_robust
from fre.metrics.k_lag import k_lag
from fre.analysis.partial_corr import k_partial_reward
from fre.analysis.nulls_fdr import null_k_distributions

@dataclass
class AdversarialCondition:
    """Configuration for an adversarial attack condition."""
    name: str
    perturbation_type: str
    perturbation_strength: float
    perturbation_frequency: float
    target: str
    description: str

# ✅ PATCH 2: Add TorchPolicyWrapper for gradient computation
class TorchPolicyWrapper(nn.Module):
    """Wraps numpy policy weights as PyTorch module for FGSM."""

    def __init__(self, policy_weights: np.ndarray):
        super().__init__()
        self.weights = nn.Parameter(torch.from_numpy(policy_weights).float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: action = tanh(W @ obs)"""
        return torch.tanh(torch.matmul(self.weights, obs))

class AdversarialEnvironment:
    """Environment that can apply various adversarial perturbations."""

    def __init__(self, obs_dim: int, action_dim: int, max_steps: int,
                 base_difficulty: float, condition: AdversarialCondition):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.base_difficulty = base_difficulty
        self.condition = condition

        self.state = None
        self.step_count = 0
        self.current_episode = 0

    def reset(self):
        """Reset environment for new episode."""
        self.state = np.random.randn(self.obs_dim) * self.base_difficulty
        self.step_count = 0
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Get observation, potentially with adversarial perturbation."""
        obs = self.state.copy()

        # Apply observation perturbations if applicable
        # NOTE: gradient_based (FGSM) is handled in run_episode_with_fgsm(), not here
        if self.condition.target == "observations":
            if self.condition.perturbation_type != "gradient_based":
                if np.random.random() < self.condition.perturbation_frequency:
                    obs = self.apply_perturbation(obs, self.condition.perturbation_type)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return next state, reward, done."""
        # Potentially perturb action
        if self.condition.target == "actions":
            if np.random.random() < self.condition.perturbation_frequency:
                action = self.apply_perturbation(action, self.condition.perturbation_type)

        # Compute reward based on observation-action alignment
        action_quality = np.dot(action, self.state[:self.action_dim]) / (
            self.action_dim * self.base_difficulty
        )
        reward = np.tanh(action_quality)

        # Potentially spoof reward
        if self.condition.target == "rewards":
            if np.random.random() < self.condition.perturbation_frequency:
                if self.condition.perturbation_type == "sign_flip":
                    if np.random.random() < self.condition.perturbation_strength:
                        reward = -reward

        # Update state
        self.state = 0.9 * self.state + 0.1 * np.random.randn(self.obs_dim)
        self.step_count += 1

        done = self.step_count >= self.max_steps

        return self.get_observation(), reward, done

    # ✅ PATCH 3: Replace apply_perturbation with correct gradient-based FGSM
    def apply_perturbation(self, data: np.ndarray, perturbation_type: str) -> np.ndarray:
        """Apply specific perturbation to data.

        NOTE: For gradient_based (FGSM), this is a placeholder.
        The actual FGSM is applied in run_episode_with_fgsm() using proper gradients.
        """
        perturbed = data.copy()
        strength = self.condition.perturbation_strength

        if perturbation_type == "gaussian_noise":
            # Add Gaussian noise
            noise = np.random.randn(*data.shape) * strength * np.abs(data).mean()
            perturbed = data + noise

        elif perturbation_type == "random_flip":
            # Randomly flip some dimensions
            n_flip = int(len(data) * strength)
            flip_indices = np.random.choice(len(data), n_flip, replace=False)
            perturbed[flip_indices] = -perturbed[flip_indices]

        elif perturbation_type == "gradient_based":
            # FGSM handled separately in run_episode_with_fgsm()
            # This branch should not be reached in corrected code
            raise NotImplementedError(
                "gradient_based perturbation should use run_episode_with_fgsm()"
            )

        elif perturbation_type == "none":
            # No perturbation
            pass

        return perturbed

class RobustLearner:
    """Agent that learns under adversarial conditions."""

    def __init__(self, obs_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Simple policy: linear mapping with tanh activation
        self.policy_weights = np.random.randn(action_dim, obs_dim) * 0.1

        # History for K-Index computation
        self.obs_history = []
        self.action_history = []
        self.reward_history = []  # Added for partial correlation

    def act(self, observation: np.ndarray, exploration_noise: float = 0.1) -> np.ndarray:
        """Select action based on observation."""
        # Policy forward pass
        action = np.tanh(self.policy_weights @ observation)

        # Add exploration noise
        action = action + np.random.randn(self.action_dim) * exploration_noise
        action = np.clip(action, -1, 1)

        # Record for K-Index
        self.obs_history.append(observation.copy())
        self.action_history.append(action.copy())

        return action

    def update(self, reward: float):
        """Simple policy gradient update."""
        self.reward_history.append(reward)

        if len(self.obs_history) < 2:
            return

        # Get last observation and action
        obs = self.obs_history[-1]
        action = self.action_history[-1]

        # Simple gradient: increase weight if reward positive, decrease if negative
        gradient = np.outer(action, obs) * reward
        self.policy_weights += self.learning_rate * gradient

    def get_k_index(self) -> float:
        """Compute K-Index from observation-action history."""
        if len(self.obs_history) < 10:
            return 0.0

        # Use recent history
        recent_obs = np.array(self.obs_history[-100:]) if len(self.obs_history) >= 100 else np.array(self.obs_history)
        recent_actions = np.array(self.action_history[-100:]) if len(self.action_history) >= 100 else np.array(self.action_history)

        if len(recent_obs) < 2:
            return 0.0

        # Compute norms (compress dimensionality)
        obs_norms = np.linalg.norm(recent_obs, axis=1)
        action_norms = np.linalg.norm(recent_actions, axis=1)

        if len(obs_norms) < 2 or len(action_norms) < 2:
            return 0.0

        # Use Phase 1 k_index function with bounds checking
        try:
            k = k_index(obs_norms, action_norms)
        except:
            k = 0.0

        return k

# ✅ PATCH 4: Add run_episode_with_fgsm() for correct gradient-based FGSM
def run_episode_with_fgsm(env: AdversarialEnvironment, agent: RobustLearner,
                          exploration_noise: float = 0.1,
                          epsilon: float = 0.15) -> Dict:
    """Run episode with correct gradient-based FGSM.

    Uses fre.attacks.fgsm for proper FGSM: x' = x + ε × sign(∇_x L(x,y))
    """
    obs = env.reset()
    done = False
    episode_reward = 0.0
    k_indices = []

    # FGSM sanity check logs
    base_losses = []
    adv_losses = []

    # Create PyTorch policy wrapper
    torch_policy = TorchPolicyWrapper(agent.policy_weights)
    loss_fn = nn.MSELoss()  # Simple task loss

    while not done:
        # Apply FGSM if this is an adversarial episode
        if env.condition.perturbation_type == "gradient_based":
            if np.random.random() < env.condition.perturbation_frequency:
                # Convert to torch
                obs_tensor = torch.from_numpy(obs).float()

                # Get base loss
                with torch.no_grad():
                    base_action = torch_policy(obs_tensor)
                    base_loss = loss_fn(base_action, torch.zeros_like(base_action))
                    base_losses.append(base_loss.item())

                # Apply FGSM
                target = torch.zeros(agent.action_dim)  # Simple target
                adv_obs_tensor = fgsm_observation(
                    torch_policy, obs_tensor, target, loss_fn, epsilon
                )
                obs = adv_obs_tensor.numpy()

                # Verify loss increased
                with torch.no_grad():
                    adv_action = torch_policy(adv_obs_tensor)
                    adv_loss = loss_fn(adv_action, torch.zeros_like(adv_action))
                    adv_losses.append(adv_loss.item())

        # Agent acts (numpy policy)
        action = agent.act(obs, exploration_noise)

        # Environment steps
        next_obs, reward, done = env.step(action)

        # Agent updates
        agent.update(reward)

        # Track metrics
        episode_reward += reward
        k_index_val = agent.get_k_index()
        k_indices.append(k_index_val)

        obs = next_obs

    # Final metrics
    final_k = agent.get_k_index()
    k_variance = np.var(k_indices) if len(k_indices) > 0 else 0.0

    # Compute robust K-Index variants
    if len(agent.obs_history) >= 10:
        recent_obs = np.array(agent.obs_history[-100:])
        recent_actions = np.array(agent.action_history[-100:])
        obs_norms = np.linalg.norm(recent_obs, axis=1)
        action_norms = np.linalg.norm(recent_actions, axis=1)

        robust_k = k_index_robust(obs_norms, action_norms)

        # Partial correlation (reward independence)
        if len(agent.reward_history) >= 10:
            recent_rewards = np.array(agent.reward_history[-100:])
            partial_result = k_partial_reward(
                obs_norms[-len(recent_rewards):],
                action_norms[-len(recent_rewards):],
                recent_rewards
            )
            k_raw = partial_result['k_raw']
            k_partial = partial_result['k_partial']
            delta = partial_result['delta']
        else:
            k_raw = robust_k['k_pearson']
            k_partial = k_raw
            delta = 0.0
    else:
        robust_k = {'k_pearson': final_k, 'k_pearson_z': final_k, 'k_spearman': final_k}
        k_raw = final_k
        k_partial = final_k
        delta = 0.0

    return {
        'final_k': final_k,
        'mean_k': np.mean(k_indices) if len(k_indices) > 0 else 0.0,
        'k_variance': k_variance,
        'episode_reward': episode_reward,
        'k_history': k_indices,
        # Robust variants
        'k_pearson': robust_k['k_pearson'],
        'k_pearson_z': robust_k['k_pearson_z'],
        'k_spearman': robust_k['k_spearman'],
        # Partial correlation
        'k_raw': k_raw,
        'k_partial': k_partial,
        'k_delta': delta,
        # FGSM sanity check
        'base_losses': base_losses,
        'adv_losses': adv_losses,
        'fgsm_increased': sum(1 for b, a in zip(base_losses, adv_losses) if a >= b) if base_losses else 0,
        'fgsm_total': len(base_losses)
    }

def run_episode(env: AdversarialEnvironment, agent: RobustLearner,
                exploration_noise: float = 0.1) -> Dict:
    """Run one complete episode (non-FGSM conditions)."""
    obs = env.reset()
    done = False
    episode_reward = 0.0
    k_indices = []

    while not done:
        # Agent acts
        action = agent.act(obs, exploration_noise)

        # Environment steps
        next_obs, reward, done = env.step(action)

        # Agent updates
        agent.update(reward)

        # Track metrics
        episode_reward += reward
        k_index_val = agent.get_k_index()
        k_indices.append(k_index_val)

        obs = next_obs

    # Final metrics
    final_k = agent.get_k_index()
    k_variance = np.var(k_indices) if len(k_indices) > 0 else 0.0

    # Compute robust variants
    if len(agent.obs_history) >= 10:
        recent_obs = np.array(agent.obs_history[-100:])
        recent_actions = np.array(agent.action_history[-100:])
        obs_norms = np.linalg.norm(recent_obs, axis=1)
        action_norms = np.linalg.norm(recent_actions, axis=1)

        robust_k = k_index_robust(obs_norms, action_norms)

        # Partial correlation
        if len(agent.reward_history) >= 10:
            recent_rewards = np.array(agent.reward_history[-100:])
            partial_result = k_partial_reward(
                obs_norms[-len(recent_rewards):],
                action_norms[-len(recent_rewards):],
                recent_rewards
            )
            k_raw = partial_result['k_raw']
            k_partial = partial_result['k_partial']
            delta = partial_result['delta']
        else:
            k_raw = robust_k['k_pearson']
            k_partial = k_raw
            delta = 0.0
    else:
        robust_k = {'k_pearson': final_k, 'k_pearson_z': final_k, 'k_spearman': final_k}
        k_raw = final_k
        k_partial = final_k
        delta = 0.0

    return {
        'final_k': final_k,
        'mean_k': np.mean(k_indices) if len(k_indices) > 0 else 0.0,
        'k_variance': k_variance,
        'episode_reward': episode_reward,
        'k_history': k_indices,
        'k_pearson': robust_k['k_pearson'],
        'k_pearson_z': robust_k['k_pearson_z'],
        'k_spearman': robust_k['k_spearman'],
        'k_raw': k_raw,
        'k_partial': k_partial,
        'k_delta': delta,
        'base_losses': [],
        'adv_losses': [],
        'fgsm_increased': 0,
        'fgsm_total': 0
    }

# ✅ PATCH 5: Enhanced run_condition with detailed per-episode logging
def run_condition(condition: AdversarialCondition, config: Dict,
                  baseline_k: Optional[float] = None) -> Dict:
    """Run all episodes for one adversarial condition with enhanced logging."""
    print(f"\n{'='*80}")
    print(f"Running condition: {condition.name}")
    print(f"Description: {condition.description}")
    print(f"Perturbation: {condition.perturbation_type}, strength={condition.perturbation_strength}, freq={condition.perturbation_frequency}")
    print(f"{'='*80}\n")

    # Extract configuration
    env_config = config['environment']
    agent_config = config['agent']
    n_episodes = config['experiment']['n_episodes']

    # Create environment and agent
    env = AdversarialEnvironment(
        obs_dim=env_config['obs_dim'],
        action_dim=env_config['action_dim'],
        max_steps=env_config['max_steps'],
        base_difficulty=env_config['base_difficulty'],
        condition=condition
    )

    agent = RobustLearner(
        obs_dim=env_config['obs_dim'],
        action_dim=env_config['action_dim'],
        learning_rate=agent_config['learning_rate']
    )

    # Run episodes (use FGSM runner for gradient_based)
    results = []
    for episode in range(n_episodes):
        if condition.perturbation_type == "gradient_based":
            episode_result = run_episode_with_fgsm(
                env, agent, agent_config['exploration_noise'],
                epsilon=condition.perturbation_strength
            )
        else:
            episode_result = run_episode(env, agent, agent_config['exploration_noise'])

        results.append(episode_result)

        if (episode + 1) % config['experiment']['save_interval'] == 0:
            k_robust_str = f"K_p={episode_result['k_pearson']:.3f}, K_s={episode_result['k_spearman']:.3f}"
            partial_str = f"K_partial={episode_result['k_partial']:.3f} (Δ={episode_result['k_delta']:.3f})"
            print(f"  Episode {episode+1}/{n_episodes}: K={episode_result['final_k']:.4f}, {k_robust_str}, {partial_str}, R={episode_result['episode_reward']:.2f}")

            # FGSM sanity check
            if episode_result['fgsm_total'] > 0:
                fgsm_ok_pct = 100.0 * episode_result['fgsm_increased'] / episode_result['fgsm_total']
                print(f"    FGSM sanity: {episode_result['fgsm_increased']}/{episode_result['fgsm_total']} increased loss ({fgsm_ok_pct:.1f}%)")

    # Aggregate results
    final_ks = [r['final_k'] for r in results]
    mean_ks = [r['mean_k'] for r in results]
    k_variances = [r['k_variance'] for r in results]
    rewards = [r['episode_reward'] for r in results]
    k_pearsons = [r['k_pearson'] for r in results]
    k_spearmans = [r['k_spearman'] for r in results]
    k_deltas = [r['k_delta'] for r in results]

    aggregated = {
        'condition': condition.name,
        'mean_final_k': np.mean(final_ks),
        'std_final_k': np.std(final_ks),
        'se_final_k': np.std(final_ks) / np.sqrt(len(final_ks)),
        'max_final_k': np.max(final_ks),
        'min_final_k': np.min(final_ks),
        'mean_k_variance': np.mean(k_variances),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_k_pearson': np.mean(k_pearsons),
        'mean_k_spearman': np.mean(k_spearmans),
        'mean_k_delta': np.mean(k_deltas),
        'all_final_ks': final_ks,
        'all_mean_ks': mean_ks,
        'all_k_variances': k_variances,
        'all_rewards': rewards,
        'all_k_pearsons': k_pearsons,
        'all_k_spearmans': k_spearmans,
        'all_k_deltas': k_deltas,
        'k_histories': [r['k_history'] for r in results],
        'episode_data': results  # Full per-episode data
    }

    # Compute baseline performance ratio if baseline provided
    if baseline_k is not None and baseline_k > 0:
        aggregated['baseline_performance_ratio'] = aggregated['mean_final_k'] / baseline_k

    print(f"\nCondition {condition.name} complete:")
    print(f"  Mean K: {aggregated['mean_final_k']:.4f} ± {aggregated['se_final_k']:.4f} (SE)")
    print(f"  Robust: K_pearson={aggregated['mean_k_pearson']:.4f}, K_spearman={aggregated['mean_k_spearman']:.4f}")
    print(f"  Partial: Mean Δ={aggregated['mean_k_delta']:.4f} (reward independence)")
    print(f"  Mean Reward: {aggregated['mean_reward']:.2f} ± {aggregated['std_reward']:.2f}")
    if 'baseline_performance_ratio' in aggregated:
        print(f"  Baseline Ratio: {aggregated['baseline_performance_ratio']:.4f}")

    return aggregated

def generate_visualizations(all_results: List[Dict], output_dir: str):
    """Generate comprehensive visualizations for Track F."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}\n")

    # Set style
    sns.set_style("whitegrid")

    # Figure 1: Summary metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Track F: Adversarial Robustness Analysis (Corrected FGSM)', fontsize=16, fontweight='bold')

    conditions = [r['condition'] for r in all_results]
    mean_ks = [r['mean_final_k'] for r in all_results]
    std_ks = [r['std_final_k'] for r in all_results]

    # 1a: Mean K-Index by condition
    ax = axes[0, 0]
    bars = ax.bar(range(len(conditions)), mean_ks, yerr=std_ks, capsize=5, alpha=0.7,
                   color=sns.color_palette("viridis", len(conditions)))
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Mean Final K-Index', fontsize=12)
    ax.set_title('K-Index by Adversarial Condition', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    baseline_k = all_results[0]['mean_final_k']
    ax.axhline(y=baseline_k, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()

    # 1b: K-Index variance (stability)
    ax = axes[0, 1]
    k_variances = [r['mean_k_variance'] for r in all_results]
    bars = ax.bar(range(len(conditions)), k_variances, alpha=0.7,
                   color=sns.color_palette("rocket", len(conditions)))
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Mean K-Index Variance', fontsize=12)
    ax.set_title('Coherence Stability (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 1c: Baseline performance ratio
    ax = axes[1, 0]
    ratios = [r.get('baseline_performance_ratio', 1.0) for r in all_results]
    bars = ax.bar(range(len(conditions)), ratios, alpha=0.7,
                   color=sns.color_palette("mako", len(conditions)))
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (100%)')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Performance Ratio (vs Baseline)', fontsize=12)
    ax.set_title('Robustness: K-Index Relative to Clean Baseline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 1d: Episode reward performance
    ax = axes[1, 1]
    rewards = [r['mean_reward'] for r in all_results]
    reward_stds = [r['std_reward'] for r in all_results]
    bars = ax.bar(range(len(conditions)), rewards, yerr=reward_stds, capsize=5, alpha=0.7,
                   color=sns.color_palette("flare", len(conditions)))
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Task Performance Under Adversarial Conditions', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'adversarial_robustness_summary.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: adversarial_robustness_summary.png")
    plt.close()

    print(f"\n✓ All visualizations saved to {output_dir}")

# ✅ PATCH 6: Enhanced main() with CSV/NPZ export for analysis
def main():
    """Main execution function with detailed data export."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Track F: Adversarial Robustness Testing')
    parser.add_argument('--config', type=str,
                       default='fre/configs/track_f_adversarial.yaml',
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("TRACK F: ADVERSARIAL ROBUSTNESS TESTING (CORRECTED FGSM)")
    print("Testing K-Index stability under various perturbation types")
    print("Using proper gradient-based FGSM: x' = x + ε × sign(∇_x L(x,y))")
    print("="*80 + "\n")

    # Load configuration
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded from: {config_path}")
    print(f"Total conditions: {len(config['adversarial_conditions'])}")
    print(f"Episodes per condition: {config['experiment']['n_episodes']}")
    print(f"Total episodes: {len(config['adversarial_conditions']) * config['experiment']['n_episodes']}\n")

    # Parse adversarial conditions
    conditions = []
    for cond_dict in config['adversarial_conditions']:
        condition = AdversarialCondition(
            name=cond_dict['name'],
            perturbation_type=cond_dict['perturbation_type'],
            perturbation_strength=cond_dict['perturbation_strength'],
            perturbation_frequency=cond_dict['perturbation_frequency'],
            target=cond_dict.get('target', 'observations'),
            description=cond_dict['description']
        )
        conditions.append(condition)

    # Run all conditions
    all_results = []
    baseline_k = None

    for i, condition in enumerate(conditions):
        result = run_condition(condition, config, baseline_k)
        all_results.append(result)

        # First condition (baseline) sets the reference K-Index
        if i == 0:
            baseline_k = result['mean_final_k']
            print(f"\n  → Baseline K-Index: {baseline_k:.4f} (reference for robustness ratios)")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output']['save_path'], f'track_f_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Export per-episode CSV for fre/analyze_track_f.py
    episode_records = []
    fgsm_sanity_records = []

    for result in all_results:
        condition_name = result['condition']
        for ep_idx, ep_data in enumerate(result['episode_data']):
            episode_records.append({
                'condition': condition_name,
                'episode': ep_idx,
                'k': ep_data['final_k'],
                'k_pearson': ep_data['k_pearson'],
                'k_pearson_z': ep_data['k_pearson_z'],
                'k_spearman': ep_data['k_spearman'],
                'k_raw': ep_data['k_raw'],
                'k_partial': ep_data['k_partial'],
                'k_delta': ep_data['k_delta'],
                'reward': ep_data['episode_reward'],
                'k_variance': ep_data['k_variance']
            })

            # FGSM sanity checks
            if ep_data['fgsm_total'] > 0:
                for base_loss, adv_loss in zip(ep_data['base_losses'], ep_data['adv_losses']):
                    fgsm_sanity_records.append({
                        'condition': condition_name,
                        'episode': ep_idx,
                        'base_loss': base_loss,
                        'adv_loss': adv_loss,
                        'increased': adv_loss >= base_loss
                    })

    # Save CSV files
    episode_df = pd.DataFrame(episode_records)
    episode_csv_path = os.path.join(output_dir, 'track_f_episode_metrics.csv')
    episode_df.to_csv(episode_csv_path, index=False)
    print(f"\n✓ Episode metrics saved: {episode_csv_path}")

    if fgsm_sanity_records:
        fgsm_df = pd.DataFrame(fgsm_sanity_records)
        fgsm_csv_path = os.path.join(output_dir, 'fgsm_sanity_checks.csv')
        fgsm_df.to_csv(fgsm_csv_path, index=False)
        print(f"✓ FGSM sanity checks saved: {fgsm_csv_path}")

    # Save as NPZ (full data)
    np.savez(
        os.path.join(output_dir, f'track_f_{timestamp}.npz'),
        conditions=[r['condition'] for r in all_results],
        mean_final_ks=[r['mean_final_k'] for r in all_results],
        std_final_ks=[r['std_final_k'] for r in all_results],
        se_final_ks=[r['se_final_k'] for r in all_results],
        mean_k_variances=[r['mean_k_variance'] for r in all_results],
        mean_rewards=[r['mean_reward'] for r in all_results],
        baseline_performance_ratios=[r.get('baseline_performance_ratio', 1.0) for r in all_results],
        all_results=all_results,
        config=config
    )
    print(f"✓ NPZ archive saved: {os.path.join(output_dir, f'track_f_{timestamp}.npz')}")

    # Generate visualizations
    if config['output']['generate_figures']:
        figures_dir = os.path.join(output_dir, 'figures')
        generate_visualizations(all_results, figures_dir)

    # Print summary
    print(f"\n{'='*80}")
    print("TRACK F SUMMARY (CORRECTED FGSM)")
    print(f"{'='*80}\n")
    print(f"Baseline (clean) K-Index: {baseline_k:.4f}")
    print(f"\nRobustness Rankings (by K-Index retention):")

    # Sort by baseline performance ratio
    ranked = sorted(all_results, key=lambda x: x.get('baseline_performance_ratio', 0), reverse=True)
    for i, r in enumerate(ranked, 1):
        ratio = r.get('baseline_performance_ratio', 1.0)
        print(f"  {i}. {r['condition']:25s} - {r['mean_final_k']:.4f} K ({ratio*100:.1f}% of baseline)")

    print(f"\n✓ Track F complete: {len(all_results)} conditions tested")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Ready for analysis: python3 fre/analyze_track_f.py --input {episode_csv_path}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
