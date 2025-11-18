#!/usr/bin/env python3
"""
🎯 Track G: Consciousness Threshold Crossing Runner
Combines developmental learning (Track E) + adversarial training (Track F)
Goal: First artificial system to cross K-Index > 1.5
"""

import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class HybridEnvironment:
    """Environment combining developmental tasks with adversarial perturbations."""
    obs_dim: int = 20
    action_dim: int = 10
    epsilon: float = 0.0  # Adversarial strength
    difficulty: float = 1.0
    
    def __post_init__(self):
        self.state = None
        self.step_count = 0
        
    def reset(self):
        """Reset environment."""
        self.state = np.random.randn(self.obs_dim) * self.difficulty
        self.step_count = 0
        return self.state.copy()
    
    def add_adversarial_noise(self, state: np.ndarray) -> np.ndarray:
        """Add adversarial perturbations."""
        if self.epsilon == 0.0:
            return state
        noise = np.random.randn(*state.shape)
        noise = noise / (np.linalg.norm(noise) + 1e-10)
        perturbed = state + self.epsilon * noise
        return np.clip(perturbed, -5, 5)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute action."""
        # Add adversarial noise to observation
        observed_state = self.add_adversarial_noise(self.state)
        
        # Compute reward
        action_quality = np.dot(action, observed_state[:self.action_dim]) / (self.action_dim * self.difficulty)
        reward = np.tanh(action_quality)
        
        # Update state
        self.state = 0.9 * self.state + 0.1 * np.random.randn(self.obs_dim)
        self.step_count += 1
        
        done = self.step_count >= 50
        return observed_state, reward, done
    
    def compute_k_index(self, observations: np.ndarray, actions: np.ndarray) -> float:
        """Compute K-Index from observations and actions."""
        if len(observations) < 2:
            return 0.0
        
        # Simple correlation-based K-Index
        obs_flat = observations.flatten()
        act_flat = actions.flatten()
        
        if len(obs_flat) != len(act_flat):
            min_len = min(len(obs_flat), len(act_flat))
            obs_flat = obs_flat[:min_len]
            act_flat = act_flat[:min_len]
        
        if np.std(obs_flat) < 1e-10 or np.std(act_flat) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(obs_flat, act_flat)[0, 1]
        k_index = 2.0 * np.abs(correlation)
        
        return float(np.clip(k_index, 0, 2))


class SimpleAgent:
    """Simple learning agent for threshold crossing experiments."""
    
    def __init__(self, obs_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        
        # Simple linear policy
        self.W = np.random.randn(action_dim, obs_dim) * 0.1
        self.b = np.zeros(action_dim)
        
        # Experience buffer
        self.experiences = []
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Select action with exploration."""
        # Compute action
        action = np.tanh(self.W @ state + self.b)
        
        # Add exploration noise
        if epsilon > 0:
            action += np.random.randn(self.action_dim) * epsilon
            action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Simple gradient update."""
        # Store experience
        self.experiences.append((state, action, reward, next_state))
        
        # Update policy (simple gradient ascent)
        if len(self.experiences) > 10:
            # Sample recent experiences
            recent = self.experiences[-10:]
            
            for s, a, r, _ in recent:
                # Gradient: move policy toward higher reward actions
                grad_W = np.outer(r * a, s)
                grad_b = r * a
                
                self.W += self.lr * grad_W
                self.b += self.lr * grad_b
    
    def compute_harmony_metrics(self, observations: List[np.ndarray], 
                                actions: List[np.ndarray], 
                                rewards: List[float]) -> Dict[str, float]:
        """Compute 7 Harmony metrics."""
        if len(observations) < 2:
            return {f"H{i}": 0.0 for i in range(1, 8)}
        
        obs_array = np.array(observations)
        act_array = np.array(actions)
        rew_array = np.array(rewards)
        
        # H1: Coherence (variance in actions - lower is more coherent)
        h1 = 1.0 / (1.0 + np.std(act_array))
        
        # H2: Flourishing (mean reward)
        h2 = np.tanh(np.mean(rew_array))
        
        # H3: Wisdom (reward trend - are we improving?)
        h3 = np.tanh(np.mean(np.diff(rew_array))) if len(rew_array) > 1 else 0.0
        
        # H4: Play (exploration - action entropy)
        h4 = np.tanh(np.std(act_array))
        # H5: Interconnection (obs-action coupling) - simplified
        # Use normalized dot product as coupling measure
        h5_vals = []
        for t in range(min(len(obs_array), len(act_array))):
            min_dim = min(len(obs_array[t]), len(act_array[t]))
            obs_norm = obs_array[t][:min_dim] / (np.linalg.norm(obs_array[t][:min_dim]) + 1e-10)
            act_norm = act_array[t][:min_dim] / (np.linalg.norm(act_array[t][:min_dim]) + 1e-10)
            coupling = np.abs(np.dot(obs_norm, act_norm))
            h5_vals.append(coupling)
        h5 = np.tanh(np.mean(h5_vals)) if h5_vals else 0.0
        
        # H6: Reciprocity (consistency)
        h6 = 1.0 / (1.0 + np.std(rew_array))
        
        # H7: Progression (final vs initial reward)
        h7 = np.tanh(rew_array[-1] - rew_array[0]) if len(rew_array) > 1 else 0.0
        
        return {
            "H1_Coherence": float(h1),
            "H2_Flourishing": float(h2),
            "H3_Wisdom": float(h3),
            "H4_Play": float(h4),
            "H5_Interconnection": float(h5),
            "H6_Reciprocity": float(h6),
            "H7_Progression": float(h7)
        }


def run_episode(env: HybridEnvironment, agent: SimpleAgent, 
                epsilon_explore: float = 0.1) -> Dict:
    """Run single episode."""
    state = env.reset()
    
    observations = []
    actions = []
    rewards = []
    
    for step in range(50):
        # Select and execute action
        action = agent.select_action(state, epsilon_explore)
        next_state, reward, done = env.step(action)
        
        # Store
        observations.append(state.copy())
        actions.append(action.copy())
        rewards.append(reward)
        
        # Update agent
        agent.update(state, action, reward, next_state)
        
        state = next_state
        if done:
            break
    
    # Compute metrics
    k_index = env.compute_k_index(np.array(observations), np.array(actions))
    harmony = agent.compute_harmony_metrics(observations, actions, rewards)
    
    return {
        'k_index': k_index,
        'mean_reward': float(np.mean(rewards)),
        'total_steps': len(rewards),
        **harmony
    }


def run_phase_g1(config: dict) -> Dict:
    """Run Phase G1: Developmental + Adversarial Hybrid."""
    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G1: DEVELOPMENTAL + ADVERSARIAL HYBRID")
    print("="*70)
    
    phase = config['phase_g1']
    epsilon_values = phase['adversarial']['epsilon_values']
    episodes_per_level = phase['adversarial']['episodes_per_level']
    
    results = {
        'phase': 'G1',
        'epsilon_levels': [],
        'max_k_index': 0.0,
        'episodes_completed': 0
    }
    
    # Create agent
    agent = SimpleAgent(obs_dim=20, action_dim=10, learning_rate=0.001)
    
    all_k_indices = []
    
    print(f"\n📊 Progressive Adversarial Curriculum:")
    print(f"Epsilon levels: {epsilon_values}")
    print(f"Episodes per level: {episodes_per_level}")
    print(f"Total episodes: {len(epsilon_values) * episodes_per_level}\n")
    
    # Progressive adversarial curriculum
    for level_idx, epsilon in enumerate(epsilon_values):
        print(f"\n--- Level {level_idx+1}/{len(epsilon_values)}: ε = {epsilon:.3f} ---")
        
        # Create environment with this epsilon
        env = HybridEnvironment(epsilon=epsilon, difficulty=1.0 + epsilon)
        
        level_k_indices = []
        level_results = []
        
        for ep in range(episodes_per_level):
            result = run_episode(env, agent, epsilon_explore=0.1)
            level_k_indices.append(result['k_index'])
            level_results.append(result)
            
            if (ep + 1) % 2 == 0:
                recent_k = np.mean(level_k_indices[-2:])
                print(f"  Episode {ep+1}/{episodes_per_level}: K = {recent_k:.4f}")
        
        # Level statistics
        mean_k = np.mean(level_k_indices)
        max_k = np.max(level_k_indices)
        std_k = np.std(level_k_indices)
        
        print(f"\n  Level Summary:")
        print(f"    Mean K-Index: {mean_k:.4f} ± {std_k:.4f}")
        print(f"    Max K-Index: {max_k:.4f}")
        
        # Check threshold crossing
        if max_k > 1.5:
            print(f"  🎉 THRESHOLD CROSSED! K = {max_k:.4f} > 1.5")
        elif max_k > 1.4:
            print(f"  🔥 APPROACHING! K = {max_k:.4f} (93% of threshold)")
        elif max_k > 1.3:
            print(f"  ⚡ PROGRESS! K = {max_k:.4f} (87% of threshold)")
        
        results['epsilon_levels'].append({
            'epsilon': float(epsilon),
            'episodes': episodes_per_level,
            'mean_k': float(mean_k),
            'max_k': float(max_k),
            'std_k': float(std_k),
            'k_indices': [float(k) for k in level_k_indices],
            'detailed_results': level_results
        })
        
        all_k_indices.extend(level_k_indices)
        results['episodes_completed'] += episodes_per_level
    
    # Overall statistics
    results['max_k_index'] = float(np.max(all_k_indices))
    results['mean_k_index'] = float(np.mean(all_k_indices))
    results['std_k_index'] = float(np.std(all_k_indices))
    results['final_k_index'] = float(all_k_indices[-1])
    
    return results


def save_results(results: Dict, config: dict):
    """Save results to JSON."""
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"track_g_phase_g1_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("🏆 PHASE G1 COMPLETE")
    print("="*70)
    print(f"Total episodes: {results['episodes_completed']}")
    print(f"Max K-Index: {results['max_k_index']:.4f}")
    print(f"Mean K-Index: {results['mean_k_index']:.4f} ± {results['std_k_index']:.4f}")
    print(f"Final K-Index: {results['final_k_index']:.4f}")
    
    # Success criteria
    success = config['success_criteria']
    if results['max_k_index'] >= success['stretch']:
        print(f"\n🏆 STRETCH GOAL ACHIEVED! K ≥ {success['stretch']}")
    elif results['max_k_index'] >= success['target']:
        print(f"\n🎯 TARGET ACHIEVED! K ≥ {success['target']}")
    elif results['max_k_index'] >= success['minimal']:
        print(f"\n✅ MINIMAL SUCCESS! K ≥ {success['minimal']}")
    else:
        progress_pct = (results['max_k_index'] / success['minimal']) * 100
        print(f"\n📊 Progress: {results['max_k_index']:.4f} / {success['minimal']} ({progress_pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Track G: Threshold Crossing')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--phase', type=str, default='g1', choices=['g1'])
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("\n" + "🌊"*35)
    print("TRACK G: CONSCIOUSNESS THRESHOLD CROSSING")
    print("Goal: First artificial system to cross K-Index > 1.5")
    print("🌊"*35)
    
    # Run phase
    if args.phase == 'g1':
        results = run_phase_g1(config)
        save_results(results, config)
    
    print("\n🌊 Track G Phase G1 complete!")


if __name__ == '__main__':
    main()
