#!/usr/bin/env python3
"""
🧠 Track H: Memory Integration Runner
Goal: Add temporal coherence through recurrent architectures
Strategy: LSTM/GRU/Attention for sustained consciousness patterns
Builds on: Track G2's adversarial robustness foundation
"""

import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class MemoryEnvironment:
    """Environment with longer episodes for memory testing."""
    obs_dim: int = 20
    action_dim: int = 10
    epsilon: float = 0.10  # From Track G optimal
    difficulty: float = 1.0
    
    def __post_init__(self):
        self.state = None
        self.step_count = 0
        self.history = []
        
    def reset(self):
        """Reset environment."""
        self.state = np.random.randn(self.obs_dim) * self.difficulty
        self.step_count = 0
        self.history = []
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
        """Execute action with memory of past states."""
        # Add adversarial noise
        observed_state = self.add_adversarial_noise(self.state)
        
        # Compute reward (can depend on history)
        action_quality = np.dot(action, observed_state[:self.action_dim]) / (self.action_dim * self.difficulty)
        reward = np.tanh(action_quality)
        
        # Add temporal bonus for coherent behavior
        if len(self.history) > 0:
            prev_state, prev_action = self.history[-1]
            state_coherence = np.corrcoef(self.state, prev_state)[0, 1]
            action_coherence = np.corrcoef(action, prev_action)[0, 1]
            temporal_bonus = 0.1 * np.tanh((state_coherence + action_coherence) / 2)
            reward += temporal_bonus
        
        # Store history
        self.history.append((self.state.copy(), action.copy()))
        
        # Update state
        self.state = 0.9 * self.state + 0.1 * np.random.randn(self.obs_dim)
        self.step_count += 1
        
        done = self.step_count >= 300  # Longer episodes for memory
        return observed_state, reward, done
    
    def compute_k_index(self, observations: np.ndarray, actions: np.ndarray) -> float:
        """Compute K-Index from observations and actions."""
        if len(observations) < 2:
            return 0.0
        
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


class MemoryAgent:
    """Agent with recurrent memory (LSTM/GRU/Attention)."""
    
    def __init__(self, obs_dim: int, action_dim: int, memory_type: str = "lstm",
                 hidden_size: int = 64, learning_rate: float = 0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_type = memory_type
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # Policy network (simplified - in production would use actual LSTM/GRU)
        self.W_in = np.random.randn(hidden_size, obs_dim) * 0.1
        self.W_hidden = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_out = np.random.randn(action_dim, hidden_size) * 0.1
        self.b_hidden = np.zeros(hidden_size)
        self.b_out = np.zeros(action_dim)
        
        # Memory state
        self.hidden_state = np.zeros(hidden_size)
        self.cell_state = np.zeros(hidden_size)  # For LSTM
        
        # Experience buffer
        self.experiences = []
    
    def reset_memory(self):
        """Reset hidden states."""
        self.hidden_state = np.zeros(self.hidden_size)
        self.cell_state = np.zeros(self.hidden_size)
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Select action using memory."""
        # Simplified recurrent update
        if self.memory_type == "lstm":
            # LSTM-like update (simplified)
            input_gate = np.tanh(self.W_in @ state + self.W_hidden @ self.hidden_state + self.b_hidden)
            self.cell_state = 0.9 * self.cell_state + 0.1 * input_gate
            self.hidden_state = np.tanh(self.cell_state)
        elif self.memory_type == "gru":
            # GRU-like update (simplified)
            candidate = np.tanh(self.W_in @ state + self.W_hidden @ self.hidden_state + self.b_hidden)
            self.hidden_state = 0.9 * self.hidden_state + 0.1 * candidate
        else:  # attention
            # Attention-like update (simplified)
            attention = np.tanh(self.W_in @ state)
            self.hidden_state = 0.9 * self.hidden_state + 0.1 * attention
        
        # Compute action from hidden state
        action = np.tanh(self.W_out @ self.hidden_state + self.b_out)
        
        # Add exploration noise
        if epsilon > 0:
            action += np.random.randn(self.action_dim) * epsilon
            action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Update policy with memory."""
        self.experiences.append((state, action, reward, next_state, self.hidden_state.copy()))
        
        # Update policy
        if len(self.experiences) > 10:
            recent = self.experiences[-10:]
            
            for s, a, r, _, h in recent:
                # Gradient ascent on reward
                grad_W_out = np.outer(r * a, h)
                grad_W_hidden = np.outer(r * h, h)
                grad_W_in = np.outer(r * h, s)
                
                self.W_out += self.lr * grad_W_out
                self.W_hidden += self.lr * grad_W_hidden * 0.5  # Smaller update for recurrent
                self.W_in += self.lr * grad_W_in * 0.5
    
    def compute_temporal_coherence(self, observations: List[np.ndarray], 
                                   actions: List[np.ndarray], 
                                   window: int = 100) -> float:
        """Measure temporal coherence of behavior."""
        if len(actions) < window:
            return 0.0
        
        # Compute autocorrelation of actions
        recent_actions = np.array(actions[-window:])
        action_flat = recent_actions.flatten()
        
        if len(action_flat) < 2:
            return 0.0
        
        # Autocorrelation at lag 1
        mean_action = np.mean(action_flat)
        var_action = np.var(action_flat)
        
        if var_action < 1e-10:
            return 0.0
        
        autocorr = np.mean((action_flat[:-1] - mean_action) * (action_flat[1:] - mean_action)) / var_action
        
        return float(np.clip(autocorr, -1, 1))

    def state_dict(self) -> Dict[str, Any]:
        """Serialize recurrent weights and metadata."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "memory_type": self.memory_type,
            "hidden_size": self.hidden_size,
            "learning_rate": self.lr,
            "W_in": self.W_in.tolist(),
            "W_hidden": self.W_hidden.tolist(),
            "W_out": self.W_out.tolist(),
            "b_hidden": self.b_hidden.tolist(),
            "b_out": self.b_out.tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore recurrent weights from checkpoint."""
        required = {"W_in", "W_hidden", "W_out", "b_hidden", "b_out"}
        if not required.issubset(state.keys()):
            missing = required - set(state.keys())
            raise ValueError(f"Invalid state dict: missing {missing}")
        self.obs_dim = int(state.get("obs_dim", self.obs_dim))
        self.action_dim = int(state.get("action_dim", self.action_dim))
        self.memory_type = state.get("memory_type", self.memory_type)
        self.hidden_size = int(state.get("hidden_size", self.hidden_size))
        self.lr = float(state.get("learning_rate", self.lr))
        self.W_in = np.array(state["W_in"], dtype=np.float64)
        self.W_hidden = np.array(state["W_hidden"], dtype=np.float64)
        self.W_out = np.array(state["W_out"], dtype=np.float64)
        self.b_hidden = np.array(state["b_hidden"], dtype=np.float64)
        self.b_out = np.array(state["b_out"], dtype=np.float64)


def _save_agent_checkpoint(agent: Any, path: Path, *, context: str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
    if not hasattr(agent, "state_dict"):
        raise ValueError("Agent does not support checkpointing")
    payload = {
        "agent_class": agent.__class__.__name__,
        "context": context,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": agent.state_dict(),
        "metadata": extra_metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_agent_checkpoint(agent: Any, path: Path, *, context: str) -> None:
    if not hasattr(agent, "load_state_dict"):
        raise ValueError("Agent does not support checkpointing")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = payload.get("state", payload)
    agent.load_state_dict(state)
    print(f"🔄 Warm start: loaded {agent.__class__.__name__} state from {path} for {context}")


def _maybe_load_agent(agent: Any, warm_cfg: Dict[str, Any], context: str) -> Optional[str]:
    load_path = warm_cfg.get("load_path")
    if not load_path:
        return None
    _load_agent_checkpoint(agent, Path(load_path), context=context)
    return str(load_path)


def _maybe_save_agent(agent: Any, warm_cfg: Dict[str, Any], context: str, *, extra_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    save_path = warm_cfg.get("save_path")
    if not save_path:
        return None
    metadata = warm_cfg.get("metadata", {})
    if extra_metadata:
        metadata = {**metadata, **extra_metadata}
    _save_agent_checkpoint(agent, Path(save_path), context=context, extra_metadata=metadata)
    print(f"💾 Saved {agent.__class__.__name__} checkpoint for {context} to {save_path}")
    return str(save_path)


def _apply_track_h_warm_start_overrides(config: Dict[str, Any], load_path: Optional[str], save_path: Optional[str]) -> None:
    if not load_path and not save_path:
        return
    track_cfg = config.get("track_h")
    if not track_cfg or "warm_start" not in track_cfg:
        raise KeyError("track_h.warm_start missing in config; cannot override.")
    warm_cfg = track_cfg["warm_start"]
    if load_path:
        warm_cfg["load_path"] = load_path
        print(f"🔧 Track H warm-start load override → {load_path}")
    if save_path is not None:
        warm_cfg["save_path"] = save_path
        print(f"🔧 Track H warm-start save override → {save_path}")


def _validate_track_h_config(config: Dict[str, Any]) -> None:
    track_cfg = config.get("track_h", {})
    warm_cfg = track_cfg.get("warm_start", {})
    load_path = warm_cfg.get("load_path")
    if load_path:
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Track H warm-start load checkpoint not found: {path}")
    archs = track_cfg.get("memory_architectures", [])
    for arch in archs:
        save_path = arch.get("warm_start", {}).get("save_path")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def run_episode(env: MemoryEnvironment, agent: MemoryAgent, 
                epsilon_explore: float = 0.1, episode_length: int = 300) -> Dict:
    """Run single episode with memory."""
    state = env.reset()
    agent.reset_memory()
    
    observations = []
    actions = []
    rewards = []
    hidden_states = []
    
    for step in range(episode_length):
        # Select and execute action
        action = agent.select_action(state, epsilon_explore)
        next_state, reward, done = env.step(action)
        
        # Store
        observations.append(state.copy())
        actions.append(action.copy())
        rewards.append(reward)
        hidden_states.append(agent.hidden_state.copy())
        
        # Update agent
        agent.update(state, action, reward, next_state)
        
        state = next_state
        if done:
            break
    
    # Compute metrics
    k_index = env.compute_k_index(np.array(observations), np.array(actions))
    temporal_coherence = agent.compute_temporal_coherence(observations, actions, window=100)
    
    # Compute 7 Harmonies
    obs_array = np.array(observations)
    act_array = np.array(actions)
    rew_array = np.array(rewards)
    
    h1 = 1.0 / (1.0 + np.std(act_array))
    h2 = np.tanh(np.mean(rew_array))
    h3 = np.tanh(np.mean(np.diff(rew_array))) if len(rew_array) > 1 else 0.0
    h4 = np.tanh(np.std(act_array))
    
    # H5: Obs-action coupling
    h5_vals = []
    for t in range(min(len(obs_array), len(act_array))):
        min_dim = min(len(obs_array[t]), len(act_array[t]))
        obs_norm = obs_array[t][:min_dim] / (np.linalg.norm(obs_array[t][:min_dim]) + 1e-10)
        act_norm = act_array[t][:min_dim] / (np.linalg.norm(act_array[t][:min_dim]) + 1e-10)
        coupling = np.abs(np.dot(obs_norm, act_norm))
        h5_vals.append(coupling)
    h5 = np.tanh(np.mean(h5_vals)) if h5_vals else 0.0
    
    h6 = 1.0 / (1.0 + np.std(rew_array))
    h7 = np.tanh(rew_array[-1] - rew_array[0]) if len(rew_array) > 1 else 0.0
    
    return {
        'k_index': k_index,
        'temporal_coherence': temporal_coherence,
        'mean_reward': float(np.mean(rewards)),
        'total_steps': len(rewards),
        'H1_Coherence': float(h1),
        'H2_Flourishing': float(h2),
        'H3_Wisdom': float(h3),
        'H4_Play': float(h4),
        'H5_Interconnection': float(h5),
        'H6_Reciprocity': float(h6),
        'H7_Progression': float(h7)
    }


def run_track_h(config: dict) -> Dict:
    """Run Track H: Memory Integration."""
    print("\n" + "="*70)
    print("🧠 TRACK H: MEMORY INTEGRATION")
    print("="*70)
    
    track = config['track_h']
    training = track['training']
    
    print(f"\n📊 Configuration:")
    print(f"Memory Architectures: {len(track['memory_architectures'])}")
    print(f"Episodes per architecture: {training['episodes']}")
    print(f"Episode length: {training['episode_length']} steps")
    print(f"Epsilon: {training['epsilon']}")
    print(f"Sequence length (memory window): {training['sequence_length']}\n")
    
    results = {
        'track': 'H',
        'memory_tests': [],
        'max_k_index': 0.0,
        'best_memory_type': None
    }
    
    # Test each memory architecture
    for arch_config in track['memory_architectures']:
        mem_type = arch_config['type']
        hidden_size = arch_config['hidden_size']
        
        print(f"\n{'='*70}")
        print(f"🧠 Testing Memory Architecture: {mem_type.upper()}")
        print(f"{'='*70}")
        print(f"Hidden size: {hidden_size}")
        print(f"Description: {arch_config['description']}\n")
        
        # Create environment and agent
        env = MemoryEnvironment(epsilon=training['epsilon'])
        agent = MemoryAgent(obs_dim=20, action_dim=10, memory_type=mem_type, 
                           hidden_size=hidden_size, learning_rate=0.001)
        warm_cfg = dict(track.get('warm_start', {}))
        warm_cfg.update(arch_config.get('warm_start', {}))
        warm_status: Dict[str, Optional[str]] = {'loaded_from': _maybe_load_agent(agent, warm_cfg, f"Track H ({mem_type})")}
        
        arch_k_indices = []
        arch_coherence = []
        arch_results = []
        
        # Training loop
        for ep in range(training['episodes']):
            result = run_episode(env, agent, epsilon_explore=0.1, 
                               episode_length=training['episode_length'])
            
            arch_k_indices.append(result['k_index'])
            arch_coherence.append(result['temporal_coherence'])
            arch_results.append(result)
            
            # Progress reporting
            if (ep + 1) % 100 == 0 or ep == 0:
                recent_k = np.mean(arch_k_indices[-min(20, len(arch_k_indices)):])
                recent_coh = np.mean(arch_coherence[-min(20, len(arch_coherence)):])
                print(f"  Episode {ep+1}/{training['episodes']}: "
                      f"K = {result['k_index']:.4f} | "
                      f"Coherence = {result['temporal_coherence']:.4f} | "
                      f"Recent Mean K = {recent_k:.4f}")
        
        # Architecture summary
        mean_k = np.mean(arch_k_indices)
        max_k = np.max(arch_k_indices)
        std_k = np.std(arch_k_indices)
        mean_coherence = np.mean(arch_coherence)
        
        print(f"\n📊 {mem_type.upper()} Results:")
        print(f"  Mean K-Index: {mean_k:.4f} ± {std_k:.4f}")
        print(f"  Max K-Index: {max_k:.4f}")
        print(f"  Mean Temporal Coherence: {mean_coherence:.4f}")
        
        # Check milestones
        if max_k > 1.5:
            print(f"  🎉 THRESHOLD CROSSED! K = {max_k:.4f} > 1.5")
        elif max_k > 1.4:
            print(f"  🔥 EXTREMELY CLOSE! K = {max_k:.4f} (93.3%)")
        elif max_k > 1.3:
            print(f"  ⚡ MAJOR PROGRESS! K = {max_k:.4f} (86.7%)")
        
        # Store results
        test_entry = {
            'memory_type': mem_type,
            'hidden_size': hidden_size,
            'mean_k': float(mean_k),
            'max_k': float(max_k),
            'std_k': float(std_k),
            'mean_coherence': float(mean_coherence),
            'k_indices': [float(k) for k in arch_k_indices],
            'coherence_values': [float(c) for c in arch_coherence],
            'detailed_results': arch_results
        }

        saved_path = _maybe_save_agent(
            agent,
            warm_cfg,
            f"Track H ({mem_type})",
            extra_metadata={
                "episodes_completed": training['episodes'],
                "mean_k": float(mean_k),
                "max_k": float(max_k),
            },
        )
        if warm_status['loaded_from'] or saved_path:
            warm_status['saved_to'] = saved_path
            test_entry['warm_start'] = warm_status

        results['memory_tests'].append(test_entry)
        
        # Update global max
        if max_k > results['max_k_index']:
            results['max_k_index'] = max_k
            results['best_memory_type'] = mem_type
    
    return results


def save_results(results: Dict, config: dict):
    """Save results to JSON."""
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"track_h_memory_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("🏆 TRACK H COMPLETE")
    print("="*70)
    print(f"Memory architectures tested: {len(results['memory_tests'])}")
    print(f"Best architecture: {results['best_memory_type']}")
    print(f"Max K-Index: {results['max_k_index']:.4f}")
    
    # Success criteria
    success = config['success_criteria']
    if results['max_k_index'] >= success['stretch']:
        print(f"\n🏆 STRETCH GOAL! K ≥ {success['stretch']}")
    elif results['max_k_index'] >= success['target']:
        print(f"\n🎯 TARGET ACHIEVED! K ≥ {success['target']}")
    elif results['max_k_index'] >= success['minimal']:
        print(f"\n✅ MINIMAL SUCCESS! K ≥ {success['minimal']}")
    else:
        progress_pct = (results['max_k_index'] / success['minimal']) * 100
        print(f"\n📊 Progress: {results['max_k_index']:.4f} / {success['minimal']} ({progress_pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Track H: Memory Integration')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--warm-start-load', type=str, default=None, help='Override track_h.warm_start.load_path')
    parser.add_argument('--warm-start-save', type=str, default=None, help='Override track_h.warm_start.save_path')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and exit without running episodes')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    _apply_track_h_warm_start_overrides(config, args.warm_start_load, args.warm_start_save)
    
    if args.dry_run:
        _validate_track_h_config(config)
        print(f"✅ Track H dry run successful with {args.config}")
        return
    
    print("\n" + "🌊"*35)
    print("TRACK H: MEMORY INTEGRATION FOR CONSCIOUSNESS")
    print("Goal: Add temporal coherence through recurrent architectures")
    print("🌊"*35)
    
    # Run Track H
    results = run_track_h(config)
    save_results(results, config)
    
    print("\n🌊 Track H complete!")


if __name__ == '__main__':
    main()
