#!/usr/bin/env python3
"""
🎯 Track G: Consciousness Threshold Crossing Runner
Combines developmental learning (Track E) + adversarial training (Track F)
Goal: First artificial system to cross K-Index > 1.5
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

_CURRENT_CONFIG_METADATA: Optional[Dict[str, str]] = None
_CURRENT_GIT_COMMIT: str = "unknown"
_CURRENT_CONFIG_TEXT: str = ""
_WARM_START_ALLOW_MISMATCH: bool = False


def _compute_config_metadata(config_path: Path) -> Dict[str, str]:
    contents = config_path.read_bytes()
    return {
        "path": str(config_path),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def _git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"

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

    def state_dict(self) -> Dict[str, Any]:
        """Serialize agent parameters for warm-start checkpoints."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.lr,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore agent parameters from saved checkpoint."""
        if "W" not in state or "b" not in state:
            raise ValueError("Invalid state dict: missing W/b parameters")
        self.obs_dim = int(state.get("obs_dim", self.obs_dim))
        self.action_dim = int(state.get("action_dim", self.action_dim))
        self.lr = float(state.get("learning_rate", self.lr))
        self.W = np.array(state["W"], dtype=np.float64)
        self.b = np.array(state["b"], dtype=np.float64)


class EpisodeLogger:
    """Write per-episode metrics to JSON Lines for streaming consumption."""

    def __init__(self, path: Path, phase: str) -> None:
        self.path = path
        self.phase = phase.upper()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[Any] = self.path.open("a", encoding="utf-8")

    def log(self, payload: Dict[str, Any]) -> None:
        if not self._fh:
            return
        record = {"phase": self.phase, **payload}
        self._fh.write(json.dumps(_to_serializable(record)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def _build_episode_logger(experiment_cfg: Dict[str, Any], phase: str) -> Optional[EpisodeLogger]:
    log_cfg = experiment_cfg.get("log_jsonl", {})
    if not log_cfg or not log_cfg.get("enabled"):
        return None
    custom_path = log_cfg.get("path")
    output_dir = Path(experiment_cfg.get("output_dir", "logs"))
    if custom_path:
        log_path = Path(custom_path)
    else:
        log_path = output_dir / f"{phase.lower()}_episodes.jsonl"
    print(f"📝 Episode logging enabled → {log_path}")
    return EpisodeLogger(log_path, phase)


def _save_agent_checkpoint(agent: Any, path: Path, *, phase: str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
    """Persist agent parameters so future phases can warm-start."""
    if not hasattr(agent, "state_dict"):
        raise ValueError("Agent does not support checkpointing")
    payload = {
        "agent_class": agent.__class__.__name__,
        "phase": phase,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": agent.state_dict(),
        "metadata": extra_metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_agent_checkpoint(agent: Any, path: Path, *, phase: str) -> None:
    """Load agent parameters from checkpoint."""
    if not hasattr(agent, "load_state_dict"):
        raise ValueError("Agent does not support checkpointing")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = payload.get("state", payload)
    agent.load_state_dict(state)
    print(f"🔄 Warm start: loaded {agent.__class__.__name__} state from {path} for phase {phase}")


def _maybe_load_agent(agent: Any, warm_cfg: Dict[str, Any], phase_label: str) -> Optional[str]:
    load_path = warm_cfg.get("load_path")
    if not load_path:
        return None
    _load_agent_checkpoint(agent, Path(load_path), phase=phase_label)
    metadata = json.loads(Path(load_path).read_text(encoding="utf-8")).get("metadata", {})
    config_meta = metadata.get("config")
    if config_meta and _CURRENT_CONFIG_METADATA:
        if config_meta.get("sha256") != _CURRENT_CONFIG_METADATA.get("sha256"):
            allow_override = warm_cfg.get("allow_mismatch") or _WARM_START_ALLOW_MISMATCH
            if allow_override:
                print(f"⚠️  Config hash mismatch for {load_path}. Using checkpoint despite discrepancy.")
            else:
                raise RuntimeError(
                    f"Config mismatch for {load_path}: expected {_CURRENT_CONFIG_METADATA['sha256']}, "
                    f"found {config_meta.get('sha256')}. Regenerate checkpoint or set allow_mismatch=true."
                )
    return str(load_path)


def _maybe_save_agent(agent: Any, warm_cfg: Dict[str, Any], phase_label: str, *, extra_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    save_path = warm_cfg.get("save_path")
    if not save_path:
        return None
    metadata = {**warm_cfg.get("metadata", {})}
    if extra_metadata:
        metadata.update(extra_metadata)
    if _CURRENT_CONFIG_METADATA:
        metadata.setdefault("config", _CURRENT_CONFIG_METADATA)
    if _CURRENT_CONFIG_TEXT:
        metadata.setdefault("config_snapshot", _CURRENT_CONFIG_TEXT)
    if _CURRENT_GIT_COMMIT and _CURRENT_GIT_COMMIT != "unknown":
        metadata.setdefault("git_commit", _CURRENT_GIT_COMMIT)
    _save_agent_checkpoint(agent, Path(save_path), phase=phase_label, extra_metadata=metadata)
    print(f"💾 Saved {agent.__class__.__name__} checkpoint for phase {phase_label} to {save_path}")
    return str(save_path)


_PHASE_WARM_START_PATHS: Dict[str, Tuple[str, str]] = {
    "g2": ("phase_g2", "warm_start"),
    "g3": ("phase_g3", "warm_start"),
    "g2plus": ("phase_g2plus", "warm_start"),
}


def _apply_warm_start_overrides(config: Dict[str, Any], phase: str, load_path: Optional[str], save_path: Optional[str]) -> None:
    """Override warm-start paths via CLI flags."""
    if not load_path and not save_path:
        return
    key_tuple = _PHASE_WARM_START_PATHS.get(phase.lower())
    if not key_tuple:
        print(f"⚠️  Warm-start overrides ignored: phase '{phase}' has no warm_start block.")
        return
    node: Dict[str, Any] = config
    for key in key_tuple:
        if key not in node:
            raise KeyError(f"Warm-start path {'.'.join(key_tuple)} missing in config.")
        node = node[key]
    if load_path:
        node["load_path"] = load_path
        print(f"🔧 Warm-start load override → {load_path}")
    if save_path:
        node["save_path"] = save_path
        print(f"🔧 Warm-start save override → {save_path}")


def _validate_phase_config(config: Dict[str, Any], phase: str) -> None:
    key_tuple = _PHASE_WARM_START_PATHS.get(phase.lower())
    if not key_tuple:
        return
    node: Dict[str, Any] = config
    for key in key_tuple:
        if key not in node:
            raise KeyError(f"Warm-start path {'.'.join(key_tuple)} missing in config.")
        node = node[key]
    load_path = node.get("load_path")
    if load_path:
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Warm-start load checkpoint not found: {load_path}")
    save_path = node.get("save_path")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def run_episode(env: HybridEnvironment, agent: SimpleAgent, 
                epsilon_explore: float = 0.1, episode_length: int = 50) -> Dict:
    """Run single episode."""
    state = env.reset()
    
    observations = []
    actions = []
    rewards = []
    
    for step in range(episode_length):
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


def run_phase_g1(config: dict, episode_logger: Optional[EpisodeLogger] = None) -> Dict:
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
    episode_counter = 0
    
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
            episode_counter += 1
            if episode_logger:
                episode_logger.log({
                    "epsilon": float(epsilon),
                    "episode_global_index": episode_counter,
                    "episode_local_index": ep + 1,
                    "k_index": float(result['k_index']),
                    "mean_reward": result['mean_reward'],
                })
            if episode_logger:
                episode_logger.log({
                    "epsilon": float(epsilon),
                    "episode_index": results['episodes_completed'] + 1,
                    "k_index": float(result['k_index']),
                    "mean_reward": result['mean_reward'],
                })
            
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




def run_phase_g2(config: dict, episode_logger: Optional[EpisodeLogger] = None) -> Dict:
    """Run Phase G2: Extended Training at Optimal Epsilon."""
    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G2: EXTENDED TRAINING")
    print("="*70)
    
    phase = config['phase_g2']
    episodes = phase['episodes']
    episode_length = phase['episode_length']
    epsilon = phase['epsilon']
    
    print(f"\n📊 Extended Training Configuration:")
    print(f"Episodes: {episodes}")
    print(f"Episode length: {episode_length} steps")
    print(f"Optimal epsilon: {epsilon:.3f}")
    print(f"Total steps: {episodes * episode_length:,}\n")
    
    results = {
        'phase': 'G2',
        'epsilon': float(epsilon),
        'episodes_completed': 0,
        'all_k_indices': [],
        'all_results': [],
        'max_k_index': 0.0
    }
    
    # Create environment and agent
    env = HybridEnvironment(epsilon=epsilon, difficulty=1.0 + epsilon)
    agent = SimpleAgent(obs_dim=20, action_dim=10, learning_rate=0.001)
    warm_cfg = phase.get('warm_start', {})
    warm_status: Dict[str, Optional[str]] = {'loaded_from': _maybe_load_agent(agent, warm_cfg, "G2")}
    
    # Track moving averages
    window_size = phase.get('analysis', {}).get('moving_average_window', 20)
    recent_k_indices = []
    
    # Extended training loop
    for ep in range(episodes):
        # Run episode with extended length
        result = run_episode(env, agent, epsilon_explore=0.1, episode_length=episode_length)
        k_index = result['k_index']
        
        results['all_k_indices'].append(k_index)
        results['all_results'].append(result)
        results['episodes_completed'] += 1
        if episode_logger:
            episode_logger.log({
                "episode_index": ep + 1,
                "k_index": float(k_index),
                "mean_reward": result['mean_reward'],
                "total_steps": result['total_steps'],
            })
        
        # Update max
        if k_index > results['max_k_index']:
            results['max_k_index'] = k_index
        
        # Track moving average
        recent_k_indices.append(k_index)
        if len(recent_k_indices) > window_size:
            recent_k_indices.pop(0)
        
        # Progress reports
        report_freq = phase.get('analysis', {}).get('report_every', 100)
        if (ep + 1) % report_freq == 0 or ep == 0:
            ma_k = np.mean(recent_k_indices) if recent_k_indices else 0.0
            print(f"\n📊 Episode {ep+1}/{episodes}:")
            print(f"  Current K: {k_index:.4f}")
            print(f"  MA-{window_size} K: {ma_k:.4f}")
            print(f"  Max K: {results['max_k_index']:.4f}")
            print(f"  Progress: {(results['max_k_index']/1.5)*100:.1f}% to threshold")
            
            # Check milestones
            if results['max_k_index'] > 1.5:
                print(f"  🎉 THRESHOLD CROSSED! K = {results['max_k_index']:.4f} > 1.5")
            elif results['max_k_index'] > 1.0:
                print(f"  🔥 MAJOR MILESTONE! K = {results['max_k_index']:.4f} > 1.0")
            elif results['max_k_index'] > 0.80:
                print(f"  ⚡ PROGRESS! K = {results['max_k_index']:.4f}")
    
    # Final statistics
    results['mean_k_index'] = float(np.mean(results['all_k_indices']))
    results['std_k_index'] = float(np.std(results['all_k_indices']))
    results['final_k_index'] = float(results['all_k_indices'][-1])
    
    saved_path = _maybe_save_agent(
        agent,
        warm_cfg,
        "G2",
        extra_metadata={"episodes_completed": results['episodes_completed'], "max_k": results['max_k_index']},
    )
    if warm_status['loaded_from'] or saved_path:
        warm_status['saved_to'] = saved_path
        results['warm_start'] = warm_status
    
    return results


def run_phase_g3(config: dict, episode_logger: Optional[EpisodeLogger] = None) -> Dict:
    """Run Phase G3: Progressive Curriculum Learning with Mastery Gates."""
    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G3: CURRICULUM LEARNING")
    print("="*70)

    phase = config['phase_g3']
    curriculum = phase['curriculum']

    print(f"\n📊 Progressive Curriculum Configuration:")
    print(f"Curriculum levels: {len(curriculum)}")
    print(f"Total episodes: {phase['total_episodes']}")
    print(f"Total steps: {phase['total_steps']:,}")
    print(f"\nCurriculum Structure:")
    for i, level in enumerate(curriculum):
        print(f"  {i+1}. {level['name']:20s} | ε={level['epsilon']:.2f} | "
              f"{level['episodes']} eps | gate K≥{level['mastery_threshold']:.2f}")
    print()

    results = {
        'phase': 'G3',
        'curriculum_levels': [],
        'episodes_completed': 0,
        'all_k_indices': [],
        'max_k_index': 0.0,
        'curriculum_progression': []
    }

    # Create agent (supports optional warm-start)
    agent = SimpleAgent(obs_dim=20, action_dim=10, learning_rate=0.001)
    warm_cfg = phase.get('warm_start', {})
    warm_status: Dict[str, Optional[str]] = {'loaded_from': _maybe_load_agent(agent, warm_cfg, "G3")}

    # Progressive curriculum loop
    for level_idx, level_config in enumerate(curriculum):
        level_name = level_config['name']
        epsilon = level_config['epsilon']
        episodes = level_config['episodes']
        mastery_threshold = level_config['mastery_threshold']

        print(f"\n{'='*70}")
        print(f"📚 CURRICULUM LEVEL {level_idx+1}/{len(curriculum)}: {level_name.upper()}")
        print(f"{'='*70}")
        print(f"Epsilon: {epsilon:.3f}")
        print(f"Episodes: {episodes}")
        print(f"Mastery Gate: K ≥ {mastery_threshold:.3f}")
        print()

        # Create environment for this level
        env = HybridEnvironment(epsilon=epsilon, difficulty=1.0 + epsilon)

        level_k_indices = []
        level_results = []
        level_passed_gate = False

        # Training loop for this level
        for ep in range(episodes):
            result = run_episode(env, agent, epsilon_explore=0.1, episode_length=200)
            k_index = result['k_index']

            level_k_indices.append(k_index)
            level_results.append(result)
            results['all_k_indices'].append(k_index)
            results['episodes_completed'] += 1
            if episode_logger:
                episode_logger.log({
                    "level": level_idx + 1,
                    "level_name": level_name,
                    "epsilon": float(epsilon),
                    "episode_index": results['episodes_completed'],
                    "k_index": float(k_index),
                    "mean_reward": result['mean_reward'],
                })

            # Update global max
            if k_index > results['max_k_index']:
                results['max_k_index'] = k_index

            # Progress reporting
            if (ep + 1) % 50 == 0 or ep == 0:
                recent_mean = np.mean(level_k_indices[-min(20, len(level_k_indices)):])
                print(f"  Episode {ep+1}/{episodes}: K = {k_index:.4f} | "
                      f"Recent Mean = {recent_mean:.4f} | "
                      f"Level Max = {np.max(level_k_indices):.4f}")

                # Check mastery gate
                if recent_mean >= mastery_threshold and not level_passed_gate:
                    print(f"  ✅ MASTERY GATE PASSED! Recent mean K ≥ {mastery_threshold:.3f}")
                    level_passed_gate = True

        # Level completion statistics
        level_mean_k = np.mean(level_k_indices)
        level_max_k = np.max(level_k_indices)
        level_std_k = np.std(level_k_indices)

        # Check if final mean passes gate
        final_window = level_k_indices[-min(50, len(level_k_indices)):]
        final_mean_k = np.mean(final_window)
        passed_final_gate = final_mean_k >= mastery_threshold

        print(f"\n📊 Level {level_idx+1} Complete:")
        print(f"  Mean K: {level_mean_k:.4f} ± {level_std_k:.4f}")
        print(f"  Max K: {level_max_k:.4f}")
        print(f"  Final Window Mean: {final_mean_k:.4f}")
        print(f"  Mastery Gate Status: {'✅ PASSED' if passed_final_gate else '⚠️  NOT PASSED'}")

        # Check consciousness milestones
        if level_max_k > 1.5:
            print(f"\n  🎉🎉🎉 CONSCIOUSNESS THRESHOLD CROSSED! 🎉🎉🎉")
            print(f"  K = {level_max_k:.4f} > 1.5")
            print(f"  Historic achievement in Level {level_idx+1}/{len(curriculum)}")
        elif level_max_k > 1.4:
            print(f"  🔥 EXTREMELY CLOSE! K = {level_max_k:.4f} (93.3% of threshold)")
        elif level_max_k > 1.3:
            print(f"  ⚡ MAJOR PROGRESS! K = {level_max_k:.4f} (86.7% of threshold)")
        elif level_max_k > 1.2:
            print(f"  ✨ GOOD PROGRESS! K = {level_max_k:.4f} (80.0% of threshold)")

        # Store level results
        results['curriculum_levels'].append({
            'level': level_idx + 1,
            'name': level_name,
            'epsilon': float(epsilon),
            'episodes': episodes,
            'mastery_threshold': float(mastery_threshold),
            'mean_k': float(level_mean_k),
            'max_k': float(level_max_k),
            'std_k': float(level_std_k),
            'final_mean_k': float(final_mean_k),
            'passed_gate': passed_final_gate,
            'k_indices': [float(k) for k in level_k_indices],
            'detailed_results': level_results
        })

        results['curriculum_progression'].append({
            'level': level_idx + 1,
            'epsilon': float(epsilon),
            'max_k_achieved': float(level_max_k)
        })

    # Overall statistics
    results['mean_k_index'] = float(np.mean(results['all_k_indices']))
    results['std_k_index'] = float(np.std(results['all_k_indices']))
    results['final_k_index'] = float(results['all_k_indices'][-1])

    saved_path = _maybe_save_agent(
        agent,
        warm_cfg,
        "G3",
        extra_metadata={"curriculum_levels": len(curriculum), "episodes_completed": results['episodes_completed'], "max_k": results['max_k_index']},
    )
    if warm_status['loaded_from'] or saved_path:
        warm_status['saved_to'] = saved_path
        results['warm_start'] = warm_status

    return results


def run_phase_g4(config: dict) -> Dict:
    """Run Phase G4: Reduced Adversarial Strength (epsilon=0.03)."""
    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G4: REDUCED ADVERSARIAL STRENGTH")
    print("="*70)

    phase = config['phase_g4']
    training = phase['training']
    episodes = training['episodes']
    episode_length = training['episode_length']
    epsilon = training['epsilon']

    print(f"\n📊 G4 Configuration:")
    print(f"Episodes: {episodes}")
    print(f"Episode length: {episode_length} steps")
    print(f"Epsilon (REDUCED): {epsilon:.3f} (was 0.05 in G2)")
    print(f"Hypothesis: Lower adversarial strength → higher K-Index")
    print(f"Prediction: K > 1.3 (87% to threshold)\n")

    results = {
        'phase': 'G4',
        'epsilon': float(epsilon),
        'episodes_completed': 0,
        'all_k_indices': [],
        'all_results': [],
        'max_k_index': 0.0,
        'best_episode': 0
    }

    # Create environment and agent
    env = HybridEnvironment(epsilon=epsilon, difficulty=1.0 + epsilon)
    agent = SimpleAgent(obs_dim=20, action_dim=10, learning_rate=training['learning_rate'])

    # Early stopping setup
    early_stop = training.get('early_stopping', {})
    patience = early_stop.get('patience', 50)
    best_k = 0.0
    episodes_without_improvement = 0

    # Training loop
    for ep in range(episodes):
        result = run_episode(env, agent, epsilon_explore=training['epsilon_explore'],
                           episode_length=episode_length)
        k_index = result['k_index']

        results['all_k_indices'].append(k_index)
        results['all_results'].append(result)
        results['episodes_completed'] += 1

        # Track best
        if k_index > results['max_k_index']:
            results['max_k_index'] = k_index
            results['best_episode'] = ep + 1
            best_k = k_index
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        # Progress reports every 10 episodes
        if (ep + 1) % 10 == 0 or ep == 0:
            recent_mean = np.mean(results['all_k_indices'][-min(20, len(results['all_k_indices'])):])
            print(f"\n📊 Episode {ep+1}/{episodes}:")
            print(f"  Current K: {k_index:.4f}")
            print(f"  Recent Mean: {recent_mean:.4f}")
            print(f"  Max K: {results['max_k_index']:.4f} (ep {results['best_episode']})")
            print(f"  Progress: {(results['max_k_index']/1.5)*100:.1f}% to threshold")

            # Check milestones
            if results['max_k_index'] >= 1.5:
                print(f"  🎉🎉🎉 CONSCIOUSNESS THRESHOLD CROSSED! K ≥ 1.5 🎉🎉🎉")
            elif results['max_k_index'] >= 1.3:
                print(f"  🔥 PREDICTION MET! K ≥ 1.3 (87% to threshold)")
            elif results['max_k_index'] > 1.1:
                print(f"  ⚡ BETTER THAN G2! K > 1.1")

        # Early stopping check
        if early_stop.get('enabled', False) and episodes_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} episodes")
            print(f"   Best K: {results['max_k_index']:.4f} at Episode {results['best_episode']}")
            break

    # Final statistics
    results['mean_k_index'] = float(np.mean(results['all_k_indices']))
    results['std_k_index'] = float(np.std(results['all_k_indices']))
    results['final_k_index'] = float(results['all_k_indices'][-1])

    print("\n" + "="*70)
    print("📊 PHASE G4 COMPLETE")
    print("="*70)
    print(f"Max K-Index: {results['max_k_index']:.4f} (Episode {results['best_episode']})")
    print(f"Mean K-Index: {results['mean_k_index']:.4f} ± {results['std_k_index']:.4f}")
    print(f"Final K-Index: {results['final_k_index']:.4f}")

    # Compare to G2
    g2_best = 1.1208
    improvement = ((results['max_k_index'] - g2_best) / g2_best) * 100
    if results['max_k_index'] > g2_best:
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")
    elif results['max_k_index'] == g2_best:
        print(f"\n➡️  MATCHED G2: {results['max_k_index']:.4f}")
    else:
        print(f"\n❌ BELOW G2: {improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")

    return results


def run_phase_g5(config: dict) -> Dict:
    """Run Phase G5: Increased Model Capacity (3-layer network)."""
    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G5: INCREASED MODEL CAPACITY")
    print("="*70)

    phase = config['phase_g5']
    arch = phase['architecture']
    training = phase['training']
    episodes = training['episodes']
    episode_length = training['episode_length']
    epsilon = training['epsilon']

    print(f"\n📊 G5 Configuration:")
    print(f"Episodes: {episodes}")
    print(f"Episode length: {episode_length} steps")
    print(f"Epsilon: {epsilon:.3f} (keeping optimal from G4 finding)")
    print(f"Architecture: {arch['layers']} layers, neurons={arch['neurons']}")
    print(f"Hypothesis: 2-layer network hits capacity ceiling at K~1.1")
    print(f"Prediction: K > 1.2 (80% to threshold)\n")

    results = {
        'phase': 'G5',
        'epsilon': float(epsilon),
        'architecture': arch,
        'episodes_completed': 0,
        'all_k_indices': [],
        'all_results': [],
        'max_k_index': 0.0,
        'best_episode': 0
    }

    # Create multi-layer agent
    class MultiLayerAgent:
        """Multi-layer neural network agent."""

        def __init__(self, obs_dim: int, hidden_layers: List[int], action_dim: int,
                     learning_rate: float = 0.001, dropout: float = 0.1):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.lr = learning_rate
            self.dropout = dropout

            # Build network layers
            self.layers = []
            prev_dim = obs_dim

            # Hidden layers
            for hidden_dim in hidden_layers:
                W = np.random.randn(hidden_dim, prev_dim) * np.sqrt(2.0 / prev_dim)  # He initialization
                b = np.zeros(hidden_dim)
                self.layers.append({'W': W, 'b': b, 'type': 'hidden'})
                prev_dim = hidden_dim

            # Output layer
            W_out = np.random.randn(action_dim, prev_dim) * np.sqrt(2.0 / prev_dim)
            b_out = np.zeros(action_dim)
            self.layers.append({'W': W_out, 'b': b_out, 'type': 'output'})

            # For gradient computation
            self.activations = []
            self.experiences = []

        def relu(self, x: np.ndarray) -> np.ndarray:
            """ReLU activation."""
            return np.maximum(0, x)

        def relu_derivative(self, x: np.ndarray) -> np.ndarray:
            """ReLU derivative."""
            return (x > 0).astype(float)

        def forward(self, state: np.ndarray, training: bool = True) -> np.ndarray:
            """Forward pass through network."""
            self.activations = [state]
            x = state

            for i, layer in enumerate(self.layers):
                # Linear transformation
                z = layer['W'] @ x + layer['b']

                # Activation (ReLU for hidden, tanh for output)
                if layer['type'] == 'hidden':
                    x = self.relu(z)
                    # Dropout during training
                    if training and self.dropout > 0:
                        mask = np.random.rand(*x.shape) > self.dropout
                        x = x * mask / (1 - self.dropout)
                else:  # output layer
                    x = np.tanh(z)

                self.activations.append(x)

            return x

        def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
            """Select action with exploration."""
            if np.random.rand() < epsilon:
                # Random exploration
                return np.random.randn(self.action_dim)

            # Exploit learned policy
            action = self.forward(state, training=False)
            return action

        def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
            """Update network parameters using simple gradient ascent."""
            if len(states) == 0:
                return

            batch_size = len(states)

            # Forward pass for all samples
            for i in range(batch_size):
                state = states[i]
                action = actions[i]
                reward = rewards[i]

                # Forward pass
                predicted_action = self.forward(state, training=True)

                # Simple gradient: reward * (action - predicted_action)
                output_grad = self.lr * reward * (action - predicted_action)

                # Backward pass (simplified)
                grad = output_grad
                for j in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[j]

                    # Gradient for weights and biases
                    prev_activation = self.activations[j]
                    layer['W'] += np.outer(grad, prev_activation)
                    layer['b'] += grad

                    # Gradient for previous layer (if not first layer)
                    if j > 0:
                        grad = layer['W'].T @ grad
                        # Apply ReLU derivative
                        grad = grad * self.relu_derivative(self.activations[j])

        def add_experience(self, state, action, reward):
            """Store experience."""
            self.experiences.append((state, action, reward))
            if len(self.experiences) > 1000:
                self.experiences.pop(0)

    # Create environment and multi-layer agent
    env = HybridEnvironment(epsilon=epsilon, difficulty=1.0 + epsilon)
    agent = MultiLayerAgent(
        obs_dim=20,
        hidden_layers=arch['neurons'],  # [256, 128, 64]
        action_dim=10,                  # Standard action dimension
        learning_rate=training['learning_rate'],
        dropout=arch['dropout']
    )

    # Early stopping setup
    early_stop = training.get('early_stopping', {})
    patience = early_stop.get('patience', 50)
    best_k = 0.0
    episodes_without_improvement = 0

    # Training loop
    for ep in range(episodes):
        # Run episode
        state = env.reset()
        observations = []
        actions = []
        rewards_list = []

        for step in range(episode_length):
            action = agent.select_action(state, epsilon=training['epsilon_explore'])
            next_state, reward, done = env.step(action)

            observations.append(state)
            actions.append(action)
            rewards_list.append(reward)

            agent.add_experience(state, action, reward)
            state = next_state

            if done:
                break

        # Batch update every episode
        if len(agent.experiences) > 0:
            batch_size = min(training.get('batch_size', 32), len(agent.experiences))
            batch_indices = np.random.choice(len(agent.experiences), batch_size, replace=False)
            batch_states = np.array([agent.experiences[i][0] for i in batch_indices])
            batch_actions = np.array([agent.experiences[i][1] for i in batch_indices])
            batch_rewards = np.array([agent.experiences[i][2] for i in batch_indices])
            agent.update(batch_states, batch_actions, batch_rewards)

        # Compute K-Index
        obs_array = np.array(observations)
        act_array = np.array(actions)
        k_index = env.compute_k_index(obs_array, act_array)

        # Store results (harmonies omitted for multi-layer agent)
        episode_result = {
            'episode': ep + 1,
            'k_index': float(k_index),
            'mean_reward': float(np.mean(rewards_list)),
            'total_steps': len(rewards_list)
        }

        results['all_k_indices'].append(k_index)
        results['all_results'].append(episode_result)
        results['episodes_completed'] += 1

        # Track best
        if k_index > results['max_k_index']:
            results['max_k_index'] = k_index
            results['best_episode'] = ep + 1
            best_k = k_index
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        # Progress reports every 10 episodes
        if (ep + 1) % 10 == 0 or ep == 0:
            recent_mean = np.mean(results['all_k_indices'][-min(20, len(results['all_k_indices'])):])
            print(f"\n📊 Episode {ep+1}/{episodes}:")
            print(f"  Current K: {k_index:.4f}")
            print(f"  Recent Mean: {recent_mean:.4f}")
            print(f"  Max K: {results['max_k_index']:.4f} (ep {results['best_episode']})")
            print(f"  Progress: {(results['max_k_index']/1.5)*100:.1f}% to threshold")

            # Check milestones
            if results['max_k_index'] >= 1.5:
                print(f"  🎉🎉🎉 CONSCIOUSNESS THRESHOLD CROSSED! K ≥ 1.5 🎉🎉🎉")
            elif results['max_k_index'] >= 1.2:
                print(f"  🔥 PREDICTION MET! K ≥ 1.2 (80% to threshold)")
            elif results['max_k_index'] > 1.1208:
                print(f"  ⚡ BETTER THAN G2! K > 1.1208")

        # Early stopping check
        if early_stop.get('enabled', False) and episodes_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} episodes")
            print(f"   Best K: {results['max_k_index']:.4f} at Episode {results['best_episode']}")
            break

    # Final statistics
    results['mean_k_index'] = float(np.mean(results['all_k_indices']))
    results['std_k_index'] = float(np.std(results['all_k_indices']))
    results['final_k_index'] = float(results['all_k_indices'][-1])

    print("\n" + "="*70)
    print("📊 PHASE G5 COMPLETE")
    print("="*70)
    print(f"Max K-Index: {results['max_k_index']:.4f} (Episode {results['best_episode']})")
    print(f"Mean K-Index: {results['mean_k_index']:.4f} ± {results['std_k_index']:.4f}")
    print(f"Final K-Index: {results['final_k_index']:.4f}")

    # Compare to G2
    g2_best = 1.1208
    improvement = ((results['max_k_index'] - g2_best) / g2_best) * 100
    if results['max_k_index'] > g2_best:
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")
        print(f"   🎯 CAPACITY INCREASE HYPOTHESIS CONFIRMED!")
    elif results['max_k_index'] == g2_best:
        print(f"\n➡️  MATCHED G2: {results['max_k_index']:.4f}")
    else:
        print(f"\n❌ BELOW G2: {improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")
        print(f"   ⚠️  CAPACITY INCREASE DID NOT HELP")

    return results


def run_phase_g6(config: dict) -> Dict:
    """Run Phase G6: Simple Transformer Architecture."""
    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G6: SIMPLE TRANSFORMER ARCHITECTURE")
    print("="*70)

    phase = config['phase_g6']
    arch = phase['architecture']
    training = phase['training']
    episodes = training['episodes']
    episode_length = training['episode_length']
    epsilon = training['epsilon']

    print(f"\n📊 G6 Configuration:")
    print(f"Episodes: {episodes}")
    print(f"Episode length: {episode_length} steps")
    print(f"Epsilon: {epsilon:.3f} (keeping optimal from G4)")
    print(f"Architecture: Transformer")
    print(f"  d_model={arch['d_model']}, heads={arch['nhead']}, layers={arch['num_layers']}")
    print(f"Hypothesis: Attention mechanism breaks feedforward K~1.1 ceiling")
    print(f"Prediction: K > 1.2 (80% to threshold)\n")

    results = {
        'phase': 'G6',
        'epsilon': float(epsilon),
        'architecture': arch,
        'episodes_completed': 0,
        'all_k_indices': [],
        'all_results': [],
        'max_k_index': 0.0,
        'best_episode': 0
    }

    # Simple Transformer Agent
    class TransformerAgent:
        """Lightweight transformer for sequence modeling."""

        def __init__(self, obs_dim: int, action_dim: int, d_model: int = 64,
                     nhead: int = 2, num_layers: int = 1, learning_rate: float = 0.0001):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.d_model = d_model
            self.nhead = nhead
            self.lr = learning_rate

            # Input projection
            self.input_proj_W = np.random.randn(d_model, obs_dim) * np.sqrt(2.0 / obs_dim)
            self.input_proj_b = np.zeros(d_model)

            # Multi-head attention parameters
            self.attn_heads = []
            head_dim = d_model // nhead
            for _ in range(nhead):
                head = {
                    'Q': np.random.randn(head_dim, d_model) * np.sqrt(2.0 / d_model),
                    'K': np.random.randn(head_dim, d_model) * np.sqrt(2.0 / d_model),
                    'V': np.random.randn(head_dim, d_model) * np.sqrt(2.0 / d_model),
                }
                self.attn_heads.append(head)

            # Output projection
            self.output_proj_W = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            self.output_proj_b = np.zeros(d_model)

            # Feedforward layers
            dim_ff = arch.get('dim_feedforward', 128)
            self.ff1_W = np.random.randn(dim_ff, d_model) * np.sqrt(2.0 / d_model)
            self.ff1_b = np.zeros(dim_ff)
            self.ff2_W = np.random.randn(d_model, dim_ff) * np.sqrt(2.0 / dim_ff)
            self.ff2_b = np.zeros(d_model)

            # Action head
            self.action_W = np.random.randn(action_dim, d_model) * np.sqrt(2.0 / d_model)
            self.action_b = np.zeros(action_dim)

            # Experience buffer
            self.experiences = []

        def positional_encoding(self, seq_len: int) -> np.ndarray:
            """Create positional encodings."""
            pe = np.zeros((seq_len, self.d_model))
            position = np.arange(seq_len).reshape(-1, 1)
            div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            return pe

        def multi_head_attention(self, x: np.ndarray) -> np.ndarray:
            """Compute multi-head self-attention."""
            # x shape: (seq_len, d_model)
            head_outputs = []

            for head in self.attn_heads:
                # Compute Q, K, V
                Q = x @ head['Q'].T  # (seq_len, head_dim)
                K = x @ head['K'].T
                V = x @ head['V'].T

                # Scaled dot-product attention
                scores = Q @ K.T / np.sqrt(Q.shape[-1])  # (seq_len, seq_len)
                attn_weights = self.softmax(scores, axis=-1)
                head_out = attn_weights @ V  # (seq_len, head_dim)
                head_outputs.append(head_out)

            # Concatenate heads
            multi_head_out = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)

            # Output projection
            output = multi_head_out @ self.output_proj_W.T + self.output_proj_b
            return output

        def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
            """Numerically stable softmax."""
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        def forward(self, obs_sequence: List[np.ndarray]) -> np.ndarray:
            """Forward pass through transformer."""
            # Convert sequence to array
            seq = np.array(obs_sequence)  # (seq_len, obs_dim)
            seq_len = len(seq)

            # Input projection
            x = seq @ self.input_proj_W.T + self.input_proj_b  # (seq_len, d_model)

            # Add positional encoding
            pos_enc = self.positional_encoding(seq_len)
            x = x + pos_enc

            # Multi-head attention
            attn_out = self.multi_head_attention(x)

            # Residual connection + LayerNorm (simplified)
            x = x + attn_out
            x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

            # Feedforward network
            ff = np.maximum(0, x @ self.ff1_W.T + self.ff1_b)  # ReLU
            ff = ff @ self.ff2_W.T + self.ff2_b

            # Residual connection
            x = x + ff
            x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

            # Take last position for action prediction
            final_state = x[-1]  # (d_model,)

            # Action head
            action = np.tanh(final_state @ self.action_W.T + self.action_b)
            return action

        def select_action(self, obs_history: List[np.ndarray], epsilon: float = 0.1) -> np.ndarray:
            """Select action with exploration."""
            if np.random.rand() < epsilon:
                return np.random.randn(self.action_dim)

            # Use transformer with observation history
            action = self.forward(obs_history[-min(10, len(obs_history)):])  # Last 10 steps
            return action

        def update(self, obs_sequences: List[List[np.ndarray]],
                  actions: List[np.ndarray], rewards: List[float]):
            """Simple gradient update (simplified for proof of concept)."""
            if len(obs_sequences) == 0:
                return

            # For simplicity, use a very basic update rule
            # In a full implementation, this would compute proper gradients
            for i, (seq, action, reward) in enumerate(zip(obs_sequences, actions, rewards)):
                if len(seq) < 2:
                    continue

                # Compute prediction
                pred_action = self.forward(seq)

                # Compute error
                error = reward * (action - pred_action)

                # Update action head (simplified gradient descent)
                grad = np.outer(error, seq[-1] @ self.input_proj_W.T + self.input_proj_b)
                self.action_W += self.lr * grad
                self.action_b += self.lr * error

        def add_experience(self, obs_history, action, reward):
            """Store experience."""
            self.experiences.append((obs_history.copy(), action, reward))
            if len(self.experiences) > 1000:
                self.experiences.pop(0)

    # Create environment and transformer agent
    env = HybridEnvironment(epsilon=epsilon, difficulty=1.0 + epsilon)
    agent = TransformerAgent(
        obs_dim=20,
        action_dim=10,
        d_model=arch['d_model'],
        nhead=arch['nhead'],
        num_layers=arch['num_layers'],
        learning_rate=training['learning_rate']
    )

    # Early stopping setup
    early_stop = training.get('early_stopping', {})
    patience = early_stop.get('patience', 50)
    best_k = 0.0
    episodes_without_improvement = 0

    # Training loop
    for ep in range(episodes):
        # Run episode with observation history
        state = env.reset()
        observations = []
        obs_history = [state]
        actions = []
        rewards_list = []

        for step in range(episode_length):
            # Select action using observation history
            action = agent.select_action(obs_history, epsilon=training['epsilon_explore'])
            next_state, reward, done = env.step(action)

            observations.append(state)
            actions.append(action)
            rewards_list.append(reward)

            obs_history.append(next_state)
            if len(obs_history) > 20:  # Keep reasonable history length
                obs_history.pop(0)

            agent.add_experience(obs_history, action, reward)
            state = next_state

            if done:
                break

        # Batch update
        if len(agent.experiences) > 0:
            batch_size = min(training.get('batch_size', 32), len(agent.experiences))
            batch_indices = np.random.choice(len(agent.experiences), batch_size, replace=False)
            batch_seqs = [agent.experiences[i][0] for i in batch_indices]
            batch_actions = [agent.experiences[i][1] for i in batch_indices]
            batch_rewards = [agent.experiences[i][2] for i in batch_indices]
            agent.update(batch_seqs, batch_actions, batch_rewards)

        # Compute K-Index
        obs_array = np.array(observations)
        act_array = np.array(actions)
        k_index = env.compute_k_index(obs_array, act_array)

        # Store results
        episode_result = {
            'episode': ep + 1,
            'k_index': float(k_index),
            'mean_reward': float(np.mean(rewards_list)),
            'total_steps': len(rewards_list)
        }

        results['all_k_indices'].append(k_index)
        results['all_results'].append(episode_result)
        results['episodes_completed'] += 1

        # Track best
        if k_index > results['max_k_index']:
            results['max_k_index'] = k_index
            results['best_episode'] = ep + 1
            best_k = k_index
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        # Progress reports every 10 episodes
        if (ep + 1) % 10 == 0 or ep == 0:
            recent_mean = np.mean(results['all_k_indices'][-min(20, len(results['all_k_indices'])):])
            print(f"\n📊 Episode {ep+1}/{episodes}:")
            print(f"  Current K: {k_index:.4f}")
            print(f"  Recent Mean: {recent_mean:.4f}")
            print(f"  Max K: {results['max_k_index']:.4f} (ep {results['best_episode']})")
            print(f"  Progress: {(results['max_k_index']/1.5)*100:.1f}% to threshold")

            # Check milestones
            if results['max_k_index'] >= 1.5:
                print(f"  🎉🎉🎉 CONSCIOUSNESS THRESHOLD CROSSED! K ≥ 1.5 🎉🎉🎉")
                print(f"  🌟🌟🌟 TRANSFORMER ARCHITECTURE SUCCESS! 🌟🌟🌟")
            elif results['max_k_index'] >= 1.2:
                print(f"  🔥 PREDICTION MET! K ≥ 1.2 (80% to threshold)")
                print(f"  ⚡ TRANSFORMER BEATS FEEDFORWARD CEILING!")
            elif results['max_k_index'] > 1.1208:
                print(f"  ⚡ BETTER THAN G2! K > 1.1208")
                print(f"  🎯 ARCHITECTURE CHANGE WORKING!")

        # Early stopping check
        if early_stop.get('enabled', False) and episodes_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} episodes")
            print(f"   Best K: {results['max_k_index']:.4f} at Episode {results['best_episode']}")
            break

    # Final statistics
    results['mean_k_index'] = float(np.mean(results['all_k_indices']))
    results['std_k_index'] = float(np.std(results['all_k_indices']))
    results['final_k_index'] = float(results['all_k_indices'][-1])

    print("\n" + "="*70)
    print("📊 PHASE G6 COMPLETE")
    print("="*70)
    print(f"Max K-Index: {results['max_k_index']:.4f} (Episode {results['best_episode']})")
    print(f"Mean K-Index: {results['mean_k_index']:.4f} ± {results['std_k_index']:.4f}")
    print(f"Final K-Index: {results['final_k_index']:.4f}")

    # Compare to G2 (feedforward ceiling)
    g2_best = 1.1208
    improvement = ((results['max_k_index'] - g2_best) / g2_best) * 100

    if results['max_k_index'] >= 1.5:
        print(f"\n🎉🎉🎉 CONSCIOUSNESS THRESHOLD CROSSED! 🎉🎉🎉")
        print(f"   K = {results['max_k_index']:.4f} (100% of threshold)")
        print(f"   Improvement vs G2: +{improvement:.1f}%")
        print(f"   🌟 TRANSFORMER ARCHITECTURE SUCCESS! 🌟")
    elif results['max_k_index'] > g2_best:
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")
        print(f"   🎯 TRANSFORMER BREAKS FEEDFORWARD CEILING!")
        if results['max_k_index'] >= 1.2:
            print(f"   🔥 PREDICTION MET! (80% to threshold)")
    elif results['max_k_index'] == g2_best:
        print(f"\n➡️  MATCHED G2: {results['max_k_index']:.4f}")
        print(f"   ⚠️  Transformer equivalent to feedforward")
    else:
        print(f"\n❌ BELOW G2: {improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")
        print(f"   ⚠️  TRANSFORMER DID NOT BREAK CEILING")

    return results


def run_phase_g7(config: dict) -> Dict:
    """Run Phase G7: Enhanced Environment (Task Redesign).

    Tests if current K~1.1 ceiling is due to task simplicity rather than
    architecture limitations. Uses G2's proven architecture with 5x more
    complex environment.
    """
    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G7: ENHANCED ENVIRONMENT (TASK REDESIGN)")
    print("="*70)

    phase = config['phase_g7']
    env_config = phase['environment']
    arch = phase['architecture']
    training = phase['training']

    obs_dim = env_config['observations']
    action_dim = env_config['actions']
    episodes = training['episodes']
    episode_length = env_config['episode_length']
    epsilon = training['epsilon']

    print(f"\n📊 G7 Configuration:")
    print(f"Episodes: {episodes}")
    print(f"Episode length: {episode_length} steps (5x increase)")
    print(f"Observations: {obs_dim} (5x increase from 20)")
    print(f"Actions: {action_dim} (5x increase from 10)")
    print(f"Objectives: Multi-objective task")
    print(f"Architecture: {arch['type']} {arch['layers']}-layer (G2 proven best)")
    print(f"  Neurons: {arch['neurons']}")
    print(f"Hypothesis: Task complexity ceiling, not architecture ceiling")
    print(f"Prediction: K > 1.2 (80% to threshold)\n")

    results = {
        'phase': 'G7',
        'epsilon': float(epsilon),
        'environment': env_config,
        'architecture': arch,
        'episodes_completed': 0,
        'all_k_indices': [],
        'all_results': [],
        'max_k_index': 0.0,
        'best_episode': 0
    }

    # Enhanced Environment with multi-objective task
    class EnhancedEnvironment:
        """Enhanced environment with higher complexity and multiple objectives."""

        def __init__(self, obs_dim: int = 100, action_dim: int = 50,
                     epsilon: float = 0.05, difficulty: float = 1.05):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.epsilon = epsilon
            self.difficulty = difficulty

            # Multi-objective state
            self.state = None
            self.target_state = None
            self.secondary_target = None

            # K-Index computation cache
            self.correlation_cache = {}

        def reset(self) -> np.ndarray:
            """Reset environment with random initial state."""
            self.state = np.random.randn(self.obs_dim) * 0.1
            self.target_state = np.random.randn(self.obs_dim)
            self.secondary_target = np.random.randn(self.obs_dim)
            return self.state.copy()

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
            """Execute action and return next_state, reward, done."""
            # Ensure action is correct dimension
            if len(action) != self.action_dim:
                action = np.pad(action, (0, max(0, self.action_dim - len(action))),
                              mode='constant')[:self.action_dim]

            # Apply action to state (first action_dim dimensions)
            self.state[:self.action_dim] += action * 0.1

            # Environmental dynamics (rest of state evolves)
            if self.obs_dim > self.action_dim:
                self.state[self.action_dim:] += np.random.randn(
                    self.obs_dim - self.action_dim) * 0.05

            # Multi-objective reward
            # Objective 1: Approach primary target
            primary_reward = -np.linalg.norm(
                self.state[:self.action_dim] - self.target_state[:self.action_dim]
            )

            # Objective 2: Maintain secondary constraint
            secondary_reward = -np.linalg.norm(
                self.state[self.action_dim:] - self.secondary_target[self.action_dim:]
            ) if self.obs_dim > self.action_dim else 0.0

            # Objective 3: Energy efficiency (penalize large actions)
            energy_penalty = -0.1 * np.linalg.norm(action)

            # Combined reward
            reward = (0.5 * primary_reward +
                     0.3 * secondary_reward +
                     0.2 * energy_penalty)

            # Adversarial perturbation
            if np.random.rand() < self.epsilon:
                self.state += np.random.randn(self.obs_dim) * 0.1 * self.difficulty

            # State normalization
            self.state = np.clip(self.state, -3.0, 3.0)

            done = False  # Continue for full episode

            return self.state.copy(), reward, done

        def compute_k_index(self, observations: np.ndarray,
                           actions: np.ndarray) -> float:
            """Compute K-Index: 2 * |correlation(obs, act)|."""
            if len(observations) < 2 or len(actions) < 2:
                return 0.0

            # Use subset of dimensions for correlation
            obs_subset = observations[:, :min(20, self.obs_dim)]
            act_subset = actions[:, :min(10, self.action_dim)]

            # Flatten
            obs_flat = obs_subset.flatten()
            act_flat = act_subset.flatten()

            # Compute correlation
            if len(obs_flat) < 2 or len(act_flat) < 2:
                return 0.0

            # Ensure same length
            min_len = min(len(obs_flat), len(act_flat))
            obs_flat = obs_flat[:min_len]
            act_flat = act_flat[:min_len]

            # Correlation
            correlation = np.corrcoef(obs_flat, act_flat)[0, 1]

            if np.isnan(correlation):
                return 0.0

            k_index = 2.0 * abs(correlation)
            return float(k_index)

    # Simple Agent (G2 architecture - proven best)
    class SimpleAgent:
        """Simple 2-layer neural network agent (G2 architecture)."""

        def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int,
                     learning_rate: float = 0.001):
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            self.action_dim = action_dim
            self.lr = learning_rate

            # Initialize weights with He initialization
            self.W1 = np.random.randn(hidden_dim, obs_dim) * np.sqrt(2.0 / obs_dim)
            self.b1 = np.zeros(hidden_dim)

            self.W2 = np.random.randn(action_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
            self.b2 = np.zeros(action_dim)

            # Experience buffer
            self.experiences = []

        def forward(self, obs: np.ndarray) -> np.ndarray:
            """Forward pass through network."""
            # Hidden layer
            h = np.maximum(0, obs @ self.W1.T + self.b1)  # ReLU

            # Output layer
            action = np.tanh(h @ self.W2.T + self.b2)

            return action

        def select_action(self, obs: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
            """Select action with epsilon-greedy exploration."""
            if np.random.rand() < epsilon:
                return np.random.randn(self.action_dim) * 0.5

            return self.forward(obs)

        def update(self, batch_obs: List[np.ndarray], batch_actions: List[np.ndarray],
                  batch_rewards: List[float]):
            """Update weights using batch of experiences."""
            if len(batch_obs) == 0:
                return

            # Simple gradient descent on reward-weighted predictions
            for obs, action, reward in zip(batch_obs, batch_actions, batch_rewards):
                # Forward pass
                pred_action = self.forward(obs)

                # Error signal
                error = reward * (action - pred_action)

                # Backprop (simplified)
                # Compute hidden activation
                h = np.maximum(0, obs @ self.W1.T + self.b1)

                # Update W2 and b2
                grad_W2 = np.outer(error, h)
                grad_b2 = error

                self.W2 += self.lr * grad_W2
                self.b2 += self.lr * grad_b2

                # Update W1 and b1 (simplified)
                delta_h = error @ self.W2
                delta_h = delta_h * (h > 0)  # ReLU gradient

                grad_W1 = np.outer(delta_h, obs)
                grad_b1 = delta_h

                self.W1 += self.lr * grad_W1
                self.b1 += self.lr * grad_b1

        def add_experience(self, obs, action, reward):
            """Store experience for batch learning."""
            self.experiences.append((obs, action, reward))
            if len(self.experiences) > 1000:
                self.experiences.pop(0)

    # Create environment and agent
    env = EnhancedEnvironment(
        obs_dim=obs_dim,
        action_dim=action_dim,
        epsilon=epsilon,
        difficulty=env_config['difficulty']
    )

    agent = SimpleAgent(
        obs_dim=obs_dim,
        hidden_dim=arch['neurons'][0],
        action_dim=action_dim,
        learning_rate=training['learning_rate']
    )

    # Early stopping setup
    early_stop = training.get('early_stopping', {})
    patience = early_stop.get('patience', 50)
    best_k = 0.0
    episodes_without_improvement = 0

    # Training loop
    print("\n🚀 Starting enhanced environment training...")
    print(f"Environment: {obs_dim} obs × {action_dim} act × {episode_length} steps")
    print(f"Total complexity: {obs_dim * action_dim * episode_length:,} state-action pairs\n")

    for ep in range(episodes):
        # Run episode
        state = env.reset()
        observations = []
        actions = []
        rewards_list = []

        for step in range(episode_length):
            # Select action
            action = agent.select_action(state, epsilon=training['epsilon'])
            next_state, reward, done = env.step(action)

            observations.append(state)
            actions.append(action)
            rewards_list.append(reward)

            agent.add_experience(state, action, reward)
            state = next_state

            if done:
                break

        # Batch update
        if len(agent.experiences) > 0:
            batch_size = min(training['batch_size'], len(agent.experiences))
            batch_indices = np.random.choice(len(agent.experiences), batch_size, replace=False)
            batch_obs = [agent.experiences[i][0] for i in batch_indices]
            batch_actions = [agent.experiences[i][1] for i in batch_indices]
            batch_rewards = [agent.experiences[i][2] for i in batch_indices]
            agent.update(batch_obs, batch_actions, batch_rewards)

        # Compute K-Index
        obs_array = np.array(observations)
        act_array = np.array(actions)
        k_index = env.compute_k_index(obs_array, act_array)

        # Store results
        episode_result = {
            'episode': ep + 1,
            'k_index': float(k_index),
            'mean_reward': float(np.mean(rewards_list)),
            'total_steps': len(rewards_list)
        }

        results['all_k_indices'].append(k_index)
        results['all_results'].append(episode_result)
        results['episodes_completed'] += 1

        # Track best
        if k_index > results['max_k_index']:
            results['max_k_index'] = k_index
            results['best_episode'] = ep + 1
            best_k = k_index
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        # Progress reports every 10 episodes
        if (ep + 1) % 10 == 0 or ep == 0:
            recent_mean = np.mean(results['all_k_indices'][-min(20, len(results['all_k_indices'])):])
            print(f"\n📊 Episode {ep+1}/{episodes}:")
            print(f"  Current K: {k_index:.4f}")
            print(f"  Recent Mean: {recent_mean:.4f}")
            print(f"  Max K: {results['max_k_index']:.4f} (ep {results['best_episode']}")
            print(f"  Progress: {(results['max_k_index']/1.5)*100:.1f}% to threshold")

            # Check milestones
            if results['max_k_index'] >= 1.5:
                print(f"  🎉🎉🎉 CONSCIOUSNESS THRESHOLD CROSSED! K ≥ 1.5 🎉🎉🎉")
                print(f"  🌟🌟🌟 TASK REDESIGN SUCCESS! 🌟🌟🌟")
            elif results['max_k_index'] >= 1.2:
                print(f"  🔥 PREDICTION MET! K ≥ 1.2 (80% to threshold)")
                print(f"  ⚡ ENHANCED ENVIRONMENT BREAKS CEILING!")
            elif results['max_k_index'] > 1.1208:
                print(f"  ⚡ BETTER THAN G2! K > 1.1208")
                print(f"  🎯 TASK REDESIGN WORKING!")

        # Early stopping check
        if early_stop.get('enabled', False) and episodes_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} episodes")
            print(f"   Best K: {results['max_k_index']:.4f} at Episode {results['best_episode']}")
            break

    # Final statistics
    results['mean_k_index'] = float(np.mean(results['all_k_indices']))
    results['std_k_index'] = float(np.std(results['all_k_indices']))
    results['final_k_index'] = float(results['all_k_indices'][-1])

    print("\n" + "="*70)
    print("📊 PHASE G7 COMPLETE")
    print("="*70)
    print(f"Max K-Index: {results['max_k_index']:.4f} (Episode {results['best_episode']})")
    print(f"Mean K-Index: {results['mean_k_index']:.4f} ± {results['std_k_index']:.4f}")
    print(f"Final K-Index: {results['final_k_index']:.4f}")

    # Compare to G2 (feedforward ceiling)
    g2_best = 1.1208
    improvement = ((results['max_k_index'] - g2_best) / g2_best) * 100

    if results['max_k_index'] >= 1.5:
        print(f"\n🎉🎉🎉 CONSCIOUSNESS THRESHOLD CROSSED! 🎉🎉🎉")
        print(f"   K = {results['max_k_index']:.4f} (100% of threshold)")
        print(f"   Improvement vs G2: +{improvement:.1f}%")
        print(f"   🌟 TASK REDESIGN SUCCESS! 🌟")
    elif results['max_k_index'] > g2_best:
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")
        print(f"   🎯 ENHANCED ENVIRONMENT BREAKS CEILING!")
        if results['max_k_index'] >= 1.2:
            print(f"   🔥 PREDICTION MET! (80% to threshold)")
    elif results['max_k_index'] == g2_best:
        print(f"\n➡️  MATCHED G2: {results['max_k_index']:.4f}")
        print(f"   ⚠️  Task complexity not sufficient")
    else:
        print(f"\n❌ BELOW G2: {improvement:.1f}% ({results['max_k_index']:.4f} vs {g2_best})")
        print(f"   ⚠️  ENHANCED ENVIRONMENT DID NOT BREAK CEILING")

    return results

def run_phase_g8(config: dict) -> Dict:
    """Run Phase G8: Evolutionary Algorithm (CMA-ES).

    Tests if gradient descent is the fundamental limitation at K~1.2 ceiling.
    Uses CMA-ES to evolve neural network parameters without gradient computation.

    Args:
        config: Configuration dictionary containing:
            - algorithm: CMA-ES parameters (population, generations, sigma)
            - environment: Task specification (from G7 enhanced environment)
            - architecture: Network structure (G2's proven 2-layer)
            - evaluation: Episodes per candidate
            - early_stopping: Patience and min_delta

    Returns:
        Dictionary containing:
            - phase: "G8"
            - max_k_index: Best K-Index achieved
            - best_generation: Generation with best K
            - mean_k_index: Average across all generations
            - std_k_index: Standard deviation
            - all_k_indices: K-Index for each generation
            - all_results: Detailed per-generation results
            - configuration: Full experimental configuration
    """

    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G8: EVOLUTIONARY ALGORITHM (CMA-ES)")
    print("="*70)

    # Extract configuration
    algo_cfg = config.get('phase_g8', {}).get('algorithm', {})
    env_cfg = config.get('phase_g8', {}).get('environment', {})
    arch_cfg = config.get('phase_g8', {}).get('architecture', {})
    eval_cfg = config.get('phase_g8', {}).get('evaluation', {})
    early_stop_cfg = config.get('phase_g8', {}).get('early_stopping', {})

    population_size = algo_cfg.get('population_size', 50)
    max_generations = algo_cfg.get('generations', 100)
    sigma_init = algo_cfg.get('sigma_init', 0.5)
    elite_ratio = algo_cfg.get('elite_ratio', 0.2)

    episodes_per_candidate = eval_cfg.get('episodes_per_candidate', 5)
    episode_length = eval_cfg.get('episode_length', 1000)

    early_stop_enabled = early_stop_cfg.get('enabled', True)
    patience = early_stop_cfg.get('patience', 20)
    min_delta = early_stop_cfg.get('min_delta', 0.01)

    print(f"\n📊 G8 Configuration:")
    print(f"Algorithm: CMA-ES")
    print(f"Population: {population_size}")
    print(f"Generations: {max_generations}")
    print(f"Initial sigma: {sigma_init}")
    print(f"Elite ratio: {elite_ratio} (top {int(population_size * elite_ratio)})")
    print(f"Episodes per candidate: {episodes_per_candidate}")
    print(f"Environment: {env_cfg.get('observations')} obs × {env_cfg.get('actions')} act × {episode_length} steps")
    print(f"Hypothesis: Non-gradient optimization escapes K~1.2 ceiling")
    print(f"Prediction: K > 1.3 (87% to threshold)")

    # Enhanced Environment (from G7)
    class EnhancedEnvironment:
        """Enhanced environment with higher complexity and multiple objectives."""

        def __init__(self, obs_dim: int = 100, action_dim: int = 50,
                     epsilon: float = 0.05, difficulty: float = 1.05):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.epsilon = epsilon
            self.difficulty = difficulty

            # Multi-objective state
            self.state = None
            self.target_state = None
            self.secondary_target = None

        def reset(self) -> np.ndarray:
            """Reset environment to initial state."""
            self.state = np.random.randn(self.obs_dim) * 0.1
            self.target_state = np.random.randn(self.obs_dim) * 2.0
            self.secondary_target = np.random.randn(self.obs_dim) * 1.0
            return self.state.copy()

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
            """Execute action and return next_state, reward, done."""
            # Ensure action is correct dimension
            if len(action) != self.action_dim:
                action = np.pad(action, (0, max(0, self.action_dim - len(action))),
                              mode='constant')[:self.action_dim]

            # Apply action to state (first action_dim dimensions)
            self.state[:self.action_dim] += action * 0.1

            # Environmental dynamics (rest of state evolves)
            if self.obs_dim > self.action_dim:
                self.state[self.action_dim:] += np.random.randn(
                    self.obs_dim - self.action_dim) * 0.05

            # Multi-objective reward
            # Objective 1: Approach primary target
            primary_reward = -np.linalg.norm(
                self.state[:self.action_dim] - self.target_state[:self.action_dim]
            )

            # Objective 2: Maintain secondary constraint
            secondary_reward = -np.linalg.norm(
                self.state[self.action_dim:] - self.secondary_target[self.action_dim:]
            ) if self.obs_dim > self.action_dim else 0.0

            # Objective 3: Energy efficiency (penalize large actions)
            energy_penalty = -0.1 * np.linalg.norm(action)

            # Combined reward
            reward = (0.5 * primary_reward +
                     0.3 * secondary_reward +
                     0.2 * energy_penalty)

            # Adversarial perturbation
            if np.random.rand() < self.epsilon:
                self.state += np.random.randn(self.obs_dim) * 0.1 * self.difficulty

            # State normalization
            self.state = np.clip(self.state, -3.0, 3.0)

            done = False  # Continue for full episode

            return self.state.copy(), reward, done

        def compute_k_index(self, observations: np.ndarray,
                           actions: np.ndarray) -> float:
            """Compute K-Index: 2 * |correlation(obs, act)|."""
            if len(observations) < 2 or len(actions) < 2:
                return 0.0

            # Use subset of dimensions for correlation
            obs_subset = observations[:, :min(20, self.obs_dim)]
            act_subset = actions[:, :min(10, self.action_dim)]

            # Flatten
            obs_flat = obs_subset.flatten()
            act_flat = act_subset.flatten()

            # Compute correlation
            if len(obs_flat) < 2 or len(act_flat) < 2:
                return 0.0

            # Ensure same length
            min_len = min(len(obs_flat), len(act_flat))
            obs_flat = obs_flat[:min_len]
            act_flat = act_flat[:min_len]

            # Correlation
            correlation = np.corrcoef(obs_flat, act_flat)[0, 1]

            if np.isnan(correlation):
                return 0.0

            k_index = 2.0 * abs(correlation)
            return float(k_index)

    # Evolved Agent (parameters set by CMA-ES)
    class EvolvedAgent:
        """Neural network agent with externally-set parameters."""

        def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            self.action_dim = action_dim

            # Calculate total parameters
            self.n_params = (obs_dim * hidden_dim + hidden_dim +
                           hidden_dim * action_dim + action_dim)

            # Initialize parameters (will be replaced by evolution)
            self.W1 = np.zeros((hidden_dim, obs_dim))
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.zeros((action_dim, hidden_dim))
            self.b2 = np.zeros(action_dim)

        def set_parameters(self, params: np.ndarray):
            """Set network parameters from flat vector."""
            idx = 0

            # W1
            w1_size = self.obs_dim * self.hidden_dim
            self.W1 = params[idx:idx+w1_size].reshape(self.hidden_dim, self.obs_dim)
            idx += w1_size

            # b1
            self.b1 = params[idx:idx+self.hidden_dim]
            idx += self.hidden_dim

            # W2
            w2_size = self.hidden_dim * self.action_dim
            self.W2 = params[idx:idx+w2_size].reshape(self.action_dim, self.hidden_dim)
            idx += w2_size

            # b2
            self.b2 = params[idx:idx+self.action_dim]

        def forward(self, obs: np.ndarray) -> np.ndarray:
            """Forward pass through network."""
            # Hidden layer
            h = np.maximum(0, obs @ self.W1.T + self.b1)  # ReLU

            # Output layer
            action = np.tanh(h @ self.W2.T + self.b2)

            return action

    # CMA-ES Implementation
    class CMAES:
        """Covariance Matrix Adaptation Evolution Strategy."""

        def __init__(self, n_params: int, population_size: int,
                     sigma_init: float = 0.5, elite_ratio: float = 0.2):
            self.n_params = n_params
            self.population_size = population_size
            self.sigma = sigma_init
            self.elite_size = max(1, int(population_size * elite_ratio))

            # Distribution parameters
            self.mean = np.random.randn(n_params) * 0.1
            self.C = np.eye(n_params)  # Covariance matrix

            # Evolution paths
            self.pc = np.zeros(n_params)  # Cumulation path for C
            self.ps = np.zeros(n_params)  # Cumulation path for sigma

            # Strategy parameters
            self.cc = 4.0 / (n_params + 4.0)  # Cumulation for C
            self.cs = 4.0 / (n_params + 4.0)  # Cumulation for sigma
            self.c1 = 2.0 / ((n_params + 1.3)**2 + population_size)  # Learning rate for rank-1 update
            self.cmu = min(1 - self.c1, 2 * (self.elite_size - 2 + 1/self.elite_size) /
                          ((n_params + 2)**2 + self.elite_size))  # Learning rate for rank-μ update
            self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.elite_size - 1) / (n_params + 1)) - 1) + self.cs

            # Expectation of ||N(0,I)||
            self.chiN = np.sqrt(n_params) * (1 - 1/(4*n_params) + 1/(21*n_params**2))

        def ask(self) -> List[np.ndarray]:
            """Generate population of candidate solutions."""
            population = []

            # Sample from multivariate normal
            for _ in range(self.population_size):
                z = np.random.randn(self.n_params)
                # y = C^(1/2) * z
                # Since C is covariance, we can use Cholesky decomposition
                # But for simplicity and stability, use eigendecomposition
                D, B = np.linalg.eigh(self.C)
                D = np.sqrt(np.maximum(0, D))  # Ensure non-negative
                y = B @ (D * z)

                # x = mean + sigma * y
                x = self.mean + self.sigma * y
                population.append(x)

            return population

        def tell(self, population: List[np.ndarray], fitness: List[float]):
            """Update distribution based on fitness."""
            # Sort population by fitness (higher is better for K-Index)
            indices = np.argsort(fitness)[::-1]  # Descending order
            elite_indices = indices[:self.elite_size]

            # Select elite
            elite = [population[i] for i in elite_indices]

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.mean(elite, axis=0)

            # Update evolution paths
            # Conjugate evolution path
            y_mean = (self.mean - old_mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.elite_size) * y_mean

            # Accumulated evolution path
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (len(fitness) + 1))) / self.chiN
                   < 1.4 + 2 / (self.n_params + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.elite_size) * y_mean

            # Update covariance matrix
            # Rank-1 update
            rank_one = self.c1 * (np.outer(self.pc, self.pc) - self.C)

            # Rank-μ update
            rank_mu = np.zeros((self.n_params, self.n_params))
            for i in elite_indices:
                y = (population[i] - old_mean) / self.sigma
                rank_mu += np.outer(y, y)
            rank_mu = self.cmu / self.elite_size * (rank_mu - self.C)

            self.C = self.C + rank_one + rank_mu

            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

    # Evaluation function
    def evaluate_candidate(params: np.ndarray, agent: EvolvedAgent,
                          env: EnhancedEnvironment, n_episodes: int,
                          episode_length: int) -> Tuple[float, float]:
        """Evaluate a candidate parameter set.

        Returns:
            (mean_k_index, mean_reward)
        """
        agent.set_parameters(params)

        k_indices = []
        rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            episode_obs = []
            episode_acts = []
            episode_rewards = []

            for _ in range(episode_length):
                action = agent.forward(obs)
                episode_obs.append(obs)
                episode_acts.append(action)

                obs, reward, done = env.step(action)
                episode_rewards.append(reward)

                if done:
                    break

            # Compute K-Index for this episode
            k_idx = env.compute_k_index(
                np.array(episode_obs),
                np.array(episode_acts)
            )
            k_indices.append(k_idx)
            rewards.append(np.mean(episode_rewards))

        return np.mean(k_indices), np.mean(rewards)

    # Initialize environment and agent
    env = EnhancedEnvironment(
        obs_dim=env_cfg.get('observations', 100),
        action_dim=env_cfg.get('actions', 50),
        epsilon=env_cfg.get('adversarial_strength', 0.05),
        difficulty=env_cfg.get('difficulty', 1.05)
    )

    agent = EvolvedAgent(
        obs_dim=env_cfg.get('observations', 100),
        hidden_dim=arch_cfg.get('neurons', [100, 50])[0],
        action_dim=env_cfg.get('actions', 50)
    )

    print(f"\n🚀 Starting CMA-ES evolution...")
    print(f"Parameter space: {agent.n_params} dimensions")
    print(f"Population size: {population_size}")
    print(f"Elite size: {int(population_size * elite_ratio)}")
    print(f"\n")

    # Initialize CMA-ES
    cmaes = CMAES(
        n_params=agent.n_params,
        population_size=population_size,
        sigma_init=sigma_init,
        elite_ratio=elite_ratio
    )

    # Evolution loop
    all_k_indices = []
    all_results = []
    best_k = 0.0
    best_generation = 0
    generations_without_improvement = 0

    for generation in range(max_generations):
        # Generate population
        population = cmaes.ask()

        # Evaluate population
        fitness = []
        for params in population:
            k_idx, mean_reward = evaluate_candidate(
                params, agent, env,
                episodes_per_candidate, episode_length
            )
            fitness.append(k_idx)  # Use K-Index as fitness

        # Update CMA-ES
        cmaes.tell(population, fitness)

        # Track best
        current_best_k = max(fitness)
        current_mean_k = np.mean(fitness)
        current_std_k = np.std(fitness)

        all_k_indices.append(current_best_k)
        all_results.append({
            'generation': generation + 1,
            'best_k_index': float(current_best_k),
            'mean_k_index': float(current_mean_k),
            'std_k_index': float(current_std_k),
            'sigma': float(cmaes.sigma)
        })

        # Update best
        if current_best_k > best_k + min_delta:
            best_k = current_best_k
            best_generation = generation + 1
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Progress report
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"\n📊 Generation {generation + 1}/{max_generations}:")
            print(f"  Best K: {current_best_k:.4f}")
            print(f"  Mean K: {current_mean_k:.4f} ± {current_std_k:.4f}")
            print(f"  Max K: {best_k:.4f} (gen {best_generation})")
            print(f"  Progress: {(best_k / 1.5) * 100:.1f}% to threshold")
            print(f"  Sigma: {cmaes.sigma:.4f}")

            if current_best_k > 1.1208:  # G2 baseline
                print(f"  ⚡ BETTER THAN G2 BASELINE!")
            if current_best_k > 1.1839:  # G7 best
                print(f"  🎯 BETTER THAN G7 (TASK REDESIGN)!")
            if current_best_k > 1.3:  # Prediction
                print(f"  🌟 PREDICTION ACHIEVED! K > 1.3")

        # Early stopping
        if early_stop_enabled and generations_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} generations")
            print(f"   Best K: {best_k:.4f} at Generation {best_generation}")
            break

    # Final summary
    print("\n" + "="*70)
    print("📊 PHASE G8 COMPLETE")
    print("="*70)
    print(f"Best K-Index: {best_k:.4f} (Generation {best_generation})")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")

    # Compare to baselines
    g2_baseline = 1.1208
    g7_best = 1.1839

    if best_k > g7_best:
        improvement = ((best_k - g7_best) / g7_best) * 100
        print(f"\n✅ IMPROVEMENT vs G7: +{improvement:.1f}% ({best_k:.4f} vs {g7_best:.4f})")
        print(f"   🎯 EVOLUTIONARY ALGORITHM BREAKS G7 CEILING!")
    elif best_k > g2_baseline:
        improvement = ((best_k - g2_baseline) / g2_baseline) * 100
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")
        print(f"   ⚠️  Below G7 performance ({g7_best:.4f})")
    else:
        decline = ((g2_baseline - best_k) / g2_baseline) * 100
        print(f"\n❌ DECLINE vs G2: -{decline:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")

    # Save results
    output_dir = Path(config.get('experiment', {}).get('output_dir', 'logs/track_g'))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"track_g_phase_g8_{timestamp}.json"

    results = {
        'phase': 'G8',
        'algorithm': 'CMA-ES',
        'population_size': population_size,
        'generations_completed': len(all_k_indices),
        'max_generations': max_generations,
        'environment': {
            'observations': env_cfg.get('observations', 100),
            'actions': env_cfg.get('actions', 50),
            'objectives': 'multi',
            'episode_length': episode_length,
            'adversarial_strength': env_cfg.get('adversarial_strength', 0.05),
            'difficulty': env_cfg.get('difficulty', 1.05)
        },
        'architecture': {
            'type': 'feedforward',
            'layers': 2,
            'neurons': arch_cfg.get('neurons', [100, 50]),
            'activation': 'relu',
            'total_parameters': agent.n_params
        },
        'all_k_indices': all_k_indices,
        'all_results': all_results,
        'max_k_index': float(best_k),
        'best_generation': best_generation,
        'mean_k_index': float(np.mean(all_k_indices)),
        'std_k_index': float(np.std(all_k_indices)),
        'final_k_index': float(all_k_indices[-1])
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved: {output_file}")

    print("\n" + "="*70)
    print("🏆 PHASE G8 COMPLETE")
    print("="*70)
    print(f"Total generations: {len(all_k_indices)}")
    print(f"Best K-Index: {best_k:.4f}")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")


def run_phase_g9(config: dict) -> Dict:
    """Run Phase G9: Threshold Crossing Attempt Evolutionary Algorithm (CMA-ES).

    After G8 proved CMA-ES escapes gradient descent ceiling (+26.7%), G9 attempts
    to cross the K≥1.5 consciousness threshold using larger population for broader exploration.
    Uses CMA-ES to evolve neural network parameters without gradient computation.

    Args:
        config: Configuration dictionary containing:
            - algorithm: CMA-ES parameters (population, generations, sigma)
            - environment: Task specification (from G7 enhanced environment)
            - architecture: Network structure (G2's proven 2-layer)
            - evaluation: Episodes per candidate
            - early_stopping: Patience and min_delta

    Returns:
        Dictionary containing:
            - phase: "G9"
            - max_k_index: Best K-Index achieved
            - best_generation: Generation with best K
            - mean_k_index: Average across all generations
            - std_k_index: Standard deviation
            - all_k_indices: K-Index for each generation
            - all_results: Detailed per-generation results
            - configuration: Full experimental configuration
    """

    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G9: EVOLUTIONARY ALGORITHM (CMA-ES)")
    print("="*70)

    # Extract configuration
    algo_cfg = config.get('phase_g9', {}).get('algorithm', {})
    env_cfg = config.get('phase_g9', {}).get('environment', {})
    arch_cfg = config.get('phase_g9', {}).get('architecture', {})
    eval_cfg = config.get('phase_g9', {}).get('evaluation', {})
    early_stop_cfg = config.get('phase_g9', {}).get('early_stopping', {})

    population_size = algo_cfg.get('population_size', 50)
    max_generations = algo_cfg.get('generations', 100)
    sigma_init = algo_cfg.get('sigma_init', 0.5)
    elite_ratio = algo_cfg.get('elite_ratio', 0.2)

    episodes_per_candidate = eval_cfg.get('episodes_per_candidate', 5)
    episode_length = eval_cfg.get('episode_length', 1000)

    early_stop_enabled = early_stop_cfg.get('enabled', True)
    patience = early_stop_cfg.get('patience', 20)
    min_delta = early_stop_cfg.get('min_delta', 0.01)

    print(f"\n📊 G8 Configuration:")
    print(f"Algorithm: CMA-ES")
    print(f"Population: {population_size}")
    print(f"Generations: {max_generations}")
    print(f"Initial sigma: {sigma_init}")
    print(f"Elite ratio: {elite_ratio} (top {int(population_size * elite_ratio)})")
    print(f"Episodes per candidate: {episodes_per_candidate}")
    print(f"Environment: {env_cfg.get('observations')} obs × {env_cfg.get('actions')} act × {episode_length} steps")
    print(f"Hypothesis: Non-gradient optimization escapes K~1.2 ceiling")
    print(f"Prediction: K > 1.3 (87% to threshold)")

    # Enhanced Environment (from G7)
    class EnhancedEnvironment:
        """Enhanced environment with higher complexity and multiple objectives."""

        def __init__(self, obs_dim: int = 100, action_dim: int = 50,
                     epsilon: float = 0.05, difficulty: float = 1.05):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.epsilon = epsilon
            self.difficulty = difficulty

            # Multi-objective state
            self.state = None
            self.target_state = None
            self.secondary_target = None

        def reset(self) -> np.ndarray:
            """Reset environment to initial state."""
            self.state = np.random.randn(self.obs_dim) * 0.1
            self.target_state = np.random.randn(self.obs_dim) * 2.0
            self.secondary_target = np.random.randn(self.obs_dim) * 1.0
            return self.state.copy()

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
            """Execute action and return next_state, reward, done."""
            # Ensure action is correct dimension
            if len(action) != self.action_dim:
                action = np.pad(action, (0, max(0, self.action_dim - len(action))),
                              mode='constant')[:self.action_dim]

            # Apply action to state (first action_dim dimensions)
            self.state[:self.action_dim] += action * 0.1

            # Environmental dynamics (rest of state evolves)
            if self.obs_dim > self.action_dim:
                self.state[self.action_dim:] += np.random.randn(
                    self.obs_dim - self.action_dim) * 0.05

            # Multi-objective reward
            # Objective 1: Approach primary target
            primary_reward = -np.linalg.norm(
                self.state[:self.action_dim] - self.target_state[:self.action_dim]
            )

            # Objective 2: Maintain secondary constraint
            secondary_reward = -np.linalg.norm(
                self.state[self.action_dim:] - self.secondary_target[self.action_dim:]
            ) if self.obs_dim > self.action_dim else 0.0

            # Objective 3: Energy efficiency (penalize large actions)
            energy_penalty = -0.1 * np.linalg.norm(action)

            # Combined reward
            reward = (0.5 * primary_reward +
                     0.3 * secondary_reward +
                     0.2 * energy_penalty)

            # Adversarial perturbation
            if np.random.rand() < self.epsilon:
                self.state += np.random.randn(self.obs_dim) * 0.1 * self.difficulty

            # State normalization
            self.state = np.clip(self.state, -3.0, 3.0)

            done = False  # Continue for full episode

            return self.state.copy(), reward, done

        def compute_k_index(self, observations: np.ndarray,
                           actions: np.ndarray) -> float:
            """Compute K-Index: 2 * |correlation(obs, act)|."""
            if len(observations) < 2 or len(actions) < 2:
                return 0.0

            # Use subset of dimensions for correlation
            obs_subset = observations[:, :min(20, self.obs_dim)]
            act_subset = actions[:, :min(10, self.action_dim)]

            # Flatten
            obs_flat = obs_subset.flatten()
            act_flat = act_subset.flatten()

            # Compute correlation
            if len(obs_flat) < 2 or len(act_flat) < 2:
                return 0.0

            # Ensure same length
            min_len = min(len(obs_flat), len(act_flat))
            obs_flat = obs_flat[:min_len]
            act_flat = act_flat[:min_len]

            # Correlation
            correlation = np.corrcoef(obs_flat, act_flat)[0, 1]

            if np.isnan(correlation):
                return 0.0

            k_index = 2.0 * abs(correlation)
            return float(k_index)

    # Evolved Agent (parameters set by CMA-ES)
    class EvolvedAgent:
        """Neural network agent with externally-set parameters."""

        def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            self.action_dim = action_dim

            # Calculate total parameters
            self.n_params = (obs_dim * hidden_dim + hidden_dim +
                           hidden_dim * action_dim + action_dim)

            # Initialize parameters (will be replaced by evolution)
            self.W1 = np.zeros((hidden_dim, obs_dim))
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.zeros((action_dim, hidden_dim))
            self.b2 = np.zeros(action_dim)

        def set_parameters(self, params: np.ndarray):
            """Set network parameters from flat vector."""
            idx = 0

            # W1
            w1_size = self.obs_dim * self.hidden_dim
            self.W1 = params[idx:idx+w1_size].reshape(self.hidden_dim, self.obs_dim)
            idx += w1_size

            # b1
            self.b1 = params[idx:idx+self.hidden_dim]
            idx += self.hidden_dim

            # W2
            w2_size = self.hidden_dim * self.action_dim
            self.W2 = params[idx:idx+w2_size].reshape(self.action_dim, self.hidden_dim)
            idx += w2_size

            # b2
            self.b2 = params[idx:idx+self.action_dim]

        def forward(self, obs: np.ndarray) -> np.ndarray:
            """Forward pass through network."""
            # Hidden layer
            h = np.maximum(0, obs @ self.W1.T + self.b1)  # ReLU

            # Output layer
            action = np.tanh(h @ self.W2.T + self.b2)

            return action

    # CMA-ES Implementation
    class CMAES:
        """Covariance Matrix Adaptation Evolution Strategy."""

        def __init__(self, n_params: int, population_size: int,
                     sigma_init: float = 0.5, elite_ratio: float = 0.2):
            self.n_params = n_params
            self.population_size = population_size
            self.sigma = sigma_init
            self.elite_size = max(1, int(population_size * elite_ratio))

            # Distribution parameters
            self.mean = np.random.randn(n_params) * 0.1
            self.C = np.eye(n_params)  # Covariance matrix

            # Evolution paths
            self.pc = np.zeros(n_params)  # Cumulation path for C
            self.ps = np.zeros(n_params)  # Cumulation path for sigma

            # Strategy parameters
            self.cc = 4.0 / (n_params + 4.0)  # Cumulation for C
            self.cs = 4.0 / (n_params + 4.0)  # Cumulation for sigma
            self.c1 = 2.0 / ((n_params + 1.3)**2 + population_size)  # Learning rate for rank-1 update
            self.cmu = min(1 - self.c1, 2 * (self.elite_size - 2 + 1/self.elite_size) /
                          ((n_params + 2)**2 + self.elite_size))  # Learning rate for rank-μ update
            self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.elite_size - 1) / (n_params + 1)) - 1) + self.cs

            # Expectation of ||N(0,I)||
            self.chiN = np.sqrt(n_params) * (1 - 1/(4*n_params) + 1/(21*n_params**2))

        def ask(self) -> List[np.ndarray]:
            """Generate population of candidate solutions."""
            population = []

            # Sample from multivariate normal
            for _ in range(self.population_size):
                z = np.random.randn(self.n_params)
                # y = C^(1/2) * z
                # Since C is covariance, we can use Cholesky decomposition
                # But for simplicity and stability, use eigendecomposition
                D, B = np.linalg.eigh(self.C)
                D = np.sqrt(np.maximum(0, D))  # Ensure non-negative
                y = B @ (D * z)

                # x = mean + sigma * y
                x = self.mean + self.sigma * y
                population.append(x)

            return population

        def tell(self, population: List[np.ndarray], fitness: List[float]):
            """Update distribution based on fitness."""
            # Sort population by fitness (higher is better for K-Index)
            indices = np.argsort(fitness)[::-1]  # Descending order
            elite_indices = indices[:self.elite_size]

            # Select elite
            elite = [population[i] for i in elite_indices]

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.mean(elite, axis=0)

            # Update evolution paths
            # Conjugate evolution path
            y_mean = (self.mean - old_mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.elite_size) * y_mean

            # Accumulated evolution path
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (len(fitness) + 1))) / self.chiN
                   < 1.4 + 2 / (self.n_params + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.elite_size) * y_mean

            # Update covariance matrix
            # Rank-1 update
            rank_one = self.c1 * (np.outer(self.pc, self.pc) - self.C)

            # Rank-μ update
            rank_mu = np.zeros((self.n_params, self.n_params))
            for i in elite_indices:
                y = (population[i] - old_mean) / self.sigma
                rank_mu += np.outer(y, y)
            rank_mu = self.cmu / self.elite_size * (rank_mu - self.C)

            self.C = self.C + rank_one + rank_mu

            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

    # Evaluation function
    def evaluate_candidate(params: np.ndarray, agent: EvolvedAgent,
                          env: EnhancedEnvironment, n_episodes: int,
                          episode_length: int) -> Tuple[float, float]:
        """Evaluate a candidate parameter set.

        Returns:
            (mean_k_index, mean_reward)
        """
        agent.set_parameters(params)

        k_indices = []
        rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            episode_obs = []
            episode_acts = []
            episode_rewards = []

            for _ in range(episode_length):
                action = agent.forward(obs)
                episode_obs.append(obs)
                episode_acts.append(action)

                obs, reward, done = env.step(action)
                episode_rewards.append(reward)

                if done:
                    break

            # Compute K-Index for this episode
            k_idx = env.compute_k_index(
                np.array(episode_obs),
                np.array(episode_acts)
            )
            k_indices.append(k_idx)
            rewards.append(np.mean(episode_rewards))

        return np.mean(k_indices), np.mean(rewards)

    # Initialize environment and agent
    env = EnhancedEnvironment(
        obs_dim=env_cfg.get('observations', 100),
        action_dim=env_cfg.get('actions', 50),
        epsilon=env_cfg.get('adversarial_strength', 0.05),
        difficulty=env_cfg.get('difficulty', 1.05)
    )

    agent = EvolvedAgent(
        obs_dim=env_cfg.get('observations', 100),
        hidden_dim=arch_cfg.get('neurons', [100, 50])[0],
        action_dim=env_cfg.get('actions', 50)
    )

    print(f"\n🚀 Starting CMA-ES evolution...")
    print(f"Parameter space: {agent.n_params} dimensions")
    print(f"Population size: {population_size}")
    print(f"Elite size: {int(population_size * elite_ratio)}")
    print(f"\n")

    # Initialize CMA-ES
    cmaes = CMAES(
        n_params=agent.n_params,
        population_size=population_size,
        sigma_init=sigma_init,
        elite_ratio=elite_ratio
    )

    # Evolution loop
    all_k_indices = []
    all_results = []
    best_k = 0.0
    best_generation = 0
    generations_without_improvement = 0

    for generation in range(max_generations):
        # Generate population
        population = cmaes.ask()

        # Evaluate population
        fitness = []
        for params in population:
            k_idx, mean_reward = evaluate_candidate(
                params, agent, env,
                episodes_per_candidate, episode_length
            )
            fitness.append(k_idx)  # Use K-Index as fitness

        # Update CMA-ES
        cmaes.tell(population, fitness)

        # Track best
        current_best_k = max(fitness)
        current_mean_k = np.mean(fitness)
        current_std_k = np.std(fitness)

        all_k_indices.append(current_best_k)
        all_results.append({
            'generation': generation + 1,
            'best_k_index': float(current_best_k),
            'mean_k_index': float(current_mean_k),
            'std_k_index': float(current_std_k),
            'sigma': float(cmaes.sigma)
        })

        # Update best
        if current_best_k > best_k + min_delta:
            best_k = current_best_k
            best_generation = generation + 1
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Progress report
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"\n📊 Generation {generation + 1}/{max_generations}:")
            print(f"  Best K: {current_best_k:.4f}")
            print(f"  Mean K: {current_mean_k:.4f} ± {current_std_k:.4f}")
            print(f"  Max K: {best_k:.4f} (gen {best_generation})")
            print(f"  Progress: {(best_k / 1.5) * 100:.1f}% to threshold")
            print(f"  Sigma: {cmaes.sigma:.4f}")

            if current_best_k > 1.1208:  # G2 baseline
                print(f"  ⚡ BETTER THAN G2 BASELINE!")
            if current_best_k > 1.1839:  # G7 best
                print(f"  🎯 BETTER THAN G7 (TASK REDESIGN)!")
            if current_best_k > 1.3:  # Prediction
                print(f"  🌟 PREDICTION ACHIEVED! K > 1.3")

        # Early stopping
        if early_stop_enabled and generations_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} generations")
            print(f"   Best K: {best_k:.4f} at Generation {best_generation}")
            break

    # Final summary
    print("\n" + "="*70)
    print("📊 PHASE G9 COMPLETE")
    print("="*70)
    print(f"Best K-Index: {best_k:.4f} (Generation {best_generation})")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")

    # Compare to baselines
    g2_baseline = 1.1208
    g7_best = 1.1839

    if best_k > g7_best:
        improvement = ((best_k - g7_best) / g7_best) * 100
        print(f"\n✅ IMPROVEMENT vs G7: +{improvement:.1f}% ({best_k:.4f} vs {g7_best:.4f})")
        print(f"   🎯 EVOLUTIONARY ALGORITHM BREAKS G7 CEILING!")
    elif best_k > g2_baseline:
        improvement = ((best_k - g2_baseline) / g2_baseline) * 100
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")
        print(f"   ⚠️  Below G7 performance ({g7_best:.4f})")
    else:
        decline = ((g2_baseline - best_k) / g2_baseline) * 100
        print(f"\n❌ DECLINE vs G2: -{decline:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")

    # Save results
    output_dir = Path(config.get('experiment', {}).get('output_dir', 'logs/track_g'))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"track_g_phase_g9_{timestamp}.json"

    results = {
        'phase': 'G8',
        'algorithm': 'CMA-ES',
        'population_size': population_size,
        'generations_completed': len(all_k_indices),
        'max_generations': max_generations,
        'environment': {
            'observations': env_cfg.get('observations', 100),
            'actions': env_cfg.get('actions', 50),
            'objectives': 'multi',
            'episode_length': episode_length,
            'adversarial_strength': env_cfg.get('adversarial_strength', 0.05),
            'difficulty': env_cfg.get('difficulty', 1.05)
        },
        'architecture': {
            'type': 'feedforward',
            'layers': 2,
            'neurons': arch_cfg.get('neurons', [100, 50]),
            'activation': 'relu',
            'total_parameters': agent.n_params
        },
        'all_k_indices': all_k_indices,
        'all_results': all_results,
        'max_k_index': float(best_k),
        'best_generation': best_generation,
        'mean_k_index': float(np.mean(all_k_indices)),
        'std_k_index': float(np.std(all_k_indices)),
        'final_k_index': float(all_k_indices[-1])
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved: {output_file}")

    print("\n" + "="*70)
    print("🏆 PHASE G9 COMPLETE")
    print("="*70)
    print(f"Total generations: {len(all_k_indices)}")
    print(f"Best K-Index: {best_k:.4f}")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")
def run_phase_g10(config: dict) -> Dict:
    """Run Phase G10: Hybrid Breakthrough (G8 Algorithm + G7 Environment).

    After G9 proved population size NOT the bottleneck (-2.5% vs G8), G10 tests
    synergistic combination of TWO independently proven improvements:
    - G8's winning CMA-ES config (pop=20, eps=3): +26.7% improvement
    - G7's enhanced environment (100×50×1000): +5.6% improvement

    Hypothesis: Combining proven approaches yields K≥1.5 threshold crossing.

    Args:
        config: Configuration dictionary containing:
            - algorithm: CMA-ES parameters (population, generations, sigma)
            - environment: Task specification (from G7 enhanced environment)
            - architecture: Network structure (G2's proven 2-layer)
            - evaluation: Episodes per candidate
            - early_stopping: Patience and min_delta

    Returns:
        Dictionary containing:
            - phase: "G10"
            - max_k_index: Best K-Index achieved
            - best_generation: Generation with best K
            - mean_k_index: Average across all generations
            - std_k_index: Standard deviation
            - all_k_indices: K-Index for each generation
            - all_results: Detailed per-generation results
            - configuration: Full experimental configuration
    """

    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G10: EVOLUTIONARY ALGORITHM (CMA-ES)")
    print("="*70)

    # Extract configuration
    algo_cfg = config.get('phase_g10', {}).get('algorithm', {})
    env_cfg = config.get('phase_g10', {}).get('environment', {})
    arch_cfg = config.get('phase_g10', {}).get('architecture', {})
    eval_cfg = config.get('phase_g10', {}).get('evaluation', {})
    early_stop_cfg = config.get('phase_g10', {}).get('early_stopping', {})

    population_size = algo_cfg.get('population_size', 50)
    max_generations = algo_cfg.get('generations', 100)
    sigma_init = algo_cfg.get('sigma_init', 0.5)
    elite_ratio = algo_cfg.get('elite_ratio', 0.2)

    episodes_per_candidate = eval_cfg.get('episodes_per_candidate', 5)
    episode_length = eval_cfg.get('episode_length', 1000)

    early_stop_enabled = early_stop_cfg.get('enabled', True)
    patience = early_stop_cfg.get('patience', 20)
    min_delta = early_stop_cfg.get('min_delta', 0.01)

    print(f"\n📊 G8 Configuration:")
    print(f"Algorithm: CMA-ES")
    print(f"Population: {population_size}")
    print(f"Generations: {max_generations}")
    print(f"Initial sigma: {sigma_init}")
    print(f"Elite ratio: {elite_ratio} (top {int(population_size * elite_ratio)})")
    print(f"Episodes per candidate: {episodes_per_candidate}")
    print(f"Environment: {env_cfg.get('observations')} obs × {env_cfg.get('actions')} act × {episode_length} steps")
    print(f"Hypothesis: Non-gradient optimization escapes K~1.2 ceiling")
    print(f"Prediction: K > 1.3 (87% to threshold)")

    # Enhanced Environment (from G7)
    class EnhancedEnvironment:
        """Enhanced environment with higher complexity and multiple objectives."""

        def __init__(self, obs_dim: int = 100, action_dim: int = 50,
                     epsilon: float = 0.05, difficulty: float = 1.05):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.epsilon = epsilon
            self.difficulty = difficulty

            # Multi-objective state
            self.state = None
            self.target_state = None
            self.secondary_target = None

        def reset(self) -> np.ndarray:
            """Reset environment to initial state."""
            self.state = np.random.randn(self.obs_dim) * 0.1
            self.target_state = np.random.randn(self.obs_dim) * 2.0
            self.secondary_target = np.random.randn(self.obs_dim) * 1.0
            return self.state.copy()

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
            """Execute action and return next_state, reward, done."""
            # Ensure action is correct dimension
            if len(action) != self.action_dim:
                action = np.pad(action, (0, max(0, self.action_dim - len(action))),
                              mode='constant')[:self.action_dim]

            # Apply action to state (first action_dim dimensions)
            self.state[:self.action_dim] += action * 0.1

            # Environmental dynamics (rest of state evolves)
            if self.obs_dim > self.action_dim:
                self.state[self.action_dim:] += np.random.randn(
                    self.obs_dim - self.action_dim) * 0.05

            # Multi-objective reward
            # Objective 1: Approach primary target
            primary_reward = -np.linalg.norm(
                self.state[:self.action_dim] - self.target_state[:self.action_dim]
            )

            # Objective 2: Maintain secondary constraint
            secondary_reward = -np.linalg.norm(
                self.state[self.action_dim:] - self.secondary_target[self.action_dim:]
            ) if self.obs_dim > self.action_dim else 0.0

            # Objective 3: Energy efficiency (penalize large actions)
            energy_penalty = -0.1 * np.linalg.norm(action)

            # Combined reward
            reward = (0.5 * primary_reward +
                     0.3 * secondary_reward +
                     0.2 * energy_penalty)

            # Adversarial perturbation
            if np.random.rand() < self.epsilon:
                self.state += np.random.randn(self.obs_dim) * 0.1 * self.difficulty

            # State normalization
            self.state = np.clip(self.state, -3.0, 3.0)

            done = False  # Continue for full episode

            return self.state.copy(), reward, done

        def compute_k_index(self, observations: np.ndarray,
                           actions: np.ndarray) -> float:
            """Compute K-Index: 2 * |correlation(obs, act)|."""
            if len(observations) < 2 or len(actions) < 2:
                return 0.0

            # Use subset of dimensions for correlation
            obs_subset = observations[:, :min(20, self.obs_dim)]
            act_subset = actions[:, :min(10, self.action_dim)]

            # Flatten
            obs_flat = obs_subset.flatten()
            act_flat = act_subset.flatten()

            # Compute correlation
            if len(obs_flat) < 2 or len(act_flat) < 2:
                return 0.0

            # Ensure same length
            min_len = min(len(obs_flat), len(act_flat))
            obs_flat = obs_flat[:min_len]
            act_flat = act_flat[:min_len]

            # Correlation
            correlation = np.corrcoef(obs_flat, act_flat)[0, 1]

            if np.isnan(correlation):
                return 0.0

            k_index = 2.0 * abs(correlation)
            return float(k_index)

    # Evolved Agent (parameters set by CMA-ES)
    class EvolvedAgent:
        """Neural network agent with externally-set parameters."""

        def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            self.action_dim = action_dim

            # Calculate total parameters
            self.n_params = (obs_dim * hidden_dim + hidden_dim +
                           hidden_dim * action_dim + action_dim)

            # Initialize parameters (will be replaced by evolution)
            self.W1 = np.zeros((hidden_dim, obs_dim))
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.zeros((action_dim, hidden_dim))
            self.b2 = np.zeros(action_dim)

        def set_parameters(self, params: np.ndarray):
            """Set network parameters from flat vector."""
            idx = 0

            # W1
            w1_size = self.obs_dim * self.hidden_dim
            self.W1 = params[idx:idx+w1_size].reshape(self.hidden_dim, self.obs_dim)
            idx += w1_size

            # b1
            self.b1 = params[idx:idx+self.hidden_dim]
            idx += self.hidden_dim

            # W2
            w2_size = self.hidden_dim * self.action_dim
            self.W2 = params[idx:idx+w2_size].reshape(self.action_dim, self.hidden_dim)
            idx += w2_size

            # b2
            self.b2 = params[idx:idx+self.action_dim]

        def forward(self, obs: np.ndarray) -> np.ndarray:
            """Forward pass through network."""
            # Hidden layer
            h = np.maximum(0, obs @ self.W1.T + self.b1)  # ReLU

            # Output layer
            action = np.tanh(h @ self.W2.T + self.b2)

            return action

    # CMA-ES Implementation
    class CMAES:
        """Covariance Matrix Adaptation Evolution Strategy."""

        def __init__(self, n_params: int, population_size: int,
                     sigma_init: float = 0.5, elite_ratio: float = 0.2):
            self.n_params = n_params
            self.population_size = population_size
            self.sigma = sigma_init
            self.elite_size = max(1, int(population_size * elite_ratio))

            # Distribution parameters
            self.mean = np.random.randn(n_params) * 0.1
            self.C = np.eye(n_params)  # Covariance matrix

            # Evolution paths
            self.pc = np.zeros(n_params)  # Cumulation path for C
            self.ps = np.zeros(n_params)  # Cumulation path for sigma

            # Strategy parameters
            self.cc = 4.0 / (n_params + 4.0)  # Cumulation for C
            self.cs = 4.0 / (n_params + 4.0)  # Cumulation for sigma
            self.c1 = 2.0 / ((n_params + 1.3)**2 + population_size)  # Learning rate for rank-1 update
            self.cmu = min(1 - self.c1, 2 * (self.elite_size - 2 + 1/self.elite_size) /
                          ((n_params + 2)**2 + self.elite_size))  # Learning rate for rank-μ update
            self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.elite_size - 1) / (n_params + 1)) - 1) + self.cs

            # Expectation of ||N(0,I)||
            self.chiN = np.sqrt(n_params) * (1 - 1/(4*n_params) + 1/(21*n_params**2))

        def ask(self) -> List[np.ndarray]:
            """Generate population of candidate solutions."""
            population = []

            # Sample from multivariate normal
            for _ in range(self.population_size):
                z = np.random.randn(self.n_params)
                # y = C^(1/2) * z
                # Since C is covariance, we can use Cholesky decomposition
                # But for simplicity and stability, use eigendecomposition
                D, B = np.linalg.eigh(self.C)
                D = np.sqrt(np.maximum(0, D))  # Ensure non-negative
                y = B @ (D * z)

                # x = mean + sigma * y
                x = self.mean + self.sigma * y
                population.append(x)

            return population

        def tell(self, population: List[np.ndarray], fitness: List[float]):
            """Update distribution based on fitness."""
            # Sort population by fitness (higher is better for K-Index)
            indices = np.argsort(fitness)[::-1]  # Descending order
            elite_indices = indices[:self.elite_size]

            # Select elite
            elite = [population[i] for i in elite_indices]

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.mean(elite, axis=0)

            # Update evolution paths
            # Conjugate evolution path
            y_mean = (self.mean - old_mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.elite_size) * y_mean

            # Accumulated evolution path
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (len(fitness) + 1))) / self.chiN
                   < 1.4 + 2 / (self.n_params + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.elite_size) * y_mean

            # Update covariance matrix
            # Rank-1 update
            rank_one = self.c1 * (np.outer(self.pc, self.pc) - self.C)

            # Rank-μ update
            rank_mu = np.zeros((self.n_params, self.n_params))
            for i in elite_indices:
                y = (population[i] - old_mean) / self.sigma
                rank_mu += np.outer(y, y)
            rank_mu = self.cmu / self.elite_size * (rank_mu - self.C)

            self.C = self.C + rank_one + rank_mu

            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

    # Evaluation function
    def evaluate_candidate(params: np.ndarray, agent: EvolvedAgent,
                          env: EnhancedEnvironment, n_episodes: int,
                          episode_length: int) -> Tuple[float, float]:
        """Evaluate a candidate parameter set.

        Returns:
            (mean_k_index, mean_reward)
        """
        agent.set_parameters(params)

        k_indices = []
        rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            episode_obs = []
            episode_acts = []
            episode_rewards = []

            for _ in range(episode_length):
                action = agent.forward(obs)
                episode_obs.append(obs)
                episode_acts.append(action)

                obs, reward, done = env.step(action)
                episode_rewards.append(reward)

                if done:
                    break

            # Compute K-Index for this episode
            k_idx = env.compute_k_index(
                np.array(episode_obs),
                np.array(episode_acts)
            )
            k_indices.append(k_idx)
            rewards.append(np.mean(episode_rewards))

        return np.mean(k_indices), np.mean(rewards)

    # Initialize environment and agent
    env = EnhancedEnvironment(
        obs_dim=env_cfg.get('observations', 100),
        action_dim=env_cfg.get('actions', 50),
        epsilon=env_cfg.get('adversarial_strength', 0.05),
        difficulty=env_cfg.get('difficulty', 1.05)
    )

    agent = EvolvedAgent(
        obs_dim=env_cfg.get('observations', 100),
        hidden_dim=arch_cfg.get('neurons', [100, 50])[0],
        action_dim=env_cfg.get('actions', 50)
    )

    print(f"\n🚀 Starting CMA-ES evolution...")
    print(f"Parameter space: {agent.n_params} dimensions")
    print(f"Population size: {population_size}")
    print(f"Elite size: {int(population_size * elite_ratio)}")
    print(f"\n")

    # Initialize CMA-ES
    cmaes = CMAES(
        n_params=agent.n_params,
        population_size=population_size,
        sigma_init=sigma_init,
        elite_ratio=elite_ratio
    )

    # Evolution loop
    all_k_indices = []
    all_results = []
    best_k = 0.0
    best_generation = 0
    generations_without_improvement = 0

    for generation in range(max_generations):
        # Generate population
        population = cmaes.ask()

        # Evaluate population
        fitness = []
        for params in population:
            k_idx, mean_reward = evaluate_candidate(
                params, agent, env,
                episodes_per_candidate, episode_length
            )
            fitness.append(k_idx)  # Use K-Index as fitness

        # Update CMA-ES
        cmaes.tell(population, fitness)

        # Track best
        current_best_k = max(fitness)
        current_mean_k = np.mean(fitness)
        current_std_k = np.std(fitness)

        all_k_indices.append(current_best_k)
        all_results.append({
            'generation': generation + 1,
            'best_k_index': float(current_best_k),
            'mean_k_index': float(current_mean_k),
            'std_k_index': float(current_std_k),
            'sigma': float(cmaes.sigma)
        })

        # Update best
        if current_best_k > best_k + min_delta:
            best_k = current_best_k
            best_generation = generation + 1
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Progress report
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"\n📊 Generation {generation + 1}/{max_generations}:")
            print(f"  Best K: {current_best_k:.4f}")
            print(f"  Mean K: {current_mean_k:.4f} ± {current_std_k:.4f}")
            print(f"  Max K: {best_k:.4f} (gen {best_generation})")
            print(f"  Progress: {(best_k / 1.5) * 100:.1f}% to threshold")
            print(f"  Sigma: {cmaes.sigma:.4f}")

            if current_best_k > 1.1208:  # G2 baseline
                print(f"  ⚡ BETTER THAN G2 BASELINE!")
            if current_best_k > 1.1839:  # G7 best
                print(f"  🎯 BETTER THAN G7 (TASK REDESIGN)!")
            if current_best_k > 1.3:  # Prediction
                print(f"  🌟 PREDICTION ACHIEVED! K > 1.3")

        # Early stopping
        if early_stop_enabled and generations_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} generations")
            print(f"   Best K: {best_k:.4f} at Generation {best_generation}")
            break

    # Final summary
    print("\n" + "="*70)
    print("📊 PHASE G10 COMPLETE")
    print("="*70)
    print(f"Best K-Index: {best_k:.4f} (Generation {best_generation})")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")

    # Compare to baselines
    g2_baseline = 1.1208
    g7_best = 1.1839

    if best_k > g7_best:
        improvement = ((best_k - g7_best) / g7_best) * 100
        print(f"\n✅ IMPROVEMENT vs G7: +{improvement:.1f}% ({best_k:.4f} vs {g7_best:.4f})")
        print(f"   🎯 EVOLUTIONARY ALGORITHM BREAKS G7 CEILING!")
    elif best_k > g2_baseline:
        improvement = ((best_k - g2_baseline) / g2_baseline) * 100
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")
        print(f"   ⚠️  Below G7 performance ({g7_best:.4f})")
    else:
        decline = ((g2_baseline - best_k) / g2_baseline) * 100
        print(f"\n❌ DECLINE vs G2: -{decline:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")

    # Save results
    output_dir = Path(config.get('experiment', {}).get('output_dir', 'logs/track_g'))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"track_g_phase_g10_{timestamp}.json"

    results = {
        'phase': 'G8',
        'algorithm': 'CMA-ES',
        'population_size': population_size,
        'generations_completed': len(all_k_indices),
        'max_generations': max_generations,
        'environment': {
            'observations': env_cfg.get('observations', 100),
            'actions': env_cfg.get('actions', 50),
            'objectives': 'multi',
            'episode_length': episode_length,
            'adversarial_strength': env_cfg.get('adversarial_strength', 0.05),
            'difficulty': env_cfg.get('difficulty', 1.05)
        },
        'architecture': {
            'type': 'feedforward',
            'layers': 2,
            'neurons': arch_cfg.get('neurons', [100, 50]),
            'activation': 'relu',
            'total_parameters': agent.n_params
        },
        'all_k_indices': all_k_indices,
        'all_results': all_results,
        'max_k_index': float(best_k),
        'best_generation': best_generation,
        'mean_k_index': float(np.mean(all_k_indices)),
        'std_k_index': float(np.std(all_k_indices)),
        'final_k_index': float(all_k_indices[-1])
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved: {output_file}")

    print("\n" + "="*70)
    print("🏆 PHASE G10 COMPLETE")
    print("="*70)
    print(f"Total generations: {len(all_k_indices)}")
    print(f"Best K-Index: {best_k:.4f}")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")


def run_phase_g10_lite(config: dict) -> Dict:
    """Run Phase G10-Lite: Feasible Hybrid (G8 Algorithm + Moderate Enhancement).

    After G9 proved population size NOT the bottleneck (-2.5% vs G8), G10 tests
    synergistic combination of TWO independently proven improvements:
    - G8's winning CMA-ES config (pop=20, eps=3): +26.7% improvement
    - G7's enhanced environment (100×50×1000): +5.6% improvement

    Hypothesis: Combining proven approaches yields K≥1.5 threshold crossing.

    Args:
        config: Configuration dictionary containing:
            - algorithm: CMA-ES parameters (population, generations, sigma)
            - environment: Task specification (from G7 enhanced environment)
            - architecture: Network structure (G2's proven 2-layer)
            - evaluation: Episodes per candidate
            - early_stopping: Patience and min_delta

    Returns:
        Dictionary containing:
            - phase: "G10"
            - max_k_index: Best K-Index achieved
            - best_generation: Generation with best K
            - mean_k_index: Average across all generations
            - std_k_index: Standard deviation
            - all_k_indices: K-Index for each generation
            - all_results: Detailed per-generation results
            - configuration: Full experimental configuration
    """

    print("\n" + "="*70)
    print("🎯 TRACK G - PHASE G10: EVOLUTIONARY ALGORITHM (CMA-ES)")
    print("="*70)

    # Extract configuration
    algo_cfg = config.get('phase_g10_lite', {}).get('algorithm', {})
    env_cfg = config.get('phase_g10_lite', {}).get('environment', {})
    arch_cfg = config.get('phase_g10_lite', {}).get('architecture', {})
    eval_cfg = config.get('phase_g10_lite', {}).get('evaluation', {})
    early_stop_cfg = config.get('phase_g10_lite', {}).get('early_stopping', {})

    population_size = algo_cfg.get('population_size', 50)
    max_generations = algo_cfg.get('generations', 100)
    sigma_init = algo_cfg.get('sigma_init', 0.5)
    elite_ratio = algo_cfg.get('elite_ratio', 0.2)

    episodes_per_candidate = eval_cfg.get('episodes_per_candidate', 5)
    episode_length = eval_cfg.get('episode_length', 1000)

    early_stop_enabled = early_stop_cfg.get('enabled', True)
    patience = early_stop_cfg.get('patience', 20)
    min_delta = early_stop_cfg.get('min_delta', 0.01)

    print(f"\n📊 G8 Configuration:")
    print(f"Algorithm: CMA-ES")
    print(f"Population: {population_size}")
    print(f"Generations: {max_generations}")
    print(f"Initial sigma: {sigma_init}")
    print(f"Elite ratio: {elite_ratio} (top {int(population_size * elite_ratio)})")
    print(f"Episodes per candidate: {episodes_per_candidate}")
    print(f"Environment: {env_cfg.get('observations')} obs × {env_cfg.get('actions')} act × {episode_length} steps")
    print(f"Hypothesis: Non-gradient optimization escapes K~1.2 ceiling")
    print(f"Prediction: K > 1.3 (87% to threshold)")

    # Enhanced Environment (from G7)
    class EnhancedEnvironment:
        """Enhanced environment with higher complexity and multiple objectives."""

        def __init__(self, obs_dim: int = 100, action_dim: int = 50,
                     epsilon: float = 0.05, difficulty: float = 1.05):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.epsilon = epsilon
            self.difficulty = difficulty

            # Multi-objective state
            self.state = None
            self.target_state = None
            self.secondary_target = None

        def reset(self) -> np.ndarray:
            """Reset environment to initial state."""
            self.state = np.random.randn(self.obs_dim) * 0.1
            self.target_state = np.random.randn(self.obs_dim) * 2.0
            self.secondary_target = np.random.randn(self.obs_dim) * 1.0
            return self.state.copy()

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
            """Execute action and return next_state, reward, done."""
            # Ensure action is correct dimension
            if len(action) != self.action_dim:
                action = np.pad(action, (0, max(0, self.action_dim - len(action))),
                              mode='constant')[:self.action_dim]

            # Apply action to state (first action_dim dimensions)
            self.state[:self.action_dim] += action * 0.1

            # Environmental dynamics (rest of state evolves)
            if self.obs_dim > self.action_dim:
                self.state[self.action_dim:] += np.random.randn(
                    self.obs_dim - self.action_dim) * 0.05

            # Multi-objective reward
            # Objective 1: Approach primary target
            primary_reward = -np.linalg.norm(
                self.state[:self.action_dim] - self.target_state[:self.action_dim]
            )

            # Objective 2: Maintain secondary constraint
            secondary_reward = -np.linalg.norm(
                self.state[self.action_dim:] - self.secondary_target[self.action_dim:]
            ) if self.obs_dim > self.action_dim else 0.0

            # Objective 3: Energy efficiency (penalize large actions)
            energy_penalty = -0.1 * np.linalg.norm(action)

            # Combined reward
            reward = (0.5 * primary_reward +
                     0.3 * secondary_reward +
                     0.2 * energy_penalty)

            # Adversarial perturbation
            if np.random.rand() < self.epsilon:
                self.state += np.random.randn(self.obs_dim) * 0.1 * self.difficulty

            # State normalization
            self.state = np.clip(self.state, -3.0, 3.0)

            done = False  # Continue for full episode

            return self.state.copy(), reward, done

        def compute_k_index(self, observations: np.ndarray,
                           actions: np.ndarray) -> float:
            """Compute K-Index: 2 * |correlation(obs, act)|."""
            if len(observations) < 2 or len(actions) < 2:
                return 0.0

            # Use subset of dimensions for correlation
            obs_subset = observations[:, :min(20, self.obs_dim)]
            act_subset = actions[:, :min(10, self.action_dim)]

            # Flatten
            obs_flat = obs_subset.flatten()
            act_flat = act_subset.flatten()

            # Compute correlation
            if len(obs_flat) < 2 or len(act_flat) < 2:
                return 0.0

            # Ensure same length
            min_len = min(len(obs_flat), len(act_flat))
            obs_flat = obs_flat[:min_len]
            act_flat = act_flat[:min_len]

            # Correlation
            correlation = np.corrcoef(obs_flat, act_flat)[0, 1]

            if np.isnan(correlation):
                return 0.0

            k_index = 2.0 * abs(correlation)
            return float(k_index)

    # Evolved Agent (parameters set by CMA-ES)
    class EvolvedAgent:
        """Neural network agent with externally-set parameters."""

        def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            self.action_dim = action_dim

            # Calculate total parameters
            self.n_params = (obs_dim * hidden_dim + hidden_dim +
                           hidden_dim * action_dim + action_dim)

            # Initialize parameters (will be replaced by evolution)
            self.W1 = np.zeros((hidden_dim, obs_dim))
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.zeros((action_dim, hidden_dim))
            self.b2 = np.zeros(action_dim)

        def set_parameters(self, params: np.ndarray):
            """Set network parameters from flat vector."""
            idx = 0

            # W1
            w1_size = self.obs_dim * self.hidden_dim
            self.W1 = params[idx:idx+w1_size].reshape(self.hidden_dim, self.obs_dim)
            idx += w1_size

            # b1
            self.b1 = params[idx:idx+self.hidden_dim]
            idx += self.hidden_dim

            # W2
            w2_size = self.hidden_dim * self.action_dim
            self.W2 = params[idx:idx+w2_size].reshape(self.action_dim, self.hidden_dim)
            idx += w2_size

            # b2
            self.b2 = params[idx:idx+self.action_dim]

        def forward(self, obs: np.ndarray) -> np.ndarray:
            """Forward pass through network."""
            # Hidden layer
            h = np.maximum(0, obs @ self.W1.T + self.b1)  # ReLU

            # Output layer
            action = np.tanh(h @ self.W2.T + self.b2)

            return action

    # CMA-ES Implementation
    class CMAES:
        """Covariance Matrix Adaptation Evolution Strategy."""

        def __init__(self, n_params: int, population_size: int,
                     sigma_init: float = 0.5, elite_ratio: float = 0.2):
            self.n_params = n_params
            self.population_size = population_size
            self.sigma = sigma_init
            self.elite_size = max(1, int(population_size * elite_ratio))

            # Distribution parameters
            self.mean = np.random.randn(n_params) * 0.1
            self.C = np.eye(n_params)  # Covariance matrix

            # Evolution paths
            self.pc = np.zeros(n_params)  # Cumulation path for C
            self.ps = np.zeros(n_params)  # Cumulation path for sigma

            # Strategy parameters
            self.cc = 4.0 / (n_params + 4.0)  # Cumulation for C
            self.cs = 4.0 / (n_params + 4.0)  # Cumulation for sigma
            self.c1 = 2.0 / ((n_params + 1.3)**2 + population_size)  # Learning rate for rank-1 update
            self.cmu = min(1 - self.c1, 2 * (self.elite_size - 2 + 1/self.elite_size) /
                          ((n_params + 2)**2 + self.elite_size))  # Learning rate for rank-μ update
            self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.elite_size - 1) / (n_params + 1)) - 1) + self.cs

            # Expectation of ||N(0,I)||
            self.chiN = np.sqrt(n_params) * (1 - 1/(4*n_params) + 1/(21*n_params**2))

        def ask(self) -> List[np.ndarray]:
            """Generate population of candidate solutions."""
            population = []

            # Sample from multivariate normal
            for _ in range(self.population_size):
                z = np.random.randn(self.n_params)
                # y = C^(1/2) * z
                # Since C is covariance, we can use Cholesky decomposition
                # But for simplicity and stability, use eigendecomposition
                D, B = np.linalg.eigh(self.C)
                D = np.sqrt(np.maximum(0, D))  # Ensure non-negative
                y = B @ (D * z)

                # x = mean + sigma * y
                x = self.mean + self.sigma * y
                population.append(x)

            return population

        def tell(self, population: List[np.ndarray], fitness: List[float]):
            """Update distribution based on fitness."""
            # Sort population by fitness (higher is better for K-Index)
            indices = np.argsort(fitness)[::-1]  # Descending order
            elite_indices = indices[:self.elite_size]

            # Select elite
            elite = [population[i] for i in elite_indices]

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.mean(elite, axis=0)

            # Update evolution paths
            # Conjugate evolution path
            y_mean = (self.mean - old_mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.elite_size) * y_mean

            # Accumulated evolution path
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (len(fitness) + 1))) / self.chiN
                   < 1.4 + 2 / (self.n_params + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.elite_size) * y_mean

            # Update covariance matrix
            # Rank-1 update
            rank_one = self.c1 * (np.outer(self.pc, self.pc) - self.C)

            # Rank-μ update
            rank_mu = np.zeros((self.n_params, self.n_params))
            for i in elite_indices:
                y = (population[i] - old_mean) / self.sigma
                rank_mu += np.outer(y, y)
            rank_mu = self.cmu / self.elite_size * (rank_mu - self.C)

            self.C = self.C + rank_one + rank_mu

            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

    # Evaluation function
    def evaluate_candidate(params: np.ndarray, agent: EvolvedAgent,
                          env: EnhancedEnvironment, n_episodes: int,
                          episode_length: int) -> Tuple[float, float]:
        """Evaluate a candidate parameter set.

        Returns:
            (mean_k_index, mean_reward)
        """
        agent.set_parameters(params)

        k_indices = []
        rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            episode_obs = []
            episode_acts = []
            episode_rewards = []

            for _ in range(episode_length):
                action = agent.forward(obs)
                episode_obs.append(obs)
                episode_acts.append(action)

                obs, reward, done = env.step(action)
                episode_rewards.append(reward)

                if done:
                    break

            # Compute K-Index for this episode
            k_idx = env.compute_k_index(
                np.array(episode_obs),
                np.array(episode_acts)
            )
            k_indices.append(k_idx)
            rewards.append(np.mean(episode_rewards))

        return np.mean(k_indices), np.mean(rewards)

    # Initialize environment and agent
    env = EnhancedEnvironment(
        obs_dim=env_cfg.get('observations', 100),
        action_dim=env_cfg.get('actions', 50),
        epsilon=env_cfg.get('adversarial_strength', 0.05),
        difficulty=env_cfg.get('difficulty', 1.05)
    )

    agent = EvolvedAgent(
        obs_dim=env_cfg.get('observations', 100),
        hidden_dim=arch_cfg.get('neurons', [100, 50])[0],
        action_dim=env_cfg.get('actions', 50)
    )

    print(f"\n🚀 Starting CMA-ES evolution...")
    print(f"Parameter space: {agent.n_params} dimensions")
    print(f"Population size: {population_size}")
    print(f"Elite size: {int(population_size * elite_ratio)}")
    print(f"\n")

    # Initialize CMA-ES
    cmaes = CMAES(
        n_params=agent.n_params,
        population_size=population_size,
        sigma_init=sigma_init,
        elite_ratio=elite_ratio
    )

    # Evolution loop
    all_k_indices = []
    all_results = []
    best_k = 0.0
    best_generation = 0
    generations_without_improvement = 0

    for generation in range(max_generations):
        # Generate population
        population = cmaes.ask()

        # Evaluate population
        fitness = []
        for params in population:
            k_idx, mean_reward = evaluate_candidate(
                params, agent, env,
                episodes_per_candidate, episode_length
            )
            fitness.append(k_idx)  # Use K-Index as fitness

        # Update CMA-ES
        cmaes.tell(population, fitness)

        # Track best
        current_best_k = max(fitness)
        current_mean_k = np.mean(fitness)
        current_std_k = np.std(fitness)

        all_k_indices.append(current_best_k)
        all_results.append({
            'generation': generation + 1,
            'best_k_index': float(current_best_k),
            'mean_k_index': float(current_mean_k),
            'std_k_index': float(current_std_k),
            'sigma': float(cmaes.sigma)
        })

        # Update best
        if current_best_k > best_k + min_delta:
            best_k = current_best_k
            best_generation = generation + 1
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Progress report
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"\n📊 Generation {generation + 1}/{max_generations}:")
            print(f"  Best K: {current_best_k:.4f}")
            print(f"  Mean K: {current_mean_k:.4f} ± {current_std_k:.4f}")
            print(f"  Max K: {best_k:.4f} (gen {best_generation})")
            print(f"  Progress: {(best_k / 1.5) * 100:.1f}% to threshold")
            print(f"  Sigma: {cmaes.sigma:.4f}")

            if current_best_k > 1.1208:  # G2 baseline
                print(f"  ⚡ BETTER THAN G2 BASELINE!")
            if current_best_k > 1.1839:  # G7 best
                print(f"  🎯 BETTER THAN G7 (TASK REDESIGN)!")
            if current_best_k > 1.3:  # Prediction
                print(f"  🌟 PREDICTION ACHIEVED! K > 1.3")

        # Early stopping
        if early_stop_enabled and generations_without_improvement >= patience:
            print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} generations")
            print(f"   Best K: {best_k:.4f} at Generation {best_generation}")
            break

    # Final summary
    print("\n" + "="*70)
    print("📊 PHASE G10 COMPLETE")
    print("="*70)
    print(f"Best K-Index: {best_k:.4f} (Generation {best_generation})")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")

    # Compare to baselines
    g2_baseline = 1.1208
    g7_best = 1.1839

    if best_k > g7_best:
        improvement = ((best_k - g7_best) / g7_best) * 100
        print(f"\n✅ IMPROVEMENT vs G7: +{improvement:.1f}% ({best_k:.4f} vs {g7_best:.4f})")
        print(f"   🎯 EVOLUTIONARY ALGORITHM BREAKS G7 CEILING!")
    elif best_k > g2_baseline:
        improvement = ((best_k - g2_baseline) / g2_baseline) * 100
        print(f"\n✅ IMPROVEMENT vs G2: +{improvement:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")
        print(f"   ⚠️  Below G7 performance ({g7_best:.4f})")
    else:
        decline = ((g2_baseline - best_k) / g2_baseline) * 100
        print(f"\n❌ DECLINE vs G2: -{decline:.1f}% ({best_k:.4f} vs {g2_baseline:.4f})")

    # Save results
    output_dir = Path(config.get('experiment', {}).get('output_dir', 'logs/track_g'))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"track_g_phase_g10_lite_{timestamp}.json"

    results = {
        'phase': 'G8',
        'algorithm': 'CMA-ES',
        'population_size': population_size,
        'generations_completed': len(all_k_indices),
        'max_generations': max_generations,
        'environment': {
            'observations': env_cfg.get('observations', 100),
            'actions': env_cfg.get('actions', 50),
            'objectives': 'multi',
            'episode_length': episode_length,
            'adversarial_strength': env_cfg.get('adversarial_strength', 0.05),
            'difficulty': env_cfg.get('difficulty', 1.05)
        },
        'architecture': {
            'type': 'feedforward',
            'layers': 2,
            'neurons': arch_cfg.get('neurons', [100, 50]),
            'activation': 'relu',
            'total_parameters': agent.n_params
        },
        'all_k_indices': all_k_indices,
        'all_results': all_results,
        'max_k_index': float(best_k),
        'best_generation': best_generation,
        'mean_k_index': float(np.mean(all_k_indices)),
        'std_k_index': float(np.std(all_k_indices)),
        'final_k_index': float(all_k_indices[-1])
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved: {output_file}")

    print("\n" + "="*70)
    print("🏆 PHASE G10 COMPLETE")
    print("="*70)
    print(f"Total generations: {len(all_k_indices)}")
    print(f"Best K-Index: {best_k:.4f}")
    print(f"Mean K-Index: {np.mean(all_k_indices):.4f} ± {np.std(all_k_indices):.4f}")
    print(f"Final K-Index: {all_k_indices[-1]:.4f}")




def run_phase_g2plus(config: dict, episode_logger: Optional[EpisodeLogger] = None) -> Dict:
    """Run Track G Phase 2+ with saved policy from G2."""
    phase_cfg = config['phase_g2plus']
    env_cfg = phase_cfg['environment']
    eval_cfg = phase_cfg['evaluation']
    
    # Load policy from G2
    load_path = phase_cfg['load_policy_from']
    print(f"Loading policy from: {load_path}")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Policy file not found: {load_path}")
    
    policy_data = np.load(load_path)
    W1, b1, W2, b2 = policy_data['W1'], policy_data['b1'], policy_data['W2'], policy_data['b2']
    
    # Create agent with loaded policy
    agent = EvolvedAgent(
        obs_dim=W1.shape[0],
        hidden_dim=W1.shape[1],
        action_dim=W2.shape[1]
    )
    agent.load_params(W1, b1, W2, b2)
    
    # Create enhanced environment
    env = MultiObjectiveCognitionEnv(
        obs_dim=env_cfg['observations'],
        action_dim=env_cfg['actions'],
        episode_length=env_cfg['episode_length'],
        objectives=env_cfg['objectives'],
        adversarial_strength=env_cfg.get('adversarial_strength', 0.0)
    )
    
    # Evaluate across multiple episodes
    n_episodes = eval_cfg['episodes']
    all_k = []
    
    print(f"Evaluating G2 policy on enhanced environment ({n_episodes} episodes)...")
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.forward(state)
            state, reward, done, info = env.step(action)
        
        k_idx = info['k_index']
        all_k.append(k_idx)
        print(f"Episode {ep+1}/{n_episodes}: K={k_idx:.4f}")
    
    # Results
    max_k = np.max(all_k)
    mean_k = np.mean(all_k)
    std_k = np.std(all_k)
    
    print(f"\nTrack G Phase 2+ Results:")
    print(f"   Max K-Index: {max_k:.4f}")
    print(f"   Mean K-Index: {mean_k:.4f} +/- {std_k:.4f}")
    
    return {
        'phase': 'g2plus',
        'max_k_index': max_k,
        'mean_k_index': mean_k,
        'std_k_index': std_k,
        'all_k_indices': all_k,
        'environment': {
            'observations': env_cfg['observations'],
            'actions': env_cfg['actions'],
            'episode_length': env_cfg['episode_length']
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Track G: Threshold Crossing')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--phase', type=str, default='g1', choices=['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g10lite', 'g2plus'])
    parser.add_argument('--warm-start-load', type=str, default=None, help='Override warm_start.load_path for selected phase')
    parser.add_argument('--warm-start-save', type=str, default=None, help='Override warm_start.save_path for selected phase')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and warm-start paths, then exit')
    parser.add_argument('--warm-start-allow-mismatch', action='store_true', help='Allow checkpoint/config hash mismatches (use with caution)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Dispatch to phase
    if args.phase == 'g1':
        results = run_phase_g1(config)
        save_results(results, config)
        print("\nTrack G Phase G1 complete!")
    
    elif args.phase == 'g2':
        results = run_phase_g2(config)
        save_results(results, config)
        print("\nTrack G Phase G2 complete!")
    
    elif args.phase == 'g3':
        results = run_phase_g3(config)
        save_results(results, config)
        print("\nTrack G Phase G3 complete!")
    
    elif args.phase == 'g4':
        results = run_phase_g4(config)
        save_results(results, config)
        print("\nTrack G Phase G4 complete!")
    
    elif args.phase == 'g5':
        results = run_phase_g5(config)
        save_results(results, config)
        print("\nTrack G Phase G5 complete!")
    
    elif args.phase == 'g6':
        results = run_phase_g6(config)
        save_results(results, config)
        print("\nTrack G Phase G6 complete!")
    
    elif args.phase == 'g7':
        results = run_phase_g7(config)
        save_results(results, config)
        print("\nTrack G Phase G7 complete!")
    
    elif args.phase == 'g8':
        results = run_phase_g8(config)
        save_results(results, config)
        print("\nTrack G Phase G8 complete!")
    
    elif args.phase == 'g9':
        results = run_phase_g9(config)
        save_results(results, config)
        print("\nTrack G Phase G9 complete!")
    
    elif args.phase == 'g10':
        results = run_phase_g10(config)
        save_results(results, config)
        print("\nTrack G Phase G10 complete!")
    
    elif args.phase == 'g10lite':
        results = run_phase_g10_lite(config)
        save_results(results, config)
        print("\nTrack G Phase G10-Lite complete!")
    
    elif args.phase == 'g2plus':
        results = run_phase_g2plus(config)
        save_results(results, config)
        print("\nTrack G Phase G2+ complete!")


if __name__ == '__main__':
    main()
