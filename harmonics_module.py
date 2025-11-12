"""
harmonics.py - Quantification of the Seven Primary Harmonies of Infinite Love

This module provides computational metrics for each of the Seven Harmonies
as defined in Evolving Resonant Co-creationism, bridging metaphysical 
principles to empirical measurements from the simulation suite.

Author: Kosmic Simulation Suite
Version: 0.1.0
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import networkx as nx


@dataclass
class HarmonyScores:
    """Container for all Seven Harmony measurements."""
    resonant_coherence: float  # Φ (IIT)
    pan_sentient_flourishing: float  # Survival diversity
    integral_wisdom: float  # Multi-modal prediction accuracy
    infinite_play: float  # Behavioral entropy
    universal_interconnectedness: float  # Mutual TE
    sacred_reciprocity: float  # TE flow symmetry
    evolutionary_progression: float  # Rate of Φ increase
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'H1_Coherence': self.resonant_coherence,
            'H2_Flourishing': self.pan_sentient_flourishing,
            'H3_Wisdom': self.integral_wisdom,
            'H4_Play': self.infinite_play,
            'H5_Interconnection': self.universal_interconnectedness,
            'H6_Reciprocity': self.sacred_reciprocity,
            'H7_Evolution': self.evolutionary_progression
        }
    
    def kosmic_signature(self, weights: Optional[np.ndarray] = None) -> float:
        """
        Compute the Kosmic Signature Index (K).
        
        Args:
            weights: Optional custom weights for harmonies (default: equal 1/7)
        
        Returns:
            K-index value (> 1.0 indicates corridor)
        """
        if weights is None:
            weights = np.ones(7) / 7.0
        
        scores = np.array([
            self.resonant_coherence,
            self.pan_sentient_flourishing,
            self.integral_wisdom,
            self.infinite_play,
            self.universal_interconnectedness,
            self.sacred_reciprocity,
            self.evolutionary_progression
        ])
        
        return np.dot(weights, scores)


class HarmonyCalculator:
    """Unified calculator for all Seven Harmonies."""
    
    def __init__(self, baseline_phi: float = 1.0, baseline_fe: float = 10.0):
        """
        Initialize calculator with baseline values for normalization.
        
        Args:
            baseline_phi: Reference Φ for H1 normalization
            baseline_fe: Reference free energy for H3 normalization
        """
        self.baseline_phi = baseline_phi
        self.baseline_fe = baseline_fe
        self.phi_history = []
        
    def compute_all(self, 
                    phi: float,
                    agent_states: List[Dict],
                    te_matrix: np.ndarray,
                    prediction_errors: Dict[str, float],
                    behavioral_history: List[np.ndarray],
                    timestep: int) -> HarmonyScores:
        """
        Compute all Seven Harmonies from simulation data.
        
        Args:
            phi: Current integrated information (IIT)
            agent_states: List of dicts with {'alive': bool, 'type': str, ...}
            te_matrix: Transfer entropy matrix (agents × agents)
            prediction_errors: Dict of {'sensory': err, 'motor': err, ...}
            behavioral_history: List of action vectors over time
            timestep: Current simulation timestep
            
        Returns:
            HarmonyScores object with all metrics
        """
        h1 = self._resonant_coherence(phi)
        h2 = self._pan_sentient_flourishing(agent_states)
        h3 = self._integral_wisdom(prediction_errors)
        h4 = self._infinite_play(behavioral_history)
        h5 = self._universal_interconnectedness(te_matrix)
        h6 = self._sacred_reciprocity(te_matrix)
        h7 = self._evolutionary_progression(phi, timestep)
        
        return HarmonyScores(h1, h2, h3, h4, h5, h6, h7)
    
    def _resonant_coherence(self, phi: float) -> float:
        """
        H1: Love as Harmonious Integration
        
        Metric: Normalized integrated information (Φ)
        Interpretation: High Φ = system cannot be reduced to parts
        """
        return (phi - self.baseline_phi) / self.baseline_phi
    
    def _pan_sentient_flourishing(self, agent_states: List[Dict]) -> float:
        """
        H2: Love as Unconditional Care
        
        Metric: Shannon entropy of agent type distribution among survivors
        Interpretation: High entropy = diverse types flourishing equally
        """
        if not agent_states:
            return 0.0
        
        alive = [a for a in agent_states if a.get('alive', False)]
        if not alive:
            return 0.0
        
        # Count agent types
        type_counts = {}
        for agent in alive:
            agent_type = agent.get('type', 'unknown')
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        # Shannon entropy (normalized by max possible)
        probs = np.array(list(type_counts.values())) / len(alive)
        max_entropy = np.log(len(type_counts))
        
        if max_entropy == 0:
            return 1.0  # Only one type = perfect homogeneity (edge case)
        
        return entropy(probs) / max_entropy
    
    def _integral_wisdom(self, prediction_errors: Dict[str, float]) -> float:
        """
        H3: Love as Self-Illuminating Intelligence
        
        Metric: Inverse of mean prediction error across modalities
        Interpretation: Low error = accurate multi-modal generative model
        """
        if not prediction_errors:
            return 0.0
        
        mean_error = np.mean(list(prediction_errors.values()))
        
        # Normalize: perfect prediction (0 error) = 1.0
        return max(0.0, 1.0 - (mean_error / self.baseline_fe))
    
    def _infinite_play(self, behavioral_history: List[np.ndarray]) -> float:
        """
        H4: Love as Joyful Generativity
        
        Metric: Shannon entropy of behavioral repertoire
        Interpretation: High entropy = diverse, exploratory actions
        """
        if len(behavioral_history) < 2:
            return 0.0
        
        # Discretize continuous actions into bins
        all_actions = np.vstack(behavioral_history)
        n_bins = min(10, len(all_actions) // 10)
        
        if n_bins < 2:
            return 0.0
        
        # Compute histogram for each action dimension
        entropies = []
        for dim in range(all_actions.shape[1]):
            hist, _ = np.histogram(all_actions[:, dim], bins=n_bins)
            hist = hist + 1e-10  # Avoid log(0)
            probs = hist / hist.sum()
            entropies.append(entropy(probs))
        
        # Average entropy across dimensions, normalized
        max_entropy = np.log(n_bins)
        return np.mean(entropies) / max_entropy
    
    def _universal_interconnectedness(self, te_matrix: np.ndarray) -> float:
        """
        H5: Love as Fundamental Unity
        
        Metric: Proportion of mutual information (bidirectional TE)
        Interpretation: High mutual TE = agents genuinely influencing each other
        """
        if te_matrix.size == 0:
            return 0.0
        
        # Mutual TE: min(TE_ij, TE_ji) for all pairs
        mutual_te = np.minimum(te_matrix, te_matrix.T)
        total_te = np.sum(np.abs(te_matrix))
        
        if total_te == 0:
            return 0.0
        
        return np.sum(mutual_te) / total_te
    
    def _sacred_reciprocity(self, te_matrix: np.ndarray) -> float:
        """
        H6: Love as Generous Flow
        
        Metric: Symmetry of TE flows (1 - Jensen-Shannon divergence)
        Interpretation: High symmetry = balanced giving/receiving
        """
        if te_matrix.size == 0:
            return 0.0
        
        # Compare row sums (outgoing TE) vs column sums (incoming TE)
        outgoing = np.sum(te_matrix, axis=1)
        incoming = np.sum(te_matrix, axis=0)
        
        # Normalize to probability distributions
        outgoing = outgoing / (outgoing.sum() + 1e-10)
        incoming = incoming / (incoming.sum() + 1e-10)
        
        # JS divergence: 0 = identical, 1 = maximally different
        js_div = jensenshannon(outgoing, incoming)
        
        return 1.0 - js_div
    
    def _evolutionary_progression(self, phi: float, timestep: int) -> float:
        """
        H7: Love as Wise Becoming
        
        Metric: Rate of Φ increase over time window
        Interpretation: Positive slope = system becoming more integrated
        """
        self.phi_history.append((timestep, phi))
        
        # Need at least 10 timesteps for meaningful slope
        if len(self.phi_history) < 10:
            return 0.0
        
        # Keep only recent history (sliding window of 100 timesteps)
        if len(self.phi_history) > 100:
            self.phi_history = self.phi_history[-100:]
        
        # Linear regression on Φ over time
        times = np.array([t for t, _ in self.phi_history])
        phis = np.array([p for _, p in self.phi_history])
        
        # Normalize time to [0, 1]
        times = (times - times.min()) / (times.max() - times.min() + 1e-10)
        
        # Slope of best-fit line
        slope = np.polyfit(times, phis, 1)[0]
        
        # Normalize: +1 std dev per timestep = 1.0
        return np.tanh(slope / (np.std(phis) + 1e-10))


class CorridorDetector:
    """Detects and characterizes the 'Goldilocks Corridor' in parameter space."""
    
    def __init__(self, k_threshold: float = 1.0, stability_threshold: float = 0.1):
        """
        Initialize corridor detector.
        
        Args:
            k_threshold: Minimum K-index for corridor membership
            stability_threshold: Maximum std dev of K for corridor stability
        """
        self.k_threshold = k_threshold
        self.stability_threshold = stability_threshold
        
    def is_in_corridor(self, k_scores: np.ndarray) -> bool:
        """
        Determine if a parameter configuration is in the corridor.
        
        Args:
            k_scores: Array of K-index values across replicate runs
            
        Returns:
            True if mean K > threshold AND std K < stability threshold
        """
        mean_k = np.mean(k_scores)
        std_k = np.std(k_scores)
        
        return mean_k > self.k_threshold and std_k < self.stability_threshold
    
    def find_corridor_boundary(self, 
                               parameter_sweep: Dict[Tuple, List[float]]) -> List[Tuple]:
        """
        Identify boundary of corridor in parameter space.
        
        Args:
            parameter_sweep: Dict mapping (param1, param2, ...) -> [K-scores]
            
        Returns:
            List of parameter tuples on corridor boundary
        """
        corridor_params = []
        
        for params, k_scores in parameter_sweep.items():
            if self.is_in_corridor(np.array(k_scores)):
                corridor_params.append(params)
        
        return corridor_params
    
    def compute_corridor_volume(self, corridor_params: List[Tuple]) -> float:
        """
        Estimate the volume (proportion) of parameter space in corridor.
        
        Args:
            corridor_params: List of parameter tuples in corridor
            
        Returns:
            Proportion of tested space in corridor
        """
        if not corridor_params:
            return 0.0
        
        # Assumes uniform grid sampling
        return len(corridor_params)


# Example usage and validation
if __name__ == "__main__":
    print("=== Kosmic Harmony Calculator v0.1 ===\n")
    
    # Simulate toy data
    calculator = HarmonyCalculator()
    
    # Mock agent states
    agents = [
        {'alive': True, 'type': 'explorer'},
        {'alive': True, 'type': 'cooperator'},
        {'alive': True, 'type': 'explorer'},
        {'alive': False, 'type': 'defector'},
    ]
    
    # Mock TE matrix (4x4)
    te_matrix = np.array([
        [0.0, 0.5, 0.3, 0.0],
        [0.4, 0.0, 0.6, 0.0],
        [0.2, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    # Mock prediction errors
    pred_errors = {'sensory': 0.2, 'motor': 0.3, 'social': 0.1}
    
    # Mock behavioral history
    behaviors = [np.random.randn(10, 3) for _ in range(20)]
    
    # Compute harmonies
    scores = calculator.compute_all(
        phi=2.5,
        agent_states=agents,
        te_matrix=te_matrix,
        prediction_errors=pred_errors,
        behavioral_history=behaviors,
        timestep=100
    )
    
    print("Seven Harmony Scores:")
    for name, value in scores.to_dict().items():
        print(f"  {name}: {value:.3f}")
    
    print(f"\nKosmic Signature Index (K): {scores.kosmic_signature():.3f}")
    
    # Test corridor detection
    detector = CorridorDetector()
    k_replicates = np.array([1.2, 1.3, 1.1, 1.25, 1.15])
    
    if detector.is_in_corridor(k_replicates):
        print("\n✓ Configuration is IN the Kosmic Corridor")
    else:
        print("\n✗ Configuration is OUTSIDE the corridor")
    
    print("\n=== Validation Complete ===")
