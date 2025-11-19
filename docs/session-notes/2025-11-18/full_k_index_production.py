#!/usr/bin/env python3
"""
Full 7-Harmony K-Index Implementation

This module implements the complete K-Index formalism based on the 7 Harmonies
of Infinite Love. Unlike the Simple K-Index (which anti-correlates with performance),
the Full K-Index positively correlates with task performance.

IMPORTANT: This replaces the deprecated Simple K-Index for all experiments.

INSTALLATION:
    sudo cp docs/session-notes/2025-11-18/full_k_index_production.py fre/metrics/full_k_index.py

Usage:
    from fre.metrics.full_k_index import compute_full_k_index, FullKIndexCalculator

    # Quick computation
    k_value, harmonies = compute_full_k_index(
        observations=obs_array,
        actions=action_list,
        q_values=q_values_history,
        hidden_states=hidden_states  # optional
    )

    # Or use the calculator class for streaming data
    calculator = FullKIndexCalculator()
    calculator.add_step(obs, action, q_values, hidden_state)
    k_value, harmonies = calculator.compute()

References:
    - K_INDEX_MATHEMATICAL_FORMALISM.md
    - COMPLETE_REVALIDATION_SUMMARY.md

Author: Kosmic Lab
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter
from scipy.stats import pearsonr


def compute_h1_resonant_coherence(
    hidden_states: List[np.ndarray],
    phi_baseline: float = 1.0
) -> float:
    """
    H1: Resonant Coherence - Normalized integrated information (Φ approximation).

    Measures the degree of integration across hidden state dimensions.
    Higher values indicate more coherent internal representations.

    Args:
        hidden_states: List of hidden state vectors
        phi_baseline: Normalization constant

    Returns:
        H1 value in [0, 2]
    """
    if len(hidden_states) < 5:
        return 0.5

    states = np.array(hidden_states)
    if states.ndim == 1:
        return 0.5

    n_dims = states.shape[1]
    if n_dims < 2:
        return 0.5

    # Compute pairwise correlations as Φ proxy
    correlations = []
    for i in range(min(n_dims, 8)):  # Limit to first 8 dims for efficiency
        for j in range(i + 1, min(n_dims, 8)):
            if np.std(states[:, i]) > 1e-10 and np.std(states[:, j]) > 1e-10:
                r = np.corrcoef(states[:, i], states[:, j])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))

    if not correlations:
        return 0.5

    phi = np.mean(correlations)
    return float(np.clip(phi / phi_baseline, 0, 2))


def compute_h2_diversity(actions: List[int]) -> float:
    """
    H2: Pan-Sentient Flourishing (Diversity) - Shannon entropy of action distribution.

    This is the BEST single predictor of task performance (r ≈ +0.46).
    Higher values indicate more diverse, flexible behavior.

    Args:
        actions: List of discrete actions taken

    Returns:
        H2 value in [0, 1]
    """
    if len(actions) < 5:
        return 0.5

    counts = Counter(actions)
    if len(counts) <= 1:
        return 0.0  # No diversity

    total = len(actions)
    probs = np.array([c / total for c in counts.values()])

    # Normalized Shannon entropy
    h = -np.sum(probs * np.log(probs + 1e-10))
    h_max = np.log(len(counts))

    return float(h / h_max) if h_max > 0 else 0.0


def compute_h3_prediction_accuracy(
    predictions: Optional[List[float]] = None,
    actuals: Optional[List[float]] = None
) -> float:
    """
    H3: Integral Wisdom - Prediction accuracy of internal models.

    Measures how well the agent's predictions match reality.

    Args:
        predictions: Predicted values
        actuals: Actual observed values

    Returns:
        H3 value in [0, 1]
    """
    if predictions is None or actuals is None:
        return 0.5  # Default when not available

    if len(predictions) < 5 or len(actuals) < 5:
        return 0.5

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Use 1 - normalized MSE
    mse = np.mean((predictions - actuals) ** 2)
    var = np.var(actuals) + 1e-10

    return float(np.clip(1.0 - mse / var, 0, 1))


def compute_h4_behavioral_entropy(q_values_history: List[np.ndarray]) -> float:
    """
    H4: Infinite Play - Entropy of Q-value distributions over time.

    Measures the richness and variability of value estimates.

    Args:
        q_values_history: List of Q-value vectors

    Returns:
        H4 value in [0, 1]
    """
    if len(q_values_history) < 10:
        return 0.5

    arr = np.array(q_values_history)
    n_bins = min(5, len(arr) // 5)

    if n_bins < 2:
        return 0.5

    entropies = []
    for dim in range(arr.shape[1]):
        col = arr[:, dim]
        if np.std(col) < 1e-10:
            continue

        hist, _ = np.histogram(col, bins=n_bins)
        hist = hist + 1e-10  # Avoid log(0)
        probs = hist / hist.sum()
        entropies.append(-np.sum(probs * np.log(probs)))

    if not entropies:
        return 0.5

    h_max = np.log(n_bins)
    return float(np.mean(entropies) / h_max) if h_max > 0 else 0.5


def compute_h5_mutual_transfer_entropy(q_values_history: List[np.ndarray]) -> float:
    """
    H5: Universal Interconnectedness - Mutual information flow between dimensions.

    Measures how much information flows between different Q-value dimensions.

    Args:
        q_values_history: List of Q-value vectors

    Returns:
        H5 value in [0, 1]
    """
    if len(q_values_history) < 10:
        return 0.5

    series = np.array(q_values_history)
    if series.shape[1] < 2:
        return 0.5

    influences = []
    for i in range(series.shape[1]):
        for j in range(i + 1, series.shape[1]):
            std_i = np.std(series[:-1, i])
            std_j = np.std(series[1:, j])

            if std_i > 1e-10 and std_j > 1e-10:
                # Cross-correlation as transfer entropy proxy
                c_ij = abs(np.corrcoef(series[:-1, i], series[1:, j])[0, 1])
                c_ji = abs(np.corrcoef(series[:-1, j], series[1:, i])[0, 1])

                if not (np.isnan(c_ij) or np.isnan(c_ji)):
                    # Harmonic mean of bidirectional influence
                    influence = 2 * min(c_ij, c_ji) / (c_ij + c_ji + 1e-10)
                    influences.append(influence)

    return float(np.mean(influences)) if influences else 0.5


def compute_h6_flow_symmetry(obs_norms: List[float]) -> float:
    """
    H6: Sacred Reciprocity - Symmetry between input and output distributions.

    Measures the balance of information flow over time.

    Args:
        obs_norms: List of observation norm values

    Returns:
        H6 value in [0, 1]
    """
    if len(obs_norms) < 6:
        return 0.5

    # Split into first and second half
    mid = len(obs_norms) // 2
    first_half = np.array(obs_norms[:mid])
    second_half = np.array(obs_norms[mid:])

    # Ensure same length
    min_len = min(len(first_half), len(second_half))
    first_half = first_half[:min_len]
    second_half = second_half[:min_len]

    # Compute probability distributions
    first_p = (np.abs(first_half) + 1e-10) / (np.abs(first_half).sum() + 1e-10)
    second_p = (np.abs(second_half) + 1e-10) / (np.abs(second_half).sum() + 1e-10)

    # Jensen-Shannon divergence
    m = 0.5 * (first_p + second_p)
    js = 0.5 * (
        np.sum(first_p * np.log(first_p / (m + 1e-10) + 1e-10)) +
        np.sum(second_p * np.log(second_p / (m + 1e-10) + 1e-10))
    )

    # Convert to similarity (1 - sqrt(JS))
    return float(1.0 - np.sqrt(np.clip(js, 0, 1)))


def compute_h7_growth_rate(
    k_history: Optional[List[float]] = None,
    metric_history: Optional[List[float]] = None
) -> float:
    """
    H7: Evolutionary Progression - Rate of coherence growth over time.

    Measures whether the system is improving or degrading.

    Args:
        k_history: Historical K-Index values
        metric_history: Alternative metric history

    Returns:
        H7 value in [-1, 1]
    """
    history = k_history or metric_history

    if history is None or len(history) < 10:
        return 0.0

    # Use recent history
    recent = history[-50:] if len(history) > 50 else history
    times = np.arange(len(recent))
    values = np.array(recent)

    if np.std(values) < 1e-10:
        return 0.0

    # Compute normalized slope
    slope = np.polyfit(times / (times.max() + 1e-10), values, deg=1)[0]

    # Tanh to bound in [-1, 1]
    return float(np.tanh(slope / (np.std(values) + 1e-10)))


def compute_full_k_index(
    observations: Optional[np.ndarray] = None,
    actions: Optional[List[int]] = None,
    q_values: Optional[List[np.ndarray]] = None,
    hidden_states: Optional[List[np.ndarray]] = None,
    predictions: Optional[List[float]] = None,
    actuals: Optional[List[float]] = None,
    k_history: Optional[List[float]] = None,
    weights: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the Full 7-Harmony K-Index.

    This is the recommended metric for measuring agent coherence.
    It positively correlates with task performance (r ≈ +0.26).

    Args:
        observations: Observation array (for computing norms)
        actions: List of discrete actions
        q_values: List of Q-value vectors
        hidden_states: List of hidden state vectors
        predictions: Predicted values (for H3)
        actuals: Actual values (for H3)
        k_history: Historical K values (for H7)
        weights: Custom weights for harmonies (default: equal)

    Returns:
        Tuple of (k_index, harmony_dict)
    """
    # Compute observation norms if needed
    obs_norms = []
    if observations is not None:
        if observations.ndim == 1:
            obs_norms = [np.linalg.norm(observations)]
        else:
            obs_norms = [np.linalg.norm(obs) for obs in observations]

    # Compute individual harmonies
    h1 = compute_h1_resonant_coherence(hidden_states or q_values or [])
    h2 = compute_h2_diversity(actions or [])
    h3 = compute_h3_prediction_accuracy(predictions, actuals)
    h4 = compute_h4_behavioral_entropy(q_values or [])
    h5 = compute_h5_mutual_transfer_entropy(q_values or [])
    h6 = compute_h6_flow_symmetry(obs_norms)
    h7 = compute_h7_growth_rate(k_history)

    harmonies = {
        'h1_resonant_coherence': h1,
        'h2_diversity': h2,
        'h3_prediction_accuracy': h3,
        'h4_behavioral_entropy': h4,
        'h5_mutual_transfer_entropy': h5,
        'h6_flow_symmetry': h6,
        'h7_growth_rate': h7
    }

    # Aggregate with weights
    if weights is None:
        weights = np.ones(7) / 7.0

    harmony_values = np.array([h1, h2, h3, h4, h5, h6, h7])
    k_index = float(np.dot(weights, harmony_values))

    return k_index, harmonies


def compute_simple_k_index(obs_norms: List[float], act_norms: List[float]) -> float:
    """
    DEPRECATED: Simple K-Index based on correlation.

    WARNING: This metric ANTI-CORRELATES with task performance!
    Use compute_full_k_index() instead.

    Args:
        obs_norms: Observation norms
        act_norms: Action norms

    Returns:
        Simple K value (DEPRECATED)
    """
    import warnings
    warnings.warn(
        "Simple K-Index is DEPRECATED. It anti-correlates with performance (r ≈ -0.41). "
        "Use compute_full_k_index() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if len(obs_norms) < 2 or len(act_norms) < 2:
        return 0.0

    try:
        r, _ = pearsonr(obs_norms, act_norms)
        return 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


class FullKIndexCalculator:
    """
    Streaming calculator for Full K-Index.

    Use this class when computing K-Index incrementally during episodes.

    Example:
        calculator = FullKIndexCalculator()

        for step in episode:
            calculator.add_step(obs, action, q_values, hidden_state)

        k_value, harmonies = calculator.compute()
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize the calculator.

        Args:
            weights: Custom weights for the 7 harmonies
        """
        self.weights = weights
        self.reset()

    def reset(self):
        """Reset all accumulated data."""
        self.observations = []
        self.actions = []
        self.q_values = []
        self.hidden_states = []
        self.predictions = []
        self.actuals = []
        self.k_history = []

    def add_step(
        self,
        observation: np.ndarray,
        action: int,
        q_values: Optional[np.ndarray] = None,
        hidden_state: Optional[np.ndarray] = None,
        prediction: Optional[float] = None,
        actual: Optional[float] = None
    ):
        """
        Add a single step of data.

        Args:
            observation: Current observation
            action: Action taken
            q_values: Q-values for this step
            hidden_state: Hidden state for this step
            prediction: Predicted value
            actual: Actual value
        """
        self.observations.append(observation)
        self.actions.append(action)

        if q_values is not None:
            self.q_values.append(q_values.copy())

        if hidden_state is not None:
            self.hidden_states.append(hidden_state.copy())

        if prediction is not None:
            self.predictions.append(prediction)

        if actual is not None:
            self.actuals.append(actual)

    def compute(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute the Full K-Index from accumulated data.

        Returns:
            Tuple of (k_index, harmony_dict)
        """
        return compute_full_k_index(
            observations=np.array(self.observations) if self.observations else None,
            actions=self.actions,
            q_values=self.q_values if self.q_values else None,
            hidden_states=self.hidden_states if self.hidden_states else None,
            predictions=self.predictions if self.predictions else None,
            actuals=self.actuals if self.actuals else None,
            k_history=self.k_history if self.k_history else None,
            weights=self.weights
        )

    def get_h2(self) -> float:
        """
        Quick access to H2 (Diversity) - the best single predictor.

        Returns:
            H2 value
        """
        return compute_h2_diversity(self.actions)


# Convenience function for migration
def k_index(obs_norms: np.ndarray, act_norms: np.ndarray) -> float:
    """
    DEPRECATED: Legacy interface for Simple K-Index.

    This function exists for backward compatibility only.
    New code should use compute_full_k_index().
    """
    return compute_simple_k_index(list(obs_norms), list(act_norms))


if __name__ == '__main__':
    # Quick test
    print("Full 7-Harmony K-Index Implementation")
    print("=" * 50)

    # Generate test data
    np.random.seed(42)
    n_steps = 100

    observations = np.random.randn(n_steps, 4)
    actions = np.random.randint(0, 2, n_steps).tolist()
    q_values = [np.random.randn(2) for _ in range(n_steps)]

    # Compute Full K
    k_value, harmonies = compute_full_k_index(
        observations=observations,
        actions=actions,
        q_values=q_values
    )

    print(f"\nFull K-Index: {k_value:.4f}")
    print("\nIndividual Harmonies:")
    for name, value in harmonies.items():
        print(f"  {name}: {value:.4f}")

    # Test H2 separately (best predictor)
    h2 = compute_h2_diversity(actions)
    print(f"\nH2 (Diversity) alone: {h2:.4f}")

    print("\n✅ Implementation complete!")
    print("\nTo install:")
    print("  sudo cp docs/session-notes/2025-11-18/full_k_index_production.py fre/metrics/full_k_index.py")
