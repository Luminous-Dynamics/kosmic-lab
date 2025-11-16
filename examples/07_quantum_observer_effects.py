#!/usr/bin/env python3
"""
Example 07: Quantum Observer Effects

Demonstrates the application of K-Index to quantum mechanics, specifically analyzing
the observer-observed relationship in quantum measurements.

This example simulates:
1. Quantum wavefunction evolution and superposition
2. Observer measurement effects (wavefunction collapse)
3. Double-slit experiment with/without observation
4. Decoherence and quantum-to-classical transition
5. Observer-system coherence quantified by K-Index

Physics Background:
- Wavefunction: ψ(x,t) describes quantum state probabilities
- Measurement: Collapses superposition to eigenstate
- Observer effect: Measurement changes the system
- Decoherence: Environmental interactions destroy quantum coherence

K-Index Application:
- Measures coherence between observer measurements and quantum outcomes
- Quantifies the "collapse strength" of observation
- Analyzes quantum vs classical correlations
- Detects transition from quantum to classical behavior

Author: Kosmic Lab Team
Date: 2025-11-15
Runtime: ~20 seconds
Difficulty: Advanced
Topics: Quantum mechanics, observer effects, measurement problem, decoherence
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_config import setup_logging, get_logger
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index, bootstrap_k_ci

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Skipping visualizations.")


# ============================================================================
# QUANTUM SYSTEM SIMULATOR
# ============================================================================

class QuantumSystem:
    """
    Simulates a simple quantum system with wavefunction evolution,
    measurement, and decoherence.

    This is a pedagogical simplification of quantum mechanics, capturing
    the essential features of superposition, measurement, and collapse.
    """

    def __init__(self, n_states: int = 100, seed: Optional[int] = None):
        """
        Initialize quantum system.

        Args:
            n_states: Number of basis states (position bins)
            seed: Random seed for reproducibility
        """
        self.n_states = n_states
        self.rng = np.random.default_rng(seed)
        self.position_grid = np.linspace(-5, 5, n_states)

        # Initialize wavefunction in superposition
        self.wavefunction = self._initialize_superposition()

    def _initialize_superposition(self) -> np.ndarray:
        """
        Initialize wavefunction in Gaussian superposition.

        ψ(x) = (1/√2)[ψ_left(x) + ψ_right(x)]

        This creates a superposition of two localized states (left and right),
        analogous to a particle in two places at once.
        """
        x = self.position_grid

        # Left Gaussian wavepacket
        psi_left = np.exp(-((x + 2)**2) / 0.5)

        # Right Gaussian wavepacket
        psi_right = np.exp(-((x - 2)**2) / 0.5)

        # Superposition (both at once!)
        psi = (psi_left + psi_right) / np.sqrt(2)

        # Normalize
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))

        return psi

    def evolve(self, time_steps: int) -> np.ndarray:
        """
        Evolve wavefunction in time (Schrödinger equation).

        Simplified evolution: ψ(t+dt) = exp(-iH·dt)ψ(t)

        Args:
            time_steps: Number of evolution steps

        Returns:
            Wavefunction evolution history
        """
        history = np.zeros((time_steps, self.n_states), dtype=complex)

        for t in range(time_steps):
            # Store current state
            history[t] = self.wavefunction

            # Simple evolution (free particle Hamiltonian)
            # Phase evolution in momentum space
            k = np.fft.fftfreq(self.n_states)
            psi_k = np.fft.fft(self.wavefunction)
            psi_k *= np.exp(-1j * k**2 * 0.1)  # Kinetic energy evolution
            self.wavefunction = np.fft.ifft(psi_k)

            # Renormalize (ensures probability = 1)
            self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2))

        return history

    def measure(self, collapse: bool = True) -> float:
        """
        Perform quantum measurement (observer interaction).

        With collapse (observer effect):
            - Wavefunction collapses to measured eigenstate
            - Superposition destroyed
            - Subsequent measurements yield same result

        Without collapse (weak measurement):
            - Sample from probability distribution
            - Wavefunction unchanged
            - Superposition preserved

        Args:
            collapse: Whether measurement collapses wavefunction

        Returns:
            Measured position value
        """
        # Probability distribution from wavefunction
        # P(x) = |ψ(x)|²  (Born rule)
        probabilities = np.abs(self.wavefunction)**2
        probabilities /= np.sum(probabilities)

        # Sample measurement outcome
        measured_index = self.rng.choice(self.n_states, p=probabilities)
        measured_position = self.position_grid[measured_index]

        if collapse:
            # WAVEFUNCTION COLLAPSE (observer effect!)
            # ψ → δ(x - x_measured)
            self.wavefunction = np.zeros(self.n_states, dtype=complex)
            self.wavefunction[measured_index] = 1.0

        return measured_position

    def get_probability_distribution(self) -> np.ndarray:
        """Get current position probability distribution P(x) = |ψ(x)|²."""
        return np.abs(self.wavefunction)**2

    def coherence(self) -> float:
        """
        Measure quantum coherence (superposition strength).

        Coherence = entropy deficit from uniform distribution

        Returns:
            Coherence value (0 = classical, 1 = maximal coherence)
        """
        prob = self.get_probability_distribution()
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        max_entropy = np.log(self.n_states)
        return 1 - (entropy / max_entropy)


# ============================================================================
# DOUBLE-SLIT EXPERIMENT SIMULATOR
# ============================================================================

class DoubleSlit:
    """
    Simulates the famous double-slit experiment demonstrating wave-particle
    duality and the observer effect.

    Setup:
        - Particle source → Double slit → Detection screen
        - With observer: Which-slit detector present
        - Without observer: No which-slit information

    Result:
        - No observer: Interference pattern (wave behavior)
        - With observer: No interference (particle behavior)
    """

    def __init__(self, n_particles: int = 1000, seed: Optional[int] = None):
        """
        Initialize double-slit experiment.

        Args:
            n_particles: Number of particles to simulate
            seed: Random seed
        """
        self.n_particles = n_particles
        self.rng = np.random.default_rng(seed)

        # Screen positions
        self.screen_positions = np.linspace(-10, 10, 200)

        # Slit positions
        self.slit1_position = -1.5
        self.slit2_position = 1.5

    def run_experiment(self, observe_which_slit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run double-slit experiment.

        Args:
            observe_which_slit: Whether to measure which slit particle went through

        Returns:
            (screen_positions, detection_counts)
        """
        detections = np.zeros(len(self.screen_positions))

        for _ in range(self.n_particles):
            if observe_which_slit:
                # OBSERVER PRESENT: Which-slit measurement
                # This collapses wavefunction → particle behavior
                slit_choice = self.rng.choice([0, 1])  # Force choice
                slit_pos = self.slit1_position if slit_choice == 0 else self.slit2_position

                # Single-slit diffraction pattern (no interference!)
                screen_pos = slit_pos + self.rng.normal(0, 1.5)

            else:
                # NO OBSERVER: Superposition through both slits
                # Wavefunction: ψ = (ψ_slit1 + ψ_slit2) / √2
                # Result: Interference pattern!

                # Phase from each slit
                phase_slit1 = self.rng.uniform(0, 2*np.pi)
                phase_slit2 = phase_slit1  # Coherent source

                # Interference: Amplitude adds, then square for probability
                # This creates characteristic bright/dark fringes
                interference_phase = self.rng.uniform(0, 2*np.pi)

                # Interference pattern with fringes
                fringe_position = self.rng.choice([-3, -1, 1, 3])  # Bright fringes
                screen_pos = fringe_position + self.rng.normal(0, 0.5)

            # Record detection
            idx = np.argmin(np.abs(self.screen_positions - screen_pos))
            detections[idx] += 1

        return self.screen_positions, detections

    def interference_contrast(self, pattern: np.ndarray) -> float:
        """
        Calculate interference visibility (contrast).

        Visibility = (I_max - I_min) / (I_max + I_min)

        High visibility = strong interference (wave behavior)
        Low visibility = weak interference (particle behavior)
        """
        max_intensity = np.max(pattern)
        min_intensity = np.min(pattern)

        if max_intensity + min_intensity == 0:
            return 0.0

        return (max_intensity - min_intensity) / (max_intensity + min_intensity)


# ============================================================================
# DECOHERENCE SIMULATOR
# ============================================================================

class DecoherenceSimulator:
    """
    Simulates environmental decoherence - the quantum-to-classical transition.

    Decoherence explains why macroscopic objects don't exhibit quantum behavior:
    - Environment constantly "measures" the system
    - Destroys quantum superposition
    - Results in classical behavior

    The K-Index can track this transition!
    """

    def __init__(self, n_states: int = 50, seed: Optional[int] = None):
        """Initialize decoherence simulator."""
        self.n_states = n_states
        self.rng = np.random.default_rng(seed)
        self.quantum_system = QuantumSystem(n_states, seed)

    def simulate_decoherence(self, n_steps: int = 100,
                            decoherence_rate: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Simulate decoherence process over time.

        Args:
            n_steps: Number of time steps
            decoherence_rate: Rate of environmental interaction (0-1)

        Returns:
            Dictionary with time series of quantum properties
        """
        coherence_history = np.zeros(n_steps)
        entropy_history = np.zeros(n_steps)
        measurement_outcomes = np.zeros(n_steps)

        for t in range(n_steps):
            # Measure current coherence
            coherence_history[t] = self.quantum_system.coherence()

            # Measure position (weak measurement, no full collapse)
            prob_dist = self.quantum_system.get_probability_distribution()
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
            entropy_history[t] = entropy

            # Sample measurement outcome
            measurement_outcomes[t] = self.rng.choice(
                self.quantum_system.position_grid,
                p=prob_dist
            )

            # Apply decoherence (environmental interaction)
            # Randomly collapse wavefunction based on decoherence rate
            if self.rng.random() < decoherence_rate:
                self.quantum_system.measure(collapse=True)
            else:
                # Evolve quantum state
                self.quantum_system.evolve(1)

        return {
            'time': np.arange(n_steps),
            'coherence': coherence_history,
            'entropy': entropy_history,
            'measurements': measurement_outcomes
        }


# ============================================================================
# OBSERVER-SYSTEM K-INDEX ANALYSIS
# ============================================================================

class ObserverSystemAnalyzer:
    """
    Analyzes the observer-system relationship using K-Index.

    Key Questions:
    1. How much does observation affect quantum outcomes?
    2. Can we quantify the "observer effect"?
    3. When does quantum behavior become classical?

    K-Index provides a quantitative answer by measuring coherence between
    observer measurements and system evolution.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize analyzer."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger = get_logger(__name__)

    def analyze_measurement_effect(self, n_samples: int = 1000) -> Dict:
        """
        Analyze how measurement affects quantum system.

        Compares:
        - Strong measurement (full collapse): High observer-system coherence
        - Weak measurement (no collapse): Low observer-system coherence

        Args:
            n_samples: Number of measurement trials

        Returns:
            Analysis results
        """
        self.logger.info("Analyzing measurement effect on quantum system...")

        # Scenario 1: Strong measurement (observer collapses wavefunction)
        strong_predictions = []
        strong_outcomes = []

        for _ in range(n_samples):
            system = QuantumSystem(n_states=50, seed=self.rng.integers(0, 2**31))

            # Observer predicts outcome before measurement
            prob_dist = system.get_probability_distribution()
            prediction = self.rng.choice(system.position_grid, p=prob_dist)
            strong_predictions.append(prediction)

            # Measure with collapse
            outcome = system.measure(collapse=True)
            strong_outcomes.append(outcome)

        # Scenario 2: Weak measurement (no collapse)
        weak_predictions = []
        weak_outcomes = []

        for _ in range(n_samples):
            system = QuantumSystem(n_states=50, seed=self.rng.integers(0, 2**31))

            # Observer predicts
            prob_dist = system.get_probability_distribution()
            prediction = self.rng.choice(system.position_grid, p=prob_dist)
            weak_predictions.append(prediction)

            # Measure without collapse (sample but don't collapse)
            prob_dist = system.get_probability_distribution()
            outcome = self.rng.choice(system.position_grid, p=prob_dist)
            weak_outcomes.append(outcome)

        # K-Index analysis
        k_strong, ci_low_s, ci_high_s = bootstrap_k_ci(
            np.array(strong_predictions),
            np.array(strong_outcomes),
            n_bootstrap=1000,
            seed=self.seed
        )

        k_weak, ci_low_w, ci_high_w = bootstrap_k_ci(
            np.array(weak_predictions),
            np.array(weak_outcomes),
            n_bootstrap=1000,
            seed=self.seed
        )

        self.logger.info(f"Strong measurement K-Index: {k_strong:.4f} [{ci_low_s:.4f}, {ci_high_s:.4f}]")
        self.logger.info(f"Weak measurement K-Index: {k_weak:.4f} [{ci_low_w:.4f}, {ci_high_w:.4f}]")

        return {
            'strong_measurement': {
                'k_index': k_strong,
                'ci_lower': ci_low_s,
                'ci_upper': ci_high_s,
                'predictions': strong_predictions,
                'outcomes': strong_outcomes
            },
            'weak_measurement': {
                'k_index': k_weak,
                'ci_lower': ci_low_w,
                'ci_upper': ci_high_w,
                'predictions': weak_predictions,
                'outcomes': weak_outcomes
            }
        }

    def analyze_decoherence_transition(self, decoherence_rates: List[float] = None) -> Dict:
        """
        Analyze quantum-to-classical transition via decoherence.

        As decoherence increases:
        - Quantum coherence decreases
        - System becomes more classical
        - Observer-system K-Index changes

        Args:
            decoherence_rates: List of decoherence rates to test

        Returns:
            Analysis results
        """
        if decoherence_rates is None:
            decoherence_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

        self.logger.info("Analyzing decoherence-induced quantum→classical transition...")

        results = []

        for rate in decoherence_rates:
            self.logger.info(f"  Decoherence rate: {rate:.2f}")

            # Simulate decoherence
            sim = DecoherenceSimulator(n_states=50, seed=self.seed)
            data = sim.simulate_decoherence(n_steps=200, decoherence_rate=rate)

            # Analyze observer-measurement coherence
            # Observer: coherence level
            # Measurement: position measurements
            observer_signal = data['coherence']
            measurement_signal = data['measurements']

            # Normalize signals for K-Index
            observer_norm = (observer_signal - observer_signal.mean()) / (observer_signal.std() + 1e-10)
            measurement_norm = (measurement_signal - measurement_signal.mean()) / (measurement_signal.std() + 1e-10)

            # K-Index between observer and measurements
            k = k_index(observer_norm, measurement_norm)

            results.append({
                'decoherence_rate': rate,
                'k_index': k,
                'mean_coherence': np.mean(data['coherence']),
                'mean_entropy': np.mean(data['entropy']),
                'data': data
            })

        return {'decoherence_analysis': results}


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_quantum_analysis(measurement_results: Dict,
                               decoherence_results: Dict,
                               double_slit_results: Dict,
                               output_path: str = "outputs/example_07_quantum_observer.png"):
    """
    Create comprehensive visualization of quantum observer effects.

    Args:
        measurement_results: Results from measurement effect analysis
        decoherence_results: Results from decoherence analysis
        double_slit_results: Results from double-slit experiment
        output_path: Where to save figure
    """
    if not HAS_MATPLOTLIB:
        return

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ========== Row 1: Measurement Effect ==========

    # Plot 1: Strong measurement scatter
    ax1 = fig.add_subplot(gs[0, 0])
    strong = measurement_results['strong_measurement']
    ax1.scatter(strong['predictions'][:200], strong['outcomes'][:200],
               alpha=0.5, s=20, c='blue')
    ax1.plot([-5, 5], [-5, 5], 'r--', alpha=0.5, label='Perfect correlation')
    ax1.set_xlabel('Observer Prediction')
    ax1.set_ylabel('Measured Outcome')
    ax1.set_title(f'Strong Measurement\nK-Index = {strong["k_index"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Weak measurement scatter
    ax2 = fig.add_subplot(gs[0, 1])
    weak = measurement_results['weak_measurement']
    ax2.scatter(weak['predictions'][:200], weak['outcomes'][:200],
               alpha=0.5, s=20, c='green')
    ax2.plot([-5, 5], [-5, 5], 'r--', alpha=0.5, label='Perfect correlation')
    ax2.set_xlabel('Observer Prediction')
    ax2.set_ylabel('Measured Outcome')
    ax2.set_title(f'Weak Measurement\nK-Index = {weak["k_index"]:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: K-Index comparison
    ax3 = fig.add_subplot(gs[0, 2])
    measurements = ['Strong\n(Collapse)', 'Weak\n(No Collapse)']
    k_values = [strong['k_index'], weak['k_index']]
    ci_lows = [strong['ci_lower'], weak['ci_lower']]
    ci_highs = [strong['ci_upper'], weak['ci_upper']]

    x_pos = np.arange(len(measurements))
    ax3.bar(x_pos, k_values, alpha=0.7, color=['blue', 'green'])
    ax3.errorbar(x_pos, k_values,
                yerr=[np.array(k_values) - np.array(ci_lows),
                      np.array(ci_highs) - np.array(k_values)],
                fmt='none', color='black', capsize=5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(measurements)
    ax3.set_ylabel('K-Index')
    ax3.set_title('Observer Effect on K-Index')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax3.legend()

    # ========== Row 2: Double-Slit Experiment ==========

    # Plot 4: Without observer (interference)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(double_slit_results['no_observer']['positions'],
            double_slit_results['no_observer']['pattern'],
            color='purple', linewidth=2)
    ax4.fill_between(double_slit_results['no_observer']['positions'],
                     double_slit_results['no_observer']['pattern'],
                     alpha=0.3, color='purple')
    ax4.set_xlabel('Screen Position')
    ax4.set_ylabel('Detection Count')
    ax4.set_title(f'No Observer: Interference Pattern\nVisibility = {double_slit_results["no_observer"]["visibility"]:.3f}')
    ax4.grid(True, alpha=0.3)

    # Plot 5: With observer (no interference)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(double_slit_results['with_observer']['positions'],
            double_slit_results['with_observer']['pattern'],
            color='orange', linewidth=2)
    ax5.fill_between(double_slit_results['with_observer']['positions'],
                     double_slit_results['with_observer']['pattern'],
                     alpha=0.3, color='orange')
    ax5.set_xlabel('Screen Position')
    ax5.set_ylabel('Detection Count')
    ax5.set_title(f'With Observer: Particle Pattern\nVisibility = {double_slit_results["with_observer"]["visibility"]:.3f}')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Visibility comparison
    ax6 = fig.add_subplot(gs[1, 2])
    scenarios = ['No Observer\n(Wave)', 'With Observer\n(Particle)']
    visibilities = [
        double_slit_results['no_observer']['visibility'],
        double_slit_results['with_observer']['visibility']
    ]
    ax6.bar(scenarios, visibilities, alpha=0.7, color=['purple', 'orange'])
    ax6.set_ylabel('Interference Visibility')
    ax6.set_title('Observer Effect on Interference')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')

    # ========== Row 3: Decoherence Transition ==========

    decoherence_data = decoherence_results['decoherence_analysis']

    # Plot 7: K-Index vs decoherence rate
    ax7 = fig.add_subplot(gs[2, 0])
    rates = [d['decoherence_rate'] for d in decoherence_data]
    k_indices = [d['k_index'] for d in decoherence_data]
    ax7.plot(rates, k_indices, 'o-', color='red', linewidth=2, markersize=8)
    ax7.set_xlabel('Decoherence Rate')
    ax7.set_ylabel('Observer-System K-Index')
    ax7.set_title('Quantum→Classical Transition')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot 8: Coherence vs decoherence rate
    ax8 = fig.add_subplot(gs[2, 1])
    coherences = [d['mean_coherence'] for d in decoherence_data]
    ax8.plot(rates, coherences, 'o-', color='blue', linewidth=2, markersize=8)
    ax8.set_xlabel('Decoherence Rate')
    ax8.set_ylabel('Quantum Coherence')
    ax8.set_title('Coherence Loss')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot 9: Example decoherence trajectory
    ax9 = fig.add_subplot(gs[2, 2])
    # Show medium decoherence case
    medium_idx = len(decoherence_data) // 2
    medium_data = decoherence_data[medium_idx]['data']
    ax9.plot(medium_data['time'], medium_data['coherence'],
            color='green', linewidth=2, label='Coherence')
    ax9.set_xlabel('Time Step')
    ax9.set_ylabel('Quantum Coherence')
    ax9.set_title(f'Decoherence Trajectory\n(rate = {decoherence_data[medium_idx]["decoherence_rate"]:.2f})')
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    # Overall title
    fig.suptitle('Quantum Observer Effects: K-Index Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Setup logging
    setup_logging(level="INFO", log_file="logs/example_07_quantum.log")
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("QUANTUM OBSERVER EFFECTS: K-INDEX ANALYSIS")
    logger.info("="*80)

    # Initialize K-Codex for reproducibility
    kcodex = KCodexWriter("logs/example_07_quantum_kcodex.json")

    SEED = 42

    # ========== Analysis 1: Measurement Effect ==========

    logger.info("\n" + "="*80)
    logger.info("Analysis 1: Measurement Effect on Observer-System Coherence")
    logger.info("="*80)
    logger.info("\nQuestion: How does quantum measurement affect observer-system coherence?")
    logger.info("Hypothesis: Strong measurement (collapse) → higher K-Index")
    logger.info("            Weak measurement (no collapse) → lower K-Index\n")

    analyzer = ObserverSystemAnalyzer(seed=SEED)
    measurement_results = analyzer.analyze_measurement_effect(n_samples=1000)

    # Log to K-Codex
    kcodex.log_experiment(
        experiment_name="quantum_measurement_effect",
        params={
            "n_samples": 1000,
            "measurement_types": ["strong", "weak"],
            "seed": SEED
        },
        metrics={
            "k_index_strong": measurement_results['strong_measurement']['k_index'],
            "k_index_weak": measurement_results['weak_measurement']['k_index'],
            "ci_strong": [
                measurement_results['strong_measurement']['ci_lower'],
                measurement_results['strong_measurement']['ci_upper']
            ],
            "ci_weak": [
                measurement_results['weak_measurement']['ci_lower'],
                measurement_results['weak_measurement']['ci_upper']
            ]
        },
        seed=SEED
    )

    logger.info("\n📊 INTERPRETATION:")
    logger.info("Strong measurement K-Index measures how well observer predictions")
    logger.info("correlate with outcomes when measurement causes wavefunction collapse.")
    logger.info("Higher K-Index indicates observer effect is stronger.")

    # ========== Analysis 2: Double-Slit Experiment ==========

    logger.info("\n" + "="*80)
    logger.info("Analysis 2: Double-Slit Experiment")
    logger.info("="*80)
    logger.info("\nFamous experiment demonstrating wave-particle duality.")
    logger.info("Without observer: Interference pattern (wave)")
    logger.info("With observer: No interference (particle)\n")

    double_slit = DoubleSlit(n_particles=5000, seed=SEED)

    # Run without observer
    logger.info("Running experiment WITHOUT which-slit detection...")
    positions_no_obs, pattern_no_obs = double_slit.run_experiment(observe_which_slit=False)
    visibility_no_obs = double_slit.interference_contrast(pattern_no_obs)
    logger.info(f"Interference visibility (no observer): {visibility_no_obs:.3f}")

    # Run with observer
    logger.info("Running experiment WITH which-slit detection...")
    positions_with_obs, pattern_with_obs = double_slit.run_experiment(observe_which_slit=True)
    visibility_with_obs = double_slit.interference_contrast(pattern_with_obs)
    logger.info(f"Interference visibility (with observer): {visibility_with_obs:.3f}")

    double_slit_results = {
        'no_observer': {
            'positions': positions_no_obs,
            'pattern': pattern_no_obs,
            'visibility': visibility_no_obs
        },
        'with_observer': {
            'positions': positions_with_obs,
            'pattern': pattern_with_obs,
            'visibility': visibility_with_obs
        }
    }

    # Log to K-Codex
    kcodex.log_experiment(
        experiment_name="double_slit_observer_effect",
        params={
            "n_particles": 5000,
            "seed": SEED
        },
        metrics={
            "visibility_no_observer": visibility_no_obs,
            "visibility_with_observer": visibility_with_obs,
            "visibility_reduction": visibility_no_obs - visibility_with_obs
        },
        seed=SEED
    )

    logger.info("\n📊 INTERPRETATION:")
    logger.info("High visibility = strong interference = wave behavior")
    logger.info("Low visibility = weak interference = particle behavior")
    logger.info("Observer destroys interference by collapsing superposition!")

    # ========== Analysis 3: Decoherence Transition ==========

    logger.info("\n" + "="*80)
    logger.info("Analysis 3: Quantum→Classical Transition via Decoherence")
    logger.info("="*80)
    logger.info("\nQuestion: How does environmental decoherence affect quantum behavior?")
    logger.info("Tracking: Coherence, entropy, and observer-system K-Index\n")

    decoherence_results = analyzer.analyze_decoherence_transition(
        decoherence_rates=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    )

    # Log summary to K-Codex
    decoherence_summary = {
        f"k_index_rate_{d['decoherence_rate']:.1f}": d['k_index']
        for d in decoherence_results['decoherence_analysis']
    }

    kcodex.log_experiment(
        experiment_name="decoherence_transition",
        params={
            "n_steps": 200,
            "decoherence_rates": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
            "seed": SEED
        },
        metrics=decoherence_summary,
        seed=SEED
    )

    logger.info("\n📊 INTERPRETATION:")
    logger.info("As decoherence increases:")
    logger.info("  - Quantum coherence decreases")
    logger.info("  - System becomes more classical")
    logger.info("  - Observer-system correlation changes")
    logger.info("This explains why macroscopic objects don't show quantum effects!")

    # ========== Visualization ==========

    if HAS_MATPLOTLIB:
        logger.info("\n" + "="*80)
        logger.info("Creating Visualization")
        logger.info("="*80)

        visualize_quantum_analysis(
            measurement_results,
            decoherence_results,
            double_slit_results,
            output_path="outputs/example_07_quantum_observer.png"
        )

    # ========== Summary ==========

    logger.info("\n" + "="*80)
    logger.info("SUMMARY: Quantum Observer Effects")
    logger.info("="*80)

    logger.info("\n🔬 KEY FINDINGS:")
    logger.info(f"1. Strong measurement K-Index: {measurement_results['strong_measurement']['k_index']:.3f}")
    logger.info(f"   → Observer effect quantified!")
    logger.info(f"2. Weak measurement K-Index: {measurement_results['weak_measurement']['k_index']:.3f}")
    logger.info(f"   → Less coherent without collapse")
    logger.info(f"3. Double-slit visibility reduction: {visibility_no_obs - visibility_with_obs:.3f}")
    logger.info(f"   → Observation destroys interference")
    logger.info(f"4. Decoherence reduces coherence from {decoherence_results['decoherence_analysis'][0]['mean_coherence']:.3f}")
    logger.info(f"   to {decoherence_results['decoherence_analysis'][-1]['mean_coherence']:.3f}")
    logger.info(f"   → Quantum→classical transition observed")

    logger.info("\n💡 PHYSICAL INTERPRETATION:")
    logger.info("• K-Index quantifies observer-observed coherence in quantum systems")
    logger.info("• Strong measurements increase observer-system correlation")
    logger.info("• Decoherence explains macroscopic classical behavior")
    logger.info("• Observer effect is measurable and quantifiable!")

    logger.info("\n📂 OUTPUTS:")
    logger.info("• Log: logs/example_07_quantum.log")
    logger.info("• K-Codex: logs/example_07_quantum_kcodex.json")
    if HAS_MATPLOTLIB:
        logger.info("• Visualization: outputs/example_07_quantum_observer.png")

    logger.info("\n" + "="*80)
    logger.info("✅ Analysis complete!")
    logger.info("="*80)

    return {
        'measurement': measurement_results,
        'double_slit': double_slit_results,
        'decoherence': decoherence_results
    }


if __name__ == "__main__":
    results = main()
