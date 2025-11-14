"""
Bioelectric rescue utilities for Track C.

These helpers implement the trigger, regeneration, and IoU metric
described in docs/track_c_rescue_spec.md. They operate on the KosmicAgent
structure defined in integration_spec.md.

VERSION HISTORY:
v1 (Initial): Pushed voltage away from target (bug)
v2 (Fixed): Pulled voltage toward target with stronger correction + momentum
v3 (Attractor-based): Modifies grid physics to create stable attractors

ARCHITECTURAL EVOLUTION (2025-11-09):
v1 → v2: Voltage scale alignment + stronger correction
v2 → v3: Discovered v2 interferes with natural dynamics (rescue worse than baseline!)
         Redesigned to work WITH natural equilibria instead of against them
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from core.bioelectric import compute_iou

if TYPE_CHECKING:
    from core.bioelectric import BioelectricGrid


def fep_to_bioelectric(agent, timestep: int) -> None:
    """Trigger bioelectric response when sensory prediction error is high.

    ARCHITECTURAL FIX V2 (2025-11-09):
    1. Pull voltage TOWARD target (-70mV), not away from it
    2. Stronger correction factor (0.5 instead of 0.15)
    3. Momentum accumulation for sustained hyperpolarization

    This ensures rescue can overcome grid diffusion dynamics.
    """
    error = agent.prediction_errors.get("sensory", 0.0)
    if error <= 0.5:
        return

    # Initialize momentum if first time
    if not hasattr(agent, "_voltage_momentum"):
        agent._voltage_momentum = 0.0

    # Pull voltage toward resting potential (-70mV) with stronger correction
    target_voltage = -70.0
    # Increased from 0.15 to 0.5 for stronger effect
    correction = (target_voltage - agent.voltage) * error * 0.5

    # Add momentum: 90% previous momentum + 10% new correction
    # This accumulates corrections over time instead of single-step changes
    agent._voltage_momentum = 0.9 * agent._voltage_momentum + 0.1 * correction
    agent.voltage += agent._voltage_momentum

    # Clamp to reasonable range (allow hyperpolarization)
    agent.voltage = np.clip(agent.voltage, -100.0, -10.0)

    for neighbor_id in list(agent.gap_junctions.keys()):
        agent.gap_junctions[neighbor_id] *= 1.1


def fep_to_bioelectric_v3(agent, grid: "BioelectricGrid", timestep: int) -> None:
    """V3: Attractor-based rescue - modifies grid physics to create stable equilibria.

    KEY INSIGHT FROM V2 FAILURE:
    - v2 forced voltage toward target (transient perturbation)
    - Natural dynamics then "corrected" this, making outcome worse than baseline
    - v3 creates STABLE ATTRACTOR at target by modifying leak reversal potential

    MECHANISM:
    - Leak term: g * (V - leak_reversal)
    - Default: leak_reversal = 0.0 mV (pulls toward depolarization)
    - Rescue: leak_reversal = -70.0 mV (pulls toward hyperpolarization)
    - This makes -70mV a STABLE equilibrium, not a forced state

    Additionally increases leak conductance to accelerate convergence when error high.
    """
    error = agent.prediction_errors.get("sensory", 0.0)
    if error <= 0.5:
        # Low error - restore natural dynamics
        grid.leak_reversal = 0.0
        return

    # High error - activate attractor-based rescue
    target_voltage = -70.0

    # Modify leak reversal to create stable attractor at target
    # Gradually shift leak reversal toward target (smooth transition)
    if not hasattr(agent, "_leak_reversal_momentum"):
        agent._leak_reversal_momentum = 0.0

    # Optimal shift rate from v3 (validation showed 0.6 is worse)
    target_shift = (target_voltage - grid.leak_reversal) * error * 0.3
    agent._leak_reversal_momentum = 0.8 * agent._leak_reversal_momentum + 0.2 * target_shift
    grid.leak_reversal += agent._leak_reversal_momentum
    grid.leak_reversal = np.clip(grid.leak_reversal, -70.0, 0.0)

    # Also temporarily increase leak conductance to accelerate convergence
    # Store original if first time
    if not hasattr(grid, "_original_g"):
        grid._original_g = grid.g

    # Increase leak proportional to error (higher error = faster convergence)
    grid.g = grid._original_g * (1.0 + error * 1.5)

    # Update agent voltage from grid for monitoring
    agent.voltage = float(np.mean(grid.V))

    for neighbor_id in list(agent.gap_junctions.keys()):
        agent.gap_junctions[neighbor_id] *= 1.1


def fep_to_bioelectric_v4_adaptive(agent, grid: "BioelectricGrid", timestep: int) -> None:
    """V4: Adaptive target voltage - matches intervention strength to damage severity.

    INSIGHT FROM VALIDATION TEST:
    - Fixed V_target = -70mV is one-size-fits-all
    - Severe damage might need stronger hyperpolarization
    - Mild damage might benefit from gentler nudge
    - Adaptive approach should improve success rate beyond v3's 20%

    ADAPTIVE MECHANISM:
    - Error > 0.7: V_target = -90mV (strong hyperpolarization for severe damage)
    - Error > 0.5: V_target = -70mV (standard rescue for moderate damage)
    - Error > 0.3: V_target = -40mV (gentle nudge for near-threshold)
    - Error ≤ 0.3: Restore natural dynamics (already close to target)

    This allows the rescue mechanism to adapt intervention strength to morphology state.
    """
    error = agent.prediction_errors.get("sensory", 0.0)

    # Adaptive target voltage based on error magnitude
    if error <= 0.3:
        # Very low error - restore natural dynamics (already close)
        grid.leak_reversal = 0.0
        return
    elif error <= 0.5:
        # Low-medium error - gentle nudge
        target_voltage = -40.0
    elif error <= 0.7:
        # Medium-high error - standard rescue
        target_voltage = -70.0
    else:
        # High error (>0.7) - strong hyperpolarization for severe damage
        target_voltage = -90.0

    # Modify leak reversal to create stable attractor at adaptive target
    # Gradually shift leak reversal toward target (smooth transition)
    if not hasattr(agent, "_leak_reversal_momentum"):
        agent._leak_reversal_momentum = 0.0

    # Use v3's optimal shift rate (validation showed 0.3 is best)
    target_shift = (target_voltage - grid.leak_reversal) * error * 0.3
    agent._leak_reversal_momentum = 0.8 * agent._leak_reversal_momentum + 0.2 * target_shift
    grid.leak_reversal += agent._leak_reversal_momentum
    grid.leak_reversal = np.clip(grid.leak_reversal, -100.0, 0.0)

    # Also temporarily increase leak conductance to accelerate convergence
    # Store original if first time
    if not hasattr(grid, "_original_g"):
        grid._original_g = grid.g

    # Increase leak proportional to error (higher error = faster convergence)
    grid.g = grid._original_g * (1.0 + error * 1.5)

    # Update agent voltage from grid for monitoring
    agent.voltage = float(np.mean(grid.V))

    for neighbor_id in list(agent.gap_junctions.keys()):
        agent.gap_junctions[neighbor_id] *= 1.1


def bioelectric_to_autopoiesis(agent, target_morphology: Dict[str, float]) -> None:
    """Regenerate membrane/boundary once voltage matches the target pattern.

    ARCHITECTURAL FIX: Relaxed voltage threshold (20mV instead of 5mV)
    to allow autopoiesis activation during rescue process.
    """
    target_voltage = target_morphology.get("voltage", -70.0)

    # Relaxed threshold: Allow activation when voltage is moving toward target
    if abs(agent.voltage - target_voltage) >= 20.0:
        return
    if agent.boundary_integrity >= 1.0:
        return

    repair = 0.01 * (1.0 - agent.boundary_integrity)
    agent.internal_state["membrane"] = agent.internal_state.get("membrane", 0.0) + repair
    agent.boundary_integrity += repair * 0.5
    agent.internal_state["ATP"] = agent.internal_state.get("ATP", 0.0) - repair * 0.1
