"""
Detailed Logging Patch for Paper 2 ℂ Functional Analysis

Adds comprehensive state logging to Track B and Track C experiments
to enable computation of consciousness functional components.

Requirements for ℂ Functional:
- Full voltage trajectories: (T, n_cells)
- Internal vs external cell boundaries
- Blanket states (actions)
- Observations (boundary signals)

Author: Kosmic Lab Team
Date: November 10, 2025
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json


# =============================================================================
# Configuration
# =============================================================================

def should_enable_detailed_logging(config: Dict[str, Any]) -> bool:
    """
    Check if detailed logging is enabled in config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if detailed logging should be enabled
    """
    return config.get("enable_detailed_logging", False)


# =============================================================================
# Track B: SAC Controller Enhanced Logging
# =============================================================================

def log_detailed_track_b_step(
    step_record: Dict[str, Any],
    metrics: Dict[str, float],
    current_params: Dict[str, float],
    raw_action: np.ndarray,
) -> Dict[str, Any]:
    """
    Enhance Track B step record with full state information.
    
    Track B uses "Harmony" components (H1-H7) as state space:
    - H1: Coherence
    - H2: Flourishing  
    - H3: Wisdom
    - H4: Play
    - H5: Interconnectedness
    - H6: Reciprocity
    - H7: Progression
    + TE_macro_micro, TE_symmetry
    
    Args:
        step_record: Basic step record from runner
        metrics: Full metrics dict containing H1-H7 + TEs
        current_params: Current control parameters
        raw_action: SAC policy output (3D action)
        
    Returns:
        Enhanced step record with detailed states
    """
    # Extract harmony components as "voltage states"
    harmony_keys = [
        "H1_Coherence",
        "H2_Flourishing", 
        "H3_Wisdom",
        "H4_Play",
        "H5_Interconnectedness",
        "H6_Reciprocity",
        "H7_Progression",
    ]
    
    # Build full state vector
    states = []
    for key in harmony_keys:
        states.append(float(metrics.get(key, 0.0)))
    
    # Add transfer entropy metrics
    states.append(float(metrics.get("TE_macro_micro", metrics.get("te_mutual", 0.0))))
    states.append(float(metrics.get("TE_symmetry", metrics.get("te_symmetry", 0.0))))
    
    # Add current parameters
    for name in sorted(current_params.keys()):
        states.append(float(current_params[name]))
    
    states_array = np.array(states, dtype=np.float32)
    
    # Define internal vs external boundaries
    # For Track B: H1-H3 are "internal" (coherence, self), H4-H7 are "external" (world)
    n_harmony = len(harmony_keys)
    internal_cells = list(range(3))  # H1, H2, H3
    external_cells = list(range(3, n_harmony))  # H4-H7
    
    # Add detailed fields
    step_record["voltage_states"] = states_array.tolist()
    step_record["actions"] = raw_action.tolist()
    step_record["observations"] = states_array.tolist()  # Use same as states
    step_record["internal_cells"] = internal_cells
    step_record["external_cells"] = external_cells
    
    return step_record


def save_track_b_detailed_npz(
    episode_records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save Track B episode to NPZ format for efficient array storage.
    
    Args:
        episode_records: List of step records from one episode
        output_path: Path to save NPZ file (e.g., track_b_ep001.npz)
    """
    # Extract arrays
    states = np.array([rec["voltage_states"] for rec in episode_records], dtype=np.float32)
    actions = np.array([rec["actions"] for rec in episode_records], dtype=np.float32)
    K_values = np.array([rec["K"] for rec in episode_records], dtype=np.float32)
    
    # Extract metadata from first record
    metadata = {
        "episode_mode": episode_records[0]["episode_mode"],
        "seed": episode_records[0]["seed"],
        "internal_cells": episode_records[0]["internal_cells"],
        "external_cells": episode_records[0]["external_cells"],
    }
    
    # Save to NPZ
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        actions=actions,
        observations=states,  # Same as states for Track B
        K_values=K_values,
        metadata=json.dumps(metadata),
    )


# =============================================================================
# Track C: Bioelectric Rescue Enhanced Logging
# =============================================================================

def log_detailed_track_c_step(
    diag_record: Dict[str, Any],
    grid_voltage: np.ndarray,
    target_mask: np.ndarray,
    rescue_action: float = 0.0,
) -> Dict[str, Any]:
    """
    Enhance Track C diagnostic record with full grid information.
    
    Track C uses full bioelectric grid voltage (H x W array).
    
    Args:
        diag_record: Basic diagnostic record from runner
        grid_voltage: Full voltage grid (H, W) in mV
        target_mask: Boolean mask (H, W) indicating target morphology
        rescue_action: Rescue intervention strength (leak_reversal change)
        
    Returns:
        Enhanced diagnostic record with detailed states
    """
    # Flatten grid for storage
    voltage_flat = grid_voltage.flatten().astype(np.float32)
    
    # Identify internal vs external cells
    internal_mask = target_mask.flatten()
    internal_cells = np.where(internal_mask)[0].tolist()
    external_cells = np.where(~internal_mask)[0].tolist()
    
    # Add detailed fields
    diag_record["voltage_states"] = voltage_flat.tolist()
    diag_record["actions"] = [float(rescue_action)]
    diag_record["observations"] = voltage_flat.tolist()  # Use same as states
    diag_record["internal_cells"] = internal_cells
    diag_record["external_cells"] = external_cells
    diag_record["grid_shape"] = list(grid_voltage.shape)
    
    return diag_record


def save_track_c_detailed_npz(
    episode_records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save Track C episode to NPZ format for efficient array storage.
    
    Args:
        episode_records: List of diagnostic records from one episode
        output_path: Path to save NPZ file (e.g., track_c_ep001.npz)
    """
    # Extract arrays
    states = np.array([rec["voltage_states"] for rec in episode_records], dtype=np.float32)
    actions = np.array([rec["actions"] for rec in episode_records], dtype=np.float32)
    K_values = np.array([rec["iou"] for rec in episode_records], dtype=np.float32)  # K = IoU
    
    # Extract metadata from first record
    metadata = {
        "mode": episode_records[0]["mode"],
        "seed": episode_records[0]["seed"],
        "internal_cells": episode_records[0]["internal_cells"],
        "external_cells": episode_records[0]["external_cells"],
        "grid_shape": episode_records[0]["grid_shape"],
    }
    
    # Save to NPZ
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        actions=actions,
        observations=states,  # Same as states for Track C
        K_values=K_values,
        metadata=json.dumps(metadata),
    )


# =============================================================================
# Integration Instructions
# =============================================================================

INTEGRATION_INSTRUCTIONS = """
# Integration Instructions for Detailed Logging

## Track B (fre/track_b_runner.py)

### Step 1: Import patch module
Add at top of file:
```python
from fre.detailed_logging_patch import (
    should_enable_detailed_logging,
    log_detailed_track_b_step,
    save_track_b_detailed_npz,
)
```

### Step 2: Check config in __init__
Add after line 48:
```python
self.enable_detailed = should_enable_detailed_logging(config)
self.detailed_records = []  # Storage for detailed logs
```

### Step 3: Enhance step recording
Modify _log_diagnostics() around line 323:
```python
def _log_diagnostics(self, rows: List[Dict[str, Any]]) -> None:
    self.diagnostics.extend(rows)
    
    # Add detailed logging if enabled
    if self.enable_detailed:
        for row in rows:
            # Enhance with full state info
            enhanced = log_detailed_track_b_step(
                step_record=row,
                metrics=self.last_metrics,  # Store this in self
                current_params=self.current_params,  # Store this in self
                raw_action=self.last_action,  # Store this in self
            )
            self.detailed_records.append(enhanced)
```

### Step 4: Save NPZ at episode end
Add in run() method after episode completion (around line 340):
```python
if self.enable_detailed and self.detailed_records:
    npz_path = self.results_dir / f"track_b_ep{episode_num:03d}.npz"
    save_track_b_detailed_npz(self.detailed_records, npz_path)
    self.detailed_records.clear()
```

## Track C (fre/track_c_runner.py)

### Step 1: Import patch module
Add at top of file:
```python
from fre.detailed_logging_patch import (
    should_enable_detailed_logging,
    log_detailed_track_c_step,
    save_track_c_detailed_npz,
)
```

### Step 2: Check config in __init__
Add after line 50:
```python
self.enable_detailed = should_enable_detailed_logging(config)
self.detailed_records = []  # Storage for detailed logs
```

### Step 3: Enhance diagnostic recording
Modify run_episode() around line 191:
```python
# Basic diagnostic (keep existing CSV)
self.diagnostics.append({...})

# Add detailed logging if enabled
if self.enable_detailed:
    enhanced = log_detailed_track_c_step(
        diag_record=self.diagnostics[-1].copy(),
        grid_voltage=self.grid.V,
        target_mask=self.target_mask,
        rescue_action=rescue_strength if rescue_triggered else 0.0,
    )
    self.detailed_records.append(enhanced)
```

### Step 4: Save NPZ at episode end
Add at end of run_episode() (around line 230):
```python
if self.enable_detailed and self.detailed_records:
    npz_path = self.results_dir / f"track_c_{mode}_s{seed:04d}.npz"
    save_track_c_detailed_npz(self.detailed_records, npz_path)
    self.detailed_records.clear()
```

## Configuration Files

Add to all config YAMLs:
```yaml
# Enable detailed logging for Paper 2 analysis
enable_detailed_logging: true
```

Files to update:
- configs/track_b_sac.yaml
- configs/track_b_baseline.yaml
- configs/track_c_v1.yaml
- configs/track_c_v3.yaml
- configs/track_c_baseline.yaml

## Running with Detailed Logging

```bash
# Run experiments (will generate both CSV and NPZ)
python -m fre.run configs/track_b_sac.yaml
python -m fre.run configs/track_c_v1.yaml

# Output will include:
# - logs/fre_track_b_diagnostics.csv (summary, as before)
# - logs/track_b_ep001.npz, track_b_ep002.npz, ... (detailed states)
# - logs/track_c/fre_track_c_diagnostics.csv (summary, as before)
# - logs/track_c/track_c_v1_s1000.npz, ... (detailed states)
```

## Using Detailed Logs in Paper 2

Update `paper2_analyses/extract_episode_data.py` to load NPZ files:

```python
def load_detailed_npz(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path)
    metadata = json.loads(str(data["metadata"]))
    
    return {
        "states": data["states"],
        "actions": data["actions"],
        "observations": data["observations"],
        "K_values": data["K_values"],
        **metadata,
    }

def extract_track_b_episodes_from_npz(logs_dir: Path) -> List[Dict]:
    episodes = []
    for npz_file in sorted(logs_dir.glob("track_b_ep*.npz")):
        ep = load_detailed_npz(npz_file)
        ep["id"] = npz_file.stem
        episodes.append(ep)
    return episodes
```
"""

def print_integration_instructions():
    """Print integration instructions to console."""
    print(INTEGRATION_INSTRUCTIONS)


if __name__ == "__main__":
    print("=" * 80)
    print("Detailed Logging Patch for Paper 2")
    print("=" * 80)
    print()
    print_integration_instructions()
