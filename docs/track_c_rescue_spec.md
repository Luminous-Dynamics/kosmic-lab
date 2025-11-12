# Track C – Bioelectric Rescue Specification

## Trigger: FEP → Bioelectric
- Function: `fep_to_bioelectric(agent: KosmicAgent, timestep: int)`.
- Trigger condition: `agent.prediction_errors['sensory'] > 0.5`.
- Updates:
  ```python
  error = agent.prediction_errors['sensory']
  agent.voltage = min(agent.voltage + error * 10.0, -20.0)
  for neighbor_id in agent.gap_junctions:
      agent.gap_junctions[neighbor_id] *= 1.1
  ```

## Regeneration: Bioelectric → Autopoiesis
- Function: `bioelectric_to_autopoiesis(agent: KosmicAgent, target_morphology: Dict)`.
- Run when voltage near target (`abs(agent.voltage - target['voltage']) < 5`) and `boundary_integrity < 1.0`.
- Updates:
  ```python
  repair = 0.01 * (1.0 - agent.boundary_integrity)
  agent.internal_state['membrane'] += repair
  agent.boundary_integrity += repair * 0.5
  agent.internal_state['ATP'] -= repair * 0.1
  ```

## IoU Metric
- Function: `compute_iou(current: np.ndarray, target: np.ndarray) -> float`.
- 3D boolean arrays; `iou = intersection / union`, returning 1.0 if union empty.

## Validation Scenario
1. Mock agent: `boundary_integrity=0.3`, `voltage=-70`, `sensory error=0.8`, `ATP=1.0`.
2. Call `fep_to_bioelectric` → expect voltage `-62`.
3. Set `voltage=-70`, call regeneration → expect:
   - `boundary_integrity ≈ 0.3035`
   - `ATP ≈ 0.9993`
4. IoU: simulate 30 overlapping voxels out of 100 → IoU = 0.3 (rises toward 1 as regeneration proceeds).

## Integration Notes
- Add functions to `fre/rescue.py` with unit tests.
- Tie into simulator loop (trigger after high surprise, regenerate once stabilized).
- Gate success using IoU ≥ 0.85 in Track C runner.
