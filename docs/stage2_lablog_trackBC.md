# Stage 2 Lab Log — Track B/C SAC Bioelectric Rescue

## Run Metadata
- **Date (UTC)**: 2025-11-03T21:05:32Z
- **Commit**: a91dbc53
- **Config**: `configs/track_c_sac.yaml`
- **Training script**: `fre/sac_train.py`
- **Evaluation script**: `fre/sac_evaluate.py`
- **Artifact archive**: `artifacts/fre_phase2/best_model.zip`
- **Training log**: `sac_train.log`

## Environment & Seeds
- Python: `Python 3.13.7`
- Gymnasium SAC environment: `fre.sac_env.KosmicRescueEnv`
- Seeds: default (`poetry run python fre/sac_train.py` with no `--seed` flag → deterministic RNG init at 0).
- Hardware: A100-class GPU (Codex harness default), run launched via `nohup` for 200 k timesteps.

## Training Summary
- **Total timesteps**: 200 000
- **Actor loss** (final): ≈ −0.058
- **Critic loss** (final): ≈ 6×10⁻⁷
- **Entropy coefficient** (final): ≈ 3×10⁻⁵
- **Mean reward (eval @200k)**: 0.344 ± 0.00
- **Mean episode length (eval @200k)**: 74.2 ± 0.75

Source files for deeper analysis:
- Evaluation history (`artifacts/sac_run_01/evaluations.npz`)
- Full training log with periodic eval blocks (`sac_train.log`)

## Evaluation Results (best_model.zip)
Executed via:

```bash
poetry run python fre/sac_evaluate.py \
    --config configs/track_c_sac.yaml \
    --model artifacts/fre_phase2/best_model.zip
```

| Episode | Reward | Steps | Final IoU |
|---------|--------|-------|-----------|
| 1 | 0.351 | 74 | 0.997 |
| 2 | 0.353 | 74 | 1.000 |
| 3 | 0.354 | 76 | 0.994 |
| 4 | 0.343 | 74 | 0.991 |
| 5 | 0.345 | 74 | 0.991 |

**All episodes achieved IoU ≥ 0.99**, comfortably surpassing the Track C success gate (IoU ≥ 0.85).

### Robustness Stress Test
- Command: 

```bash
poetry run python fre/sac_evaluate.py \
    --config configs/track_c_sac.yaml \
    --model artifacts/fre_phase2/best_model.zip \
    --episodes 5 \
    --lesion-scale 1.8 \
    --lesion-count 2
```

| Episode | Reward | Steps | Final IoU |
|---------|--------|-------|-----------|
| 1 | 0.832 | 69 | 0.994 |
| 2 | 0.974 | 16 | 0.997 |
| 3 | 0.795 | 77 | 0.994 |
| 4 | 0.646 | 74 | 0.991 |
| 5 | 0.707 | 73 | 0.994 |

Even under dual lesions scaled to 1.8× the baseline radius, the controller maintained IoU ≥ 0.991. Episode 2 terminated early (16 steps) because stimulation fully restored morphology, indicating fast recovery behaviour under severe damage.

## Files & Artifacts Moved
- `artifacts/fre_phase2/best_model.zip` ← saved policy (moved from `artifacts/sac_run_01/`).
- `artifacts/sac_run_01/sac_track_c.zip` ← full SB3 checkpoint (kept in-place).

## Notes & Next Actions
- Reward and entropy coefficient curves saved to `figs/trackBC_success_curves/reward_curve.png` and `.../entropy_curve.png` (generated via `scripts/plot_sac_training.py`).
- Extend evaluation harness to apply lesion variations (e.g., larger radius, multiple lesions) to assess robustness. *(Completed; see robustness stress test above.)*
- Incorporate policy checkpoints into the forthcoming Reciprocal Policy Coupling prototype for cross-universe co-learning experiments.
