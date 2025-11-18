# Preregistration – Fractal Reciprocity Engine (FRE) Phase 1

**Registry:** OSF (pending submission)  
**Date:** YYYY-MM-DD  
**Team:** Kosmic Lab

## 1. Objectives

Demonstrate the existence and robustness of a Goldilocks corridor (K > 1) across three universe archetypes and quantify controller performance within this corridor.

## 2. Hypotheses

### Primary
- **H₀:** No contiguous parameter region yields K > 1 with stability across seeds.  
- **H₁:** At least one universe exhibits a corridor where mean K > 1 and K-variance < 0.1 across ≥30 seeds.

### Secondary
1. Controllers (SAC) maintain time-above-threshold (TAT) ≥ 0.75 and halve recovery time relative to open-loop baselines.
2. Corridor geometry remains invariant under >3D perturbations (Jaccard ≥ 0.70, centroid shift ≤ 0.12).

## 3. Design

- **Universes:**  
  - U₁: Local lattice with short-range coupling  
  - U₂: Sparse long-range (small-world)  
  - U₃: Modular clusters with stochastic bridges
- **Parameter space:** energy, communication_cost, plasticity (primary); noise_spectrum_alpha, stimulus_jitter (robustness)
- **Sampling:** Latin Hypercube (N=60) + Sobol validation (N=30) per universe
- **Seeds:** 30 per configuration (power 0.95 for d ≥ 0.8, α=0.01)

## 4. Metrics

- **K-index**: weighted harmonies via `fre/configs/k_config.yaml`
- **Stability**: σ_K < 0.1 within corridor
- **Controller stats**: TAT, Recovery, AUC(K)
- **Reciprocity**: TE_macro→micro / TE_micro→macro, TE symmetry (1 - JS divergence)

## 5. Analysis Plan

1. Identify candidate corridor via mean K > 1 and σ_K < 0.1.  
2. Compute Jaccard overlap and centroid shift between baseline and >3D sweeps.  
3. For controllers, compare against open-loop using t-tests (α=0.01) and report Cohen’s d.  
4. Publish null findings; no HARKing.  
5. All runs logged via `schemas/k_passport.json`.

## 6. Stopping & Deviations

- Predefined run counts; no early stopping unless hardware failure.  
- Deviations (e.g., hyperparameter tuning) require documented amendment before analysis.

## 7. Data Sharing

- Configs, seeds, and metrics in repo.  
- Large artefacts (e.g., TE matrices) shared via OSF storage.
