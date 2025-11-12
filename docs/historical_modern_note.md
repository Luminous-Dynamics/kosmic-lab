# Historical K(t) – Modern Era Snapshot (1890–2010)

**Data sources:** OWID energy proxies (`build_from_owid.py`), World Bank indicators (`build_from_worldbank.py`).  
**Pipeline run:** `poetry run python historical_k/compute_k.py --config historical_k/k_config.yaml`  
**Analysis script:** `poetry run python historical_k/analysis_modern.py`

## Hypothesis Checks

| Hypothesis | Metric | Result |
|------------|--------|--------|
| H1: Reciprocity predicts higher K | corr(rec, K) | **0.98** |
| H2: Innovation/creativity correlates with K | corr(play_entropy, K) | **0.89** |
| H3: Crisis periods show K dips | K at 1910, 1930, 1940 | 0.188, 0.192, 0.195 |
| Post-shock recovery | mean K pre-1940 vs post-1940 | 0.186 → 0.306 |

## Interpretation
- Strong positive correlation between reciprocity proxies and the K-index in the modern era.
- Post-WWII recovery evident in K(t), consistent with the hypothesis that reciprocal structures aid resilience.
- Crisis years show lower K than surrounding decades (albeit modest amplitude with current proxies).

## Next Steps
- Incorporate additional datasets per `docs/historical_data_plan.md` (trade flows, aid/extraction, patents) to refine harmonies.
- Expand analysis to include regression on recovery speed (ΔK after shocks) once shock catalogue is assembled.
- Package these findings into a short “Modern Coherence” methods note for early publication.
