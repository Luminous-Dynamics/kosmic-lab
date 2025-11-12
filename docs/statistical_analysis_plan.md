# Statistical Analysis Plan (SAP) v1.0
## Kosmic Simulation Suite: Empirical Testing of Recursive Meta-Intelligence

**Document Status:** LOCKED (Preregistered on OSF)  
**Date:** [To be filled before data collection]  
**Principal Investigator:** [Name]  
**Statistician:** [Name]  
**SAP Version:** 1.0

---

## 1. Overview

This Statistical Analysis Plan (SAP) provides the complete specification of all statistical procedures, decision rules, and interpretation frameworks for the Kosmic Simulation Suite experiments. This document is **locked before any data collection** to prevent p-hacking, HARKing (Hypothesizing After Results are Known), and selective reporting.

**Key Principles:**
- All tests prespecified with exact α levels
- Effect size thresholds justified *a priori*
- Multiple comparisons corrections applied uniformly
- Negative results reported with equal prominence
- Exploratory analyses clearly labeled

---

## 2. Data Structure & Variables

### 2.1 Experimental Design

| Module | Design Type | Independent Variables | Dependent Variables | Replications |
|--------|-------------|----------------------|---------------------|--------------|
| Autopoiesis | Factorial 3×3 | Temp gradient, Flow rate | Persistence time | 30 seeds |
| IIT | Between-subjects | Network type (evolved/random/broadcast) | Φ, Robustness | 30 each |
| FEP | Within-subjects | Cooperation (on/off) | Free energy, TE | 30 agents |
| Bioelectric | Between-subjects | Intervention type | IoU, SSIM | 30 trials |
| Multiscale | Parameter sweep | (Temp, Flow, Noise) | K-index, Harmonies | 30 seeds |

### 2.2 Variable Definitions

#### Primary Outcome: Kosmic Signature Index (K)

**Formula:**
```
K = (1/7) * Σ H_i
```

where:
- H₁ = (Φ - Φ₀) / Φ₀ (Resonant Coherence)
- H₂ = H(agent_types | alive) / H_max (Pan-Sentient Flourishing)
- H₃ = 1 - (mean_error / FE₀) (Integral Wisdom)
- H₄ = H(actions) / log(n_bins) (Infinite Play)
- H₅ = TE_mutual / TE_total (Universal Interconnectedness)
- H₆ = 1 - JS(TE_out, TE_in) (Sacred Reciprocity)
- H₇ = tanh(∂Φ/∂t / σ_Φ) (Evolutionary Progression)

**Data Type:** Continuous  
**Expected Range:** [0, 2]  
**Corridor Threshold:** K > 1.0  
**Stability Threshold:** σ_K < 0.1

#### Secondary Outcomes

1. **Persistence Time** (Autopoiesis)
   - Type: Continuous, right-skewed
   - Unit: Simulation timesteps
   - Transform: log₁₀ for parametric tests

2. **Integrated Information (Φ)**
   - Type: Continuous, bounded [0, ∞)
   - Unit: Bits
   - Normality: Check with Shapiro-Wilk

3. **Free Energy (FE)**
   - Type: Continuous
   - Unit: Nats (natural log scale)
   - Expected decrease with cooperation

4. **Regeneration Fidelity**
   - IoU: Continuous [0, 1]
   - SSIM: Continuous [-1, 1], expected positive
   - Ordinal interpretation: Poor < 0.5, Fair 0.5-0.75, Good > 0.75

5. **Transfer Entropy (TE)**
   - Type: N×N matrix, continuous
   - Unit: Bits
   - Symmetry: Compare TE_ij vs TE_ji

---

## 3. Sample Size Justification

### 3.1 Power Analysis

**Target Parameters:**
- Statistical power (1-β): 0.95
- Significance level (α): 0.01 (two-tailed)
- Minimum detectable effect size: Cohen's d = 0.8

**Calculation (G*Power 3.1):**
```
For independent samples t-test:
d = 0.8, α = 0.01, power = 0.95
→ Required n per group = 27

Selected n = 30 (provides buffer for dropouts/crashes)
```

### 3.2 Sample Sizes by Module

| Module | Conditions | n per condition | Total runs |
|--------|-----------|-----------------|------------|
| Autopoiesis | 9 (3×3 grid) | 30 | 270 |
| IIT | 3 (evolved/random/broadcast) | 30 | 90 |
| FEP | 2 (with/without coop) | 30 | 60 |
| Bioelectric | 3 (damage types) | 30 | 90 |
| Multiscale | 30 (param combinations) | 30 | 900 |
| **Total** | | | **1410** |

**Computational Budget Check:**
- Estimated time: 327.5 hours (from roadmap)
- Cost: $400-800 (cloud compute)
- Status: ✓ Feasible

---

## 4. Statistical Tests by Hypothesis

### 4.1 Module 1: Autopoiesis

**H1.0:** No parameter combination yields persistence time significantly > random

**H1.1:** Goldilocks region exists with persistence ≥ 5× edge conditions

**Test:** Independent samples t-test (two-tailed)
- Groups: Top 10% vs Bottom 10% by persistence time
- α = 0.01
- Assumption checks:
  - Normality: Shapiro-Wilk on log(persistence)
  - Equal variance: Levene's test
  - If violated: Welch's t-test

**Effect Size:**
```
Cohen's d = (M_top - M_bottom) / SD_pooled
Required: d ≥ 0.8
```

**Decision Rule:**
- If p < 0.01 AND d ≥ 0.8: **Reject H1.0** → Corridor exists
- Otherwise: **Fail to reject** → No corridor

**Secondary Analysis:**
- ANOVA to test parameter main effects and interactions
- Bonferroni correction: α_family = 0.01 → α_individual = 0.01/3 = 0.0033
- Tukey HSD for post-hoc pairwise comparisons

**Exploratory:**
- Response surface methodology to map persistence time landscape
- Contour plot of parameter space

---

### 4.2 Module 2: IIT

**H2.0:** Mean Φ_evolved = Φ_random

**H2.1:** Φ_evolved ≥ 1.25 × Φ_random

**Test:** Welch's t-test (unequal variances expected)
- Groups: Evolved networks vs Random graphs
- α = 0.01
- One-tailed (directional hypothesis)

**Effect Size:**
```
Cohen's d = (M_evolved - M_random) / SD_pooled
Required: d ≥ 0.6
```

**Additional Tests:**
1. **Robustness Analysis**
   - Measure Φ after removing 10%, 20%, 30% of nodes
   - Repeated measures ANOVA
   - Within-subjects factor: Damage level
   - Between-subjects factor: Network type

2. **Adversarial Comparison**
   - Compare evolved vs broadcast networks
   - Hypothesis: Φ_broadcast < Φ_evolved
   - Mann-Whitney U (if non-normal)

**Decision Rule:**
- If p < 0.01 AND d ≥ 0.6 AND evolved > broadcast: **GO**
- Otherwise: **NO-GO**

**Preregistered Correlation:**
- Φ vs robustness: Expect r > 0.5
- Pearson or Spearman (check normality)

---

### 4.3 Module 3: FEP

**H3.0:** FE_cooperative = FE_isolated

**H3.1:** FE_cooperative ≤ 0.85 × FE_isolated

**Test:** Paired samples t-test (within-subjects)
- Each agent measured in both conditions
- α = 0.01, two-tailed
- Check sphericity assumption (Mauchly's test)

**Effect Size:**
```
Cohen's d_z = M_diff / SD_diff
Required: d_z ≥ 0.5
```

**Causal Validation:**
- Transfer Entropy analysis
- Hypothesis: TE_cooperative > TE_isolated
- Permutation test (1000 iterations)
- Reject if p < 0.01

**Group Size Analysis:**
- One-way ANOVA: Group size (2, 4, 8) × FE reduction
- Hypothesis: Superlinear benefit
- Polynomial contrast (quadratic)

**Decision Rule:**
- If p < 0.01 AND d ≥ 0.5 AND TE confirms directionality: **GO**
- If FE reduction < 10%: **NO-GO** (effect too small)

---

### 4.4 Module 4: Bioelectric

**H4.0:** Targeted bioelectric intervention = random perturbation

**H4.1:** Targeted intervention achieves IoU ≥ 0.85

**Test:** Mann-Whitney U test (ordinal fidelity data)
- Groups: Targeted vs Random
- α = 0.01, one-tailed

**Effect Size:**
```
Cliff's δ = P(IoU_targeted > IoU_random) - 0.5
Required: δ ≥ 0.5
```

**Predictive Validation:**
- Regression: IoU ~ Φ_pre-damage
- Hypothesis: β > 0, R² > 0.36 (r = 0.6)
- Bootstrap 95% CI (1000 iterations)

**Gap Junction Knockout:**
- Compare regeneration with/without gap junctions
- Hypothesis: IoU_blocked < 0.50 (failure)
- One-sample t-test against 0.50

**Decision Rule:**
- If U test p < 0.01 AND δ ≥ 0.5 AND predictive r > 0.6: **GO**
- If IoU < 0.75: **NO-GO** (insufficient regeneration)

---

### 4.5 Module 5: Multiscale Integration

**H5.0:** No contiguous corridor where K > 1.0

**H5.1:** Corridor exists, spatially contiguous, stable

**Test:** Cluster analysis + permutation test
- Method: DBSCAN on K-scores in 3D parameter space
- Null: Randomly permute K-values, recompute clusters
- α = 0.01

**Contiguity Metric:**
```
Contiguity = (# adjacent voxels in corridor) / (# total corridor voxels)
Required: > 0.7
```

**Stability Test:**
- For each voxel in corridor: Bootstrap 95% CI of K
- Reject if >30% of CIs include 1.0

**Multivariate Analysis:**
- PCA on Seven Harmonies
- Hypothesis: ≥ 4 independent factors (PC eigenvalues > 1)
- Test: Parallel analysis (Horn, 1965)

**Decision Rule:**
- If DBSCAN finds cluster AND p < 0.01 AND contiguity > 0.7: **GO**
- If corridor scattered: **NO-GO** (not a true corridor)

---

### 4.6 Meta-Hypothesis: Harmony Independence

**H6.0:** All harmonies correlated r > 0.9 (redundant)

**H6.1:** ≥ 4 harmonies with pairwise r < 0.5

**Test:** Correlation matrix + factor analysis
- Method: Pearson correlation on 1000 parameter samples
- Multiple comparisons: Bonferroni correction (21 pairs → α = 0.01/21 = 0.00048)

**Factor Analysis:**
- Method: Maximum likelihood
- Rotation: Varimax (orthogonal)
- Criterion: Eigenvalue > 1
- Loadings: |loading| > 0.4 considered meaningful

**Decision Rule:**
- If ≥ 4 pairs show r < 0.5 (corrected p < 0.00048): **Reject H6.0**
- If factor analysis yields ≥ 4 factors: **Support independence**

---

### 4.7 Adversarial Hypothesis: Anti-Kosmic

**H7.0:** K_anti-Kosmic can reach > 1.0

**H7.1:** max(K_anti) < 0.75 across all adversarial configs

**Test:** Independent samples t-test
- Groups: Corridor samples vs Anti-Kosmic samples
- α = 0.01, one-tailed

**Survival Analysis:**
- Kaplan-Meier curves for perturbation survival
- Log-rank test comparing corridor vs anti-Kosmic
- Hypothesis: Median survival_corridor > 2 × survival_anti

**Decision Rule:**
- If max(K_anti) < 0.75 AND survival_anti < 50% of corridor: **Adversarial validation passed**
- If any anti-Kosmic reaches K > 1.0: **Index is non-specific** → Major revision needed

---

## 5. Multiple Comparisons Corrections

### 5.1 Family-Wise Error Rate (FWER)

**Module-Level Correction:**
- 5 primary modules → Bonferroni: α_family = 0.01
- α_per-test = 0.01 / 5 = 0.002

**Harmony-Level Correction:**
- 7 harmonies → False Discovery Rate (Benjamini-Hochberg)
- FDR q = 0.01
- Procedure: Sort p-values, compare to (i/m) × q

### 5.2 Adaptive Procedures

For unplanned post-hoc tests:
- Use Holm-Bonferroni (more powerful than standard Bonferroni)
- Explicitly label as "exploratory"

---

## 6. Missing Data & Outliers

### 6.1 Handling Simulation Failures

**Protocol:**
1. If run crashes: Re-run once with same seed
2. If crashes twice: Code as NA, log error
3. If >10% runs fail for a config: Exclude config entirely

**Imputation:** NOT ALLOWED
- Rationale: Missingness is informative (instability)

### 6.2 Outlier Detection

**Method:** Modified Z-score (robust to non-normality)
```
M_i = 0.6745 × (x_i - median(x)) / MAD(x)
Outlier if |M_i| > 3.5
```

**Treatment:**
- Winsorize to 1st/99th percentiles (NOT remove)
- Report both winsorized and raw analyses

---

## 7. Assumption Checking

For each parametric test:

### 7.1 Normality
- Test: Shapiro-Wilk (n < 50), Kolmogorov-Smirnov (n ≥ 50)
- α = 0.05 (liberal for assumption check)
- If violated: Use non-parametric alternative

### 7.2 Homogeneity of Variance
- Test: Levene's test
- α = 0.05
- If violated: Welch's adjustment

### 7.3 Independence
- Test: Durbin-Watson (for time series)
- Expected range: 1.5-2.5
- If violated: Use mixed models with autocorrelation structure

---

## 8. Effect Size Interpretation

### 8.1 Cohen's d
- Small: 0.2
- Medium: 0.5
- Large: 0.8
- **Our threshold:** ≥ 0.6 (justify research effort)

### 8.2 Correlation (r)
- Small: 0.1
- Medium: 0.3
- Large: 0.5
- **Our threshold:** ≥ 0.5 (predictive utility)

### 8.3 Cliff's δ (ordinal data)
- Small: 0.147
- Medium: 0.33
- Large: 0.474
- **Our threshold:** ≥ 0.5

---

## 9. Sensitivity Analyses

### 9.1 K-Index Weighting

**Primary:** Equal weights (w_i = 1/7)

**Sensitivity Tests:**
1. PCA-derived weights (data-driven)
2. Theory-driven (H1, H5 weighted 2×)
3. Uniform drop-one (remove each harmony sequentially)

**Report:** Corridor boundaries under all schemes

### 9.2 Threshold Variation

Test corridor detection with:
- K > 0.9, K > 1.0, K > 1.1
- σ_K < 0.05, 0.10, 0.15

**Report:** Stability of findings across thresholds

---

## 10. Visualization Standards

All figures include:
- 95% confidence intervals or standard errors
- Sample sizes
- p-values (exact if p > 0.001, else "p < 0.001")
- Effect sizes

**Corridor Visualization:**
- 3D scatter plot with color-coded K-index
- Isosurface at K = 1.0
- Contiguity highlighted

**Reporting Negative Results:**
- Same prominence as positive results
- Bayes Factors to quantify evidence for null

---

## 11. Software & Reproducibility

### 11.1 Statistical Software
- Primary: Python 3.10 (scipy.stats, statsmodels, pingouin)
- Power analysis: G*Power 3.1
- Visualization: Matplotlib, Seaborn, Plotly

### 11.2 Random Seeds
- Master seed: 42 (for reproducibility)
- Each run: seed = 42 + trial_number
- All seeds logged in `summary.yaml`

### 11.3 Version Control
- Analysis scripts in `/analysis` directory
- Git commit hash in all output files
- Exact package versions in `requirements.txt`

---

## 12. Reporting Template

### 12.1 Results Format

For each hypothesis:

```markdown
### Hypothesis [Number]: [Name]

**Test:** [Test name]
**Samples:** n1 = X, n2 = Y
**Statistic:** [Test statistic] = [value], df = [degrees of freedom]
**p-value:** [exact value or < 0.001]
**Effect size:** [Cohen's d / r / δ] = [value], 95% CI [lower, upper]
**Decision:** [Reject/Fail to reject H0]
**Interpretation:** [Plain English summary]
```

### 12.2 Figures Checklist

- [ ] All axes labeled with units
- [ ] Legend included if >1 data series
- [ ] Color-blind friendly palette
- [ ] High resolution (≥300 dpi for publication)
- [ ] Raw data available in supplementary materials

---

## 13. Deviations from SAP

If any deviation occurs:

**Required Documentation:**
1. Date of deviation
2. Reason for deviation
3. Impact assessment
4. Approval from PI and statistician

**Transparency:**
- All deviations reported in manuscript "Deviations" section
- OSF preregistration updated with amendment

---

## 14. Interpretation Guidelines

### 14.1 Inferential Caution

Avoid:
- "Proves" or "demonstrates" (use "suggests" or "supports")
- "No effect" when p > 0.05 (use "insufficient evidence")
- Causal language for correlational results

### 14.2 Bayesian Supplement

For key null results, report Bayes Factors:
- BF₁₀ < 0.33: Evidence for null
- 0.33 < BF₁₀ < 3: Inconclusive
- BF₁₀ > 3: Evidence for alternative

**Software:** JASP or BayesFactor (R)

---

## 15. Success Criteria (Go/No-Go)

### 15.1 Overall Project Success

**GO (Proceed to publication):**
- ≥ 4 of 5 modules pass their primary hypothesis tests
- Corridor identified in multiscale integration
- Adversarial validation passed
- K-index shown to be non-redundant

**PARTIAL SUCCESS:**
- 2-3 modules pass → Publish with strong limitations section
- Focus on successful modules in abstract

**NO-GO:**
- 0-1 modules pass → Major theoretical revision needed
- Publish as negative result / cautionary tale

### 15.2 Module-Specific Go/No-Go

See Section 4 for individual criteria.

---

## 16. Timeline for Analysis

| Week | Activity |
|------|----------|
| 10 | Data collection complete |
| 11 | Data cleaning & assumption checks |
| 12 | Primary hypothesis tests |
| 13 | Sensitivity analyses |
| 14 | Visualization & interpretation |
| 15 | Manuscript draft |
| 16 | Internal review |
| 17 | Submission to preprint server |

---

## 17. Authorship & Contributions

**Analysis Team:**
- Lead Analyst: [Name] - Primary statistical analysis
- Computational Lead: [Name] - Data pipeline
- PI: [Name] - Interpretation & manuscript

**Contribution Statement:**
Will follow CRediT taxonomy (Contributor Roles Taxonomy).

---

## 18. Ethical Considerations

### 18.1 Computational Ethics
- Avoid p-hacking by strict adherence to this SAP
- Report all planned analyses, regardless of outcome
- Make all code and data publicly available

### 18.2 Interpretation Ethics
- Do not overstate implications for consciousness research
- Clearly distinguish simulation from reality
- Acknowledge limitations prominently

---

## 19. Sign-Off

This SAP is locked and cannot be modified after data collection begins without formal amendment.

**Signatures:**

Principal Investigator: _____________________ Date: _______

Lead Statistician: _____________________ Date: _______

OSF Registration: _____________________ Date: _______

---

## Appendix A: Statistical Formulas

### A.1 Cohen's d (Independent Samples)
```
d = (M₁ - M₂) / √((SD₁² + SD₂²) / 2)
```

### A.2 Transfer Entropy
```
TE(X → Y) = Σ p(y_t, y_{t-1}, x_{t-1}) log₂ [p(y_t | y_{t-1}, x_{t-1}) / p(y_t | y_{t-1})]
```

### A.3 Jensen-Shannon Divergence
```
JS(P||Q) = 0.5 × [KL(P||M) + KL(Q||M)]
where M = 0.5 × (P + Q)
```

### A.4 Integrated Information (Simplified)
```
Φ = min_{partition} [H(X_full) - Σ H(X_i)]
```

---

## Appendix B: Decision Tree

```
Start
│
├─ Module Tests
│  ├─ Autopoiesis: Corridor? → [YES] → Continue | [NO] → Flag
│  ├─ IIT: Φ elevated? → [YES] → Continue | [NO] → Flag
│  ├─ FEP: Coop reduces FE? → [YES] → Continue | [NO] → Flag
│  ├─ Bioelectric: Regen success? → [YES] → Continue | [NO] → Flag
│  └─ Multiscale: K corridor? → [YES] → Continue | [NO] → Flag
│
├─ Meta-Tests
│  ├─ Harmonies independent? → [YES] → Continue | [NO] → Revise K
│  └─ Anti-Kosmic fails? → [YES] → Continue | [NO] → Revise K
│
└─ Final Decision
   ├─ ≥4 modules pass + Meta pass → [GO: Publish]
   ├─ 2-3 modules pass → [PARTIAL: Publish with caveats]
   └─ ≤1 module pass → [NO-GO: Major revision]
```

---

**END OF STATISTICAL ANALYSIS PLAN**

*This SAP is registered on OSF and locked prior to data collection. Any deviations will be transparently reported.*
