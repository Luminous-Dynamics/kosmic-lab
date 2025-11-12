# Cover Letter - Science Submission

**Paper**: Multiple Pathways to Coherent Perception–Action Coupling in AI

---

Dear Editors,

We submit "Multiple Pathways to Coherent Perception–Action Coupling in AI" for consideration at Science.

Our central, counterintuitive result is that gradient-based adversarial perturbations (FGSM) **enhance** a signature of intrinsic perception–action coherence (K-Index) by **+136%** (Cohen's d = **4.4**, p_FDR < **5.7×10⁻²⁰**), with **100% sanity-check verification** and **reward-independence** (Δ≈**0.01**).

## Why this fits Science

**Surprising**: Adversarial attacks designed to disrupt behavior amplify intrinsic coherence—a challenge to assumptions in AI safety and computational neuroscience.

**Methodological rigor**: Correct FGSM (x′=x+ε·sign(∇ₓL)), null distributions, FDR correction, lag analyses, partial correlation, and unit tests (21 passing).

**Generalizable**: Validated across multiple tracks and an ε sweep showing a monotonic dose-response (ε∈{0.05,0.10,0.15,0.20}; up to **2.7× enhancement**).

**Impact**: Offers a practical, reproducible metric (K-Index) to separate intrinsic coherence from reward optimization, informing AI safety and theories of perception.

**Data & Code**: All configs, logs, and analysis scripts are archived and will be publicly available (GitHub/Zenodo DOI provided upon request or in the submission).

We believe the novelty, rigor, and cross-disciplinary relevance make this work appropriate for Science.

Thank you for your consideration.

Sincerely,
**Tristan Stoltz** (Luminous Dynamics)
Email: tristan.stoltz@luminousdynamics.org
ORCID: 0009-0006-5758-6059

---

## Suggested Reviewers (No Conflicts)

1. **Dawn Song, PhD** – UC Berkeley (security, adversarial ML)
2. **Aleksander Madry, PhD** – MIT (robustness, adversarial methods)
3. **Aniruddh Goyal (Anirudh Goyal), PhD** – Google DeepMind (representation, cognition)
4. **Yoshua Bengio, PhD** – Mila/Université de Montréal (representation learning, causality)
5. **Konrad Kording, PhD** – UPenn (computational neuroscience, causality)
6. **Nick Frosst** – Cohere (robustness & interpretability)

### Alternate Pool (if needed)
- Ian Goodfellow, PhD – Apple (adversarial ML pioneer)
- Daphne Bavelier, PhD – UniGe (perception–action coupling in humans)
- Tomaso Poggio, PhD – MIT CBMM (theory, vision)

---

## Submission Metadata

**Title**: Multiple Pathways to Coherent Perception–Action Coupling in AI

**One-sentence summary** (≤125 chars):
Adversarial perturbations amplify an intrinsic AI coherence signature rather than disrupt it.

**Keywords**: adversarial robustness; perception–action coupling; machine consciousness; K-Index; reinforcement learning

**Classification**: Computer Science / Artificial Intelligence; Neuroscience (computational)

**Data & Code availability**: "All code, configs, and data to reproduce figures and tables are available at [GitHub]/[Zenodo DOI]; unit tests included."

**Competing interests**: "None."

---

## Figure & Table Captions (Tightened)

**Fig. 2** — Track F robustness: Mean K±SE across conditions; grey band shows 95% null interval.

**Fig. 6** — FGSM sanity: Per-step base vs adversarial loss; 100% of steps show increased loss.

**Fig. 7** — Robust variants convergence: Pearson vs Spearman K; points near unity line indicate agreement.

**Table 1** — Summary stats: Mean, SE, 95% CI for all conditions.

**Table 2** — Pairwise comparisons: Cohen's d and FDR-adjusted p-values.

---

## Final Preflight Checklist (5 Quick Checks)

- [ ] Insert author list, affiliations, and ORCIDs
- [ ] Verify that the ε sweep numbers in Supplement match the CSVs
- [ ] Confirm Goodfellow et al., 2015 is cited for FGSM (and appears in references.bib)
- [ ] Ensure FDR-corrected p-values are what you report in the main text
- [ ] Add links/DOI in the Data & Code statement
