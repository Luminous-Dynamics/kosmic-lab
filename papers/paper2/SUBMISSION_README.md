# Paper 1: Coherence-Guided Control

**Title**: Coherence-Guided Control: Corridor Discovery and Physics-Respecting Rescue in a Bioelectric Grid

## Submission Details

**Target Journal**: PLOS Computational Biology  
**Submission Status**: ✅ PDF Ready
**Date Prepared**: November 12, 2025

## Contents

- `manuscript.tex` - Main LaTeX source (PLOS format)
- `manuscript.pdf` - ✅ Compiled PDF (13 pages, 259 KB)
- `references.bib` - BibTeX bibliography

## Research Summary

This paper presents results from **Track B (SAC Controller)** and **Track C (Bioelectric Rescue)**:

### Track B: K-Index Guided SAC Controller
- 63% improvement in corridor navigation with K-index feedback
- Demonstrates coherence-guided reinforcement learning

### Track C: Physics-Respecting Bioelectric Rescue  
- 20% success rate using novel attractor-based mechanism
- First demonstration of K-index in rescue scenarios

## Recompilation Instructions

```bash
cd /srv/luminous-dynamics/kosmic-lab/papers/paper1
nix develop /srv/luminous-dynamics/kosmic-lab --command bash -c \
  "pdflatex manuscript.tex && pdflatex manuscript.tex"
```

## Key Results

- K-Index successfully guides learning in complex environments
- Coherence metrics enable physics-respecting control
- Novel rescue mechanism through attractor-based learning
