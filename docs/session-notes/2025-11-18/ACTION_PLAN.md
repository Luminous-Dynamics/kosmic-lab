# K-Index Research: Action Plan

**Created**: November 19, 2025
**Decision Required By**: November 22, 2025
**Target Submission**: December 13, 2025 (Option A)

---

## Immediate Actions (This Week)

### Day 1-2: Team Alignment

| Action | Owner | Deadline | Status |
|--------|-------|----------|--------|
| Share `COLLABORATOR_BRIEFING.md` with all co-authors | Lead | Nov 19 | ⬜ |
| Schedule decision meeting (30 min) | Lead | Nov 20 | ⬜ |
| Review `FINAL_SESSION_SUMMARY.md` | All co-authors | Nov 21 | ⬜ |
| Decide on path (A, B, or C) | Team | Nov 22 | ⬜ |

### Decision Meeting Agenda (30 min)

1. **5 min**: Summary of validation findings
2. **10 min**: Review three options (A/B/C)
3. **10 min**: Discuss implications for each paper
4. **5 min**: Vote and assign responsibilities

---

## Option A: Reframe Papers (Recommended)

**Timeline**: 4 weeks (Nov 25 - Dec 20)
**Submission Target**: December 13, 2025

### Week 1: Abstracts & Titles (Nov 25-29)

| Task | Paper | Owner | Deadline |
|------|-------|-------|----------|
| Rewrite abstract using `COPY_PASTE_SECTIONS.md` | Paper 1 | TBD | Nov 26 |
| Rewrite abstract | Paper 4 | TBD | Nov 27 |
| Rewrite abstract | Paper 5 | TBD | Nov 28 |
| Decision on Paper 3 (withdraw/revise) | Paper 3 | Team | Nov 29 |

**Deliverable**: All new titles and abstracts approved by Nov 29

### Week 2: Results Sections (Dec 2-6)

| Task | Paper | Owner | Deadline |
|------|-------|-------|----------|
| Revise results with behavioral framing | Paper 1 | TBD | Dec 3 |
| Revise results | Paper 4 | TBD | Dec 4 |
| Revise results | Paper 5 | TBD | Dec 5 |
| Internal review of revised results | All | Dec 6 |

**Deliverable**: All results sections reframed around validated findings

### Week 3: Limitations & Future Work (Dec 9-13)

| Task | Paper | Owner | Deadline |
|------|-------|-------|----------|
| Add limitations section (use template) | All papers | TBD | Dec 10 |
| Add future work section (use template) | All papers | TBD | Dec 11 |
| Update figures/tables (relabel as behavioral) | All papers | TBD | Dec 12 |
| Final internal review | All | Dec 13 |

**Deliverable**: Complete revised drafts ready for submission

### Week 4: Submission (Dec 16-20)

| Task | Owner | Deadline |
|------|-------|----------|
| Final proofreading | All | Dec 17 |
| Format check | Lead | Dec 18 |
| Submit Paper 1 | Lead | Dec 19 |
| Submit Papers 4, 5 | Lead | Dec 20 |

---

## Option B: Redesign Experiments

**Timeline**: 6 months (Nov 2025 - May 2026)
**Not recommended** - high risk, long timeline

### If Chosen:

1. **Month 1**: Design new experiments with performance metrics
2. **Month 2-3**: Implement and run experiments
3. **Month 4**: Analyze results
4. **Month 5**: Revise papers
5. **Month 6**: Submit

---

## Option C: Validate H2 Only

**Timeline**: 2 months (Nov 2025 - Jan 2026)
**Medium risk** - may not transfer from CartPole

### If Chosen:

1. **Week 1-2**: Set up original environments with performance metrics
2. **Week 3-4**: Run H2 validation experiments
3. **Week 5-6**: Analyze results
4. **Week 7-8**: Revise papers around H2 (drop K-Index)

---

## Paper-Specific Actions

### Paper 1 (Track B+C)

**New Title**: "Learning Action Diversity Through Coherence Feedback"

| Section | Action | Template Location |
|---------|--------|-------------------|
| Abstract | Replace entirely | `COPY_PASTE_SECTIONS.md` → Paper 1 |
| Results 4.2 | Remove performance claims | `PAPER_REFRAMING_GUIDE.md` |
| Discussion | Add limitation | `COPY_PASTE_SECTIONS.md` → Universal |

**Key change**: Replace "63% improvement" with "H2: 0.0 → 0.99"

### Paper 4 (Track E)

**New Title**: "K-Index as a Training Progress Metric"

| Section | Action | Template Location |
|---------|--------|-------------------|
| Abstract | Replace entirely | `COPY_PASTE_SECTIONS.md` → Paper 4 |
| Methods 3.1 | Add task learnability note | `COPY_PASTE_SECTIONS.md` → Paper 4 |
| Results | Show K-reward dissociation | `PAPER_REFRAMING_GUIDE.md` |

**Key change**: Explicitly state K doesn't predict rewards (r = -0.01)

### Paper 5 (Track F)

**New Title**: "Behavioral Changes Under Adversarial Attack"

| Section | Action | Template Location |
|---------|--------|-------------------|
| Abstract | Replace entirely | `COPY_PASTE_SECTIONS.md` → Paper 5 |
| Results | Reframe as behavioral signature | `PAPER_REFRAMING_GUIDE.md` |

**Key change**: "Robustness" → "Behavioral stability under perturbation"

### Paper 3 (Track D)

**Recommendation**: Withdraw

**If revising**: Major reframe as "Preliminary Investigation" with explicit caveats about K-performance validity

---

## Quality Checklist

Before any paper submission, verify:

### Content Accuracy
- [ ] No claim that K predicts performance
- [ ] All percentages refer to behavioral metrics, not performance
- [ ] Limitations section is present and substantive
- [ ] Future work includes performance validation

### Language Consistency
- [ ] "Performance" → "behavior" or "training dynamics"
- [ ] "Improvement" → "change" or "difference"
- [ ] "Better" → "higher" or "more diverse"
- [ ] No unqualified performance claims

### Data Integrity
- [ ] All statistics are about K, H2, or behavior
- [ ] Effect sizes interpreted correctly
- [ ] P-values reported for all claims
- [ ] Sample sizes stated

---

## Risk Mitigation

### Risk 1: Co-author Disagreement

**Mitigation**:
- Share validation scripts for independent verification
- Point to `REPRODUCIBILITY_REPORT.md`
- Offer to discuss specific concerns in meeting

### Risk 2: Reviewer Asks About Performance

**Mitigation**:
- Proactively address in limitations section
- Include future work on performance validation
- Be transparent about what we measured

### Risk 3: Paper 3 Authors Resist Withdrawal

**Mitigation**:
- Offer to help redesign for Option B
- Propose as "preliminary investigation" with major caveats
- Emphasize integrity over publication count

---

## Communication Templates

### Email to Co-Authors

```
Subject: K-Index Validation Findings - Decision Needed by Nov 22

Dear colleagues,

After rigorous validation of our K-Index research, we've identified a
fundamental issue: K-Index does not correlate with external performance
metrics (r = -0.01 in Track E).

Please review the attached COLLABORATOR_BRIEFING.md before our meeting
on [DATE]. We need to decide whether to:

A) Reframe papers around validated behavioral findings (2-4 weeks)
B) Redesign experiments with performance metrics (3-6 months)
C) Validate H2 only in original environments (1-2 months)

I recommend Option A. The reframed papers are still valuable and publishable.

Full documentation: [link to docs/session-notes/2025-11-18/]

Best,
[Name]
```

### Reviewer Response (If Asked About Performance)

```
Thank you for this important question. As noted in Section X.X, we did
not validate K-Index against task-specific performance metrics in this
study. Our findings concern behavioral patterns (action diversity,
training dynamics) rather than task success.

We explicitly identify performance validation as future work (Section Y.Y).
Preliminary results in CartPole suggest action diversity (H2) correlates
with performance (r = +0.71), but this requires validation in the
environments studied here.
```

---

## Success Criteria

### For Option A (Reframe)

By December 20, 2025:
- [ ] Papers 1, 4, 5 submitted with honest framing
- [ ] Paper 3 withdrawn or substantially revised
- [ ] No performance claims without validation
- [ ] Clear path to future validation work

### For Any Option

- [ ] Team alignment on findings
- [ ] No unvalidated claims published
- [ ] Foundation laid for rigorous future work
- [ ] Scientific integrity maintained

---

## Resources

All materials in `/docs/session-notes/2025-11-18/`:

| Need | File |
|------|------|
| 1-page summary for team | `COLLABORATOR_BRIEFING.md` |
| Full technical findings | `FINAL_SESSION_SUMMARY.md` |
| How to reframe each paper | `PAPER_REFRAMING_GUIDE.md` |
| LaTeX text to copy-paste | `COPY_PASTE_SECTIONS.md` |
| Verify findings independently | `run_all_validations.py` |

---

**Next Step**: Share `COLLABORATOR_BRIEFING.md` with co-authors today.

*"The rigorous path is the only path worth taking."*

