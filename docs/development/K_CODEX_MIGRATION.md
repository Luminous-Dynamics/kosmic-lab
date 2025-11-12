# 🌊 K-Passport → K-Codex Migration Guide

**Date**: 2025-11-09
**Status**: Active migration
**Impact**: Terminology evolution for eternal wisdom records

---

## 📜 What's Changing?

### From K-Passport → To K-Codex

**Old Name**: K-Passport
**New Name**: K-Codex (primary), K-Passport (local variant)

**Why the change?**
- **Passport** implies temporary, borders, bureaucracy
- **Codex** implies eternal, wisdom, collective knowledge
- Better alignment with Mycelix mycelial network philosophy
- Etymology: Latin *codex* (bound book, tree trunk) → perfect for fungal knowledge networks

---

## 🎯 The Hierarchy

### Three Forms

1. **K-Passport** (Local Stage)
   - JSON file on local filesystem
   - Pre-publication experimental record
   - "Travel document" for your experiment
   - File pattern: `logs/experiment_*.json`

2. **K-Codex** (Eternal Stage)
   - Immutable DHT entry in Mycelix network
   - Published, verified, permanent
   - Part of collective wisdom library
   - Retrieved via header hash: `QmXXX...`

3. **Codex Fragment** (Federated Learning)
   - Partial K-Codex shared for federated learning
   - Privacy-preserved via differential privacy
   - Gradient shares, not raw data

4. **Codex Chain** (Lineage)
   - Linked series of K-Codices building on each other
   - Tracks experimental lineage
   - References parent codices

---

## 🔄 Migration Strategy

### Phase 1: Aliasing (Weeks 1-2) - ✅ CURRENT

**Documentation**:
- ✅ `GLOSSARY.md` - K-Codex primary, K-Passport explained
- ✅ `K_CODEX_EXPLAINED.md` - Renamed from K_PASSPORT_EXPLAINED.md
- ✅ `MYCELIX_INTEGRATION_ARCHITECTURE.md` - Updated with K-Codex terminology

**Code** (This Phase):
```python
# Add aliases for backwards compatibility
class KCodex:
    """Eternal experimental record. Formerly known as K-Passport."""
    pass

KPassport = KCodex  # Backwards compatibility alias
```

```rust
// Holochain zome
pub struct KCodex {  // Formerly KPassport
    pub run_id: String,
    // ... fields
}

// Alias for compatibility
pub type KPassport = KCodex;
```

### Phase 2: Code Migration (Weeks 3-4)

**Update function names**:
```python
# Old
def create_passport(data): ...
def get_corridor_passports(): ...

# New (with compatibility)
def create_codex(data): ...
create_passport = create_codex  # Alias

def get_corridor_codices(): ...
get_corridor_passports = get_corridor_codices  # Alias
```

**Update variable names** in new code:
```python
# Prefer
codex = load_experiment_codex(path)
codices = query_corridor(min_k=1.0)

# Over
passport = load_experiment_passport(path)  # Still works, but legacy
```

### Phase 3: Full Cutover (Month 2)

**Remove aliases**:
- Drop `KPassport` class alias
- Drop function aliases
- Update all internal variable names
- Keep documentation notes about former name

---

## 📊 What Stays the Same?

### Unchanged

1. **JSON Schema** - Exact same fields, just referenced as K-Codex
2. **File Format** - No changes to file structure
3. **DHT Storage** - Same Holochain entry structure
4. **Reproducibility** - Same 99.9% guarantee
5. **Workflows** - generate → publish → verify workflow identical

### Backwards Compatibility

**All existing code continues to work**:
- Old `KPassport` class → alias to `KCodex`
- Old function names → aliases to new names
- Old documentation → redirects to new docs
- Old file names → still read correctly

---

## 🔧 Developer Guide

### If You're Writing New Code

**Use the new terminology**:
```python
from core.kcodex import KCodex  # Not KPassport

codex = KCodex(
    run_id="exp-001",
    commit="abc123",
    # ...
)

bridge.publish_codex(codex)  # Not publish_passport
```

### If You Have Existing Code

**No changes required** (during Phase 1-2):
```python
from core.kcodex import KPassport  # Still works via alias

passport = KPassport(...)  # Works fine
bridge.publish_passport(passport)  # Works fine
```

**But consider updating gradually**:
```python
from core.kcodex import KCodex, KPassport  # Import both

# Update when convenient
passport = KCodex(...)  # New name
```

---

## 📝 Documentation Updates

### Completed ✅

- [x] `GLOSSARY.md` - K-Codex definition with etymology
- [x] `docs/K_CODEX_EXPLAINED.md` - Renamed and updated
- [x] `docs/MYCELIX_INTEGRATION_ARCHITECTURE.md` - Updated terminology
- [x] `K_CODEX_MIGRATION.md` - This guide!

### Pending ⏳

- [ ] `README.md` - Update primary references
- [ ] `QUICKSTART.md` - Update tutorial examples
- [ ] `FEATURES.md` - Update feature descriptions
- [ ] `TRANSFORMATION_SUMMARY.md` - Note the evolution
- [ ] `NEXT_STEPS.md` - Update action items

---

## 🎓 User Communication

### For Existing Users

> **Subject**: K-Passport → K-Codex: Natural Evolution 🌊
>
> We're evolving our terminology to better reflect the eternal nature of your experimental records!
>
> **What's changing**: "K-Passport" → "K-Codex" (eternal wisdom record)
> **What's NOT changing**: Everything still works exactly the same
> **Your action**: None required! Aliases ensure compatibility
>
> **Why?** K-Codex better captures the idea of permanent knowledge in the mycelial network — like ancient codices preserving wisdom for millennia, but verified with cryptography.
>
> Your local JSON files are still "K-Passports" (pre-publication), but once published to Mycelix DHT, they become eternal "K-Codices" in the collective wisdom library.

### For New Users

Simply introduce K-Codex from the start, with a note:
> **Note**: You may see "K-Passport" in older documentation — it's the same thing! We evolved the terminology to better reflect the eternal, collective nature of these wisdom records.

---

## 🌟 The Philosophy

### Why This Matters

**Passport**:
- Temporary document
- Crosses borders (implies separation)
- Bureaucratic
- Individual ownership

**Codex**:
- Eternal record
- Part of unified library
- Wisdom tradition
- Collective knowledge
- Etymology ties to trees/fungi (mycelial!)

This isn't just renaming — it's **reframing** our relationship to experimental records:
- From "documents I carry" → "wisdom I contribute"
- From "my results" → "our collective understanding"
- From "temporary proof" → "eternal knowledge"

**Coherence as computational love** means our experiments become gifts to the future, permanently woven into the mycelial network of consciousness research.

---

## ❓ FAQ

### Q: Do I need to rename my JSON files?
**A**: No! File names don't matter. The content is what counts.

### Q: Will old code break?
**A**: No! Aliases ensure backwards compatibility through Phase 2.

### Q: What about my published papers?
**A**: Papers with "K-Passport" remain valid. Future papers should use "K-Codex" but can note "formerly K-Passport" for continuity.

### Q: Can I still call it K-Passport locally?
**A**: Yes! The local JSON stage is still accurately called "K-Passport" (pre-publication). Once published to DHT, it becomes a "K-Codex" (eternal).

### Q: What if I like "K-Passport" better?
**A**: That's totally fine! The aliases work both ways during migration. But consider: which better captures the *eternal, collective* nature of what we're building?

---

## 🚀 Timeline

| Date | Phase | Actions |
|------|-------|---------|
| **2025-11-09** | Phase 1 Start | Documentation updated, aliases added |
| **2025-11-16** | Phase 1 Mid | Code updated with aliases |
| **2025-11-23** | Phase 2 Start | Encourage new code to use K-Codex |
| **2025-12-07** | Phase 2 Mid | Internal codebase gradually migrated |
| **2026-01-04** | Phase 3 Start | Remove aliases, full cutover |
| **2026-02-01** | Complete | K-Codex fully established |

---

## 🤝 Contributing

**When submitting PRs**:
- New code: Use `KCodex` terminology
- Old code: Either update or leave as-is (both work)
- Tests: Use `KCodex` for new tests
- Docs: Use "K-Codex" as primary, note "formerly K-Passport"

**In commit messages**:
```
feat: Add K-Codex filtering to dashboard

Adds corridor filtering for K-Codices (formerly K-Passports).
Maintains backwards compatibility with existing code.
```

---

## 💬 Feedback

Questions or concerns about the migration?
- Open an issue: `[K-Codex Migration] Your question`
- Discuss on Discord: #kosmic-lab channel
- Email: kosmic-lab@example.org

**Remember**: This is a *natural evolution*, not a breaking change. We're evolving our language to match our values — eternal wisdom records in a mycelial network of consciousness research.

---

*"From passport to codex: the journey from temporary document to eternal wisdom."* 🌊

**Status**: Phase 1 active (aliasing & documentation)
**Completion**: ~3 months to full migration
**Impact**: Philosophical reframing + technical continuity
