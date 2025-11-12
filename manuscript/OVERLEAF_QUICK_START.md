# 🚀 Generate PDF in 2 Minutes with Overleaf

**Fastest path to submission-ready PDF**

---

## Step 1: Go to Overleaf (30 seconds)

1. Navigate to: https://www.overleaf.com
2. Click **"Sign Up"** or **"Log In"** (free account)
3. Click **"New Project"** → **"Upload Project"**

---

## Step 2: Prepare Files (30 seconds)

### Files to Upload from `/srv/luminous-dynamics/kosmic-lab/manuscript/`:

**Required Core Files:**
- ✅ `paper5_science.tex`
- ✅ `references.bib`

**Required Figures from** `../logs/track_f/adversarial/`:
- ✅ `figure2_track_f_robustness.png`
- ✅ `figure6_fgsm_sanity.png`
- ✅ `figure7_robust_variants.png`

### Quick Upload Command (if needed):
```bash
cd /srv/luminous-dynamics/kosmic-lab
tar -czf manuscript_upload.tar.gz \
  manuscript/paper5_science.tex \
  manuscript/references.bib \
  logs/track_f/adversarial/figure2_track_f_robustness.png \
  logs/track_f/adversarial/figure6_fgsm_sanity.png \
  logs/track_f/adversarial/figure7_robust_variants.png
```

Then upload `manuscript_upload.tar.gz` to Overleaf.

---

## Step 3: Compile (30 seconds)

1. Overleaf will automatically compile when you upload
2. If not, click the green **"Recompile"** button
3. Wait ~10-15 seconds for compilation
4. PDF appears in the right panel ✅

---

## Step 4: Download PDF (30 seconds)

1. Click **"Download PDF"** button (top right)
2. Save as: `paper5_science.pdf`
3. **Done!** You now have a submission-ready PDF ✅

---

## 🎯 What You'll See in the PDF

**Complete manuscript with:**
- Title: "Multiple Pathways to Coherent Perception–Action Coupling in AI"
- Author: Tristan Stoltz (ORCID, Luminous Dynamics)
- Abstract: 250 words with all key findings
- Track F Results: Complete with all statistics
- 3 Figures: High-resolution (300 DPI)
- 2 Tables: Summary stats and pairwise comparisons
- Methods: FGSM, K-Index, statistical analysis
- Supplementary: Complete epsilon sweep dose-response
- References: 20+ citations

**Science Journal Formatting:**
- Double-spaced text
- Line numbers for review
- 1-inch margins
- 11pt font
- Professional appearance

---

## 🔧 Troubleshooting

### If figures don't show:
1. Check figure paths in Overleaf file tree
2. Figures should be in root directory (same level as .tex file)
3. If in subdirectory, update paths in paper5_science.tex:
   ```latex
   \includegraphics[width=0.9\textwidth]{figure2_track_f_robustness.png}
   ```

### If compilation errors:
1. Check the error log (red "Logs" button)
2. Most common issue: missing figures
3. Solution: Re-upload figures or fix paths

### If references don't appear:
1. Make sure `references.bib` is uploaded
2. Click "Recompile" twice (first pass generates .aux, second pass includes refs)

---

## 📋 File Checklist

Before uploading, verify you have:

**Core Files:**
- [ ] `paper5_science.tex` (13 KB, 232 lines)
- [ ] `references.bib` (6.5 KB, 20+ citations)

**Figures:**
- [ ] `figure2_track_f_robustness.png` (219 KB, 2355×1978 px, 300 DPI)
- [ ] `figure6_fgsm_sanity.png` (310 KB, 2022×2333 px, 300 DPI)
- [ ] `figure7_robust_variants.png` (442 KB, 2955×1977 px, 300 DPI)

---

## 🎉 Success Criteria

**Your PDF is ready when you see:**
1. ✅ Title page with author and affiliation
2. ✅ Abstract with all key statistics (d = 4.4, p < 5.7×10⁻²⁰)
3. ✅ Results section with Track F findings
4. ✅ All 3 figures rendered correctly
5. ✅ Both tables displaying properly
6. ✅ Supplementary section with epsilon sweep table
7. ✅ References list at end
8. ✅ Line numbers on left margin
9. ✅ Double-spaced text throughout

---

## 🚀 Alternative: Local LaTeX Installation

If you prefer to compile locally (requires 5-10 minutes setup):

### Add to NixOS Configuration:
```nix
# /etc/nixos/configuration.nix
environment.systemPackages = with pkgs; [
  texlive.combined.scheme-medium
];
```

Then rebuild:
```bash
sudo nixos-rebuild switch
```

Once installed, compile with:
```bash
cd /srv/luminous-dynamics/kosmic-lab/manuscript
bash compile.sh
```

---

## ⏱️ Time Comparison

| Method | Setup Time | Compile Time | Total Time |
|--------|------------|--------------|------------|
| **Overleaf** | 0 min | <1 min | **~2 min** ✅ |
| **NixOS Install** | 5-10 min | 1 min | 6-11 min |
| **Nix Shell** | 3-5 min | 1 min | 4-6 min |

**Recommendation**: Use Overleaf for immediate PDF generation

---

## 📥 File Locations

All files are in:
```
/srv/luminous-dynamics/kosmic-lab/manuscript/
├── paper5_science.tex
├── references.bib
└── ../logs/track_f/adversarial/
    ├── figure2_track_f_robustness.png
    ├── figure6_fgsm_sanity.png
    └── figure7_robust_variants.png
```

---

## 🎯 Next Steps After PDF Generation

Once you have the PDF:

1. **Review the PDF** (5 minutes)
   - Check all figures render correctly
   - Verify tables are formatted properly
   - Confirm all statistics are visible

2. **Add Data/Code URL** (2 minutes)
   - Replace `[GitHub repository URL]` in Code Availability section
   - Save and recompile

3. **Prepare Submission** (5 minutes)
   - Copy cover letter from `COVER_LETTER.md`
   - Have PDF ready
   - Have figures as separate files (if required by journal)

4. **Submit to Science!** 🚀

---

**Status**: Complete manuscript package ready for instant PDF generation via Overleaf

**Estimated time to submission-ready PDF**: **2 minutes**

*Guide created: November 12, 2025*
