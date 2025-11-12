#!/bin/bash
# Compile Paper 5 LaTeX manuscript to PDF
# Science journal submission format

echo "======================================"
echo "Paper 5 LaTeX Compilation Script"
echo "======================================"
echo ""

# Check if we're in the manuscript directory
if [ ! -f "paper5_science.tex" ]; then
    echo "❌ Error: paper5_science.tex not found"
    echo "   Please run this script from the manuscript/ directory"
    exit 1
fi

echo "📝 Compiling manuscript..."
echo ""

# First pass: generate document structure
echo "→ Pass 1: Initial compilation..."
pdflatex -interaction=nonstopmode paper5_science.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error in first pdflatex pass"
    echo "   Check paper5_science.log for details"
    exit 1
fi

# Generate bibliography
echo "→ Pass 2: Processing bibliography..."
bibtex paper5_science > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: BibTeX encountered issues"
    echo "   (This is often normal on first run)"
fi

# Second pass: incorporate references
echo "→ Pass 3: Incorporating references..."
pdflatex -interaction=nonstopmode paper5_science.tex > /dev/null 2>&1

# Third pass: resolve cross-references
echo "→ Pass 4: Finalizing cross-references..."
pdflatex -interaction=nonstopmode paper5_science.tex > /dev/null 2>&1

echo ""
if [ -f "paper5_science.pdf" ]; then
    echo "✅ SUCCESS! PDF generated: paper5_science.pdf"
    echo ""
    echo "📊 Document info:"
    echo "   Pages: $(pdfinfo paper5_science.pdf 2>/dev/null | grep Pages | awk '{print $2}')"
    echo "   Size: $(ls -lh paper5_science.pdf | awk '{print $5}')"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Review the PDF for any formatting issues"
    echo "   2. Fill in author information (lines 27-32)"
    echo "   3. Add Track B/C/D/E results from existing manuscripts"
    echo "   4. Write Introduction section"
    echo "   5. Expand Discussion section"
    echo ""
    echo "📦 Submission package:"
    echo "   - paper5_science.pdf (main manuscript)"
    echo "   - paper5_science.tex (source)"
    echo "   - references.bib (bibliography)"
    echo "   - ../logs/track_f/adversarial/*.png (figures)"
    echo ""
else
    echo "❌ Error: PDF was not generated"
    echo "   Check paper5_science.log for errors"
    exit 1
fi

# Optional: Clean up auxiliary files
read -p "Clean up auxiliary files (.aux, .log, .bbl, etc.)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
    echo "✅ Auxiliary files cleaned"
fi

echo ""
echo "======================================"
echo "Compilation complete!"
echo "======================================"
