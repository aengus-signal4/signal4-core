#!/bin/bash
#
# Generate Methodology LaTeX Documents and Compile to PDF
#
# This script:
# 1. Runs the Python generator to query the database and create LaTeX files
# 2. Compiles the LaTeX files to PDF using pdflatex (via TinyTeX or system TeX)
#
# Prerequisites:
#   - Python 3.8+ with psycopg2 installed
#   - TinyTeX (via R) or a LaTeX distribution (MacTeX, TeX Live, or MiKTeX)
#
# Usage:
#   ./generate_pdf.sh                    # Use defaults
#   ./generate_pdf.sh --help             # Show Python script options
#   ./generate_pdf.sh --include-speaker-id --include-classification
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/methodology"

# Add TinyTeX to PATH if available (macOS)
if [ -d "$HOME/Library/TinyTeX/bin/universal-darwin" ]; then
    export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
fi
# Linux TinyTeX location
if [ -d "$HOME/.TinyTeX/bin/x86_64-linux" ]; then
    export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
fi

echo "=============================================="
echo "Methodology Document Generator"
echo "=============================================="
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate LaTeX files from database
echo "--- Step 1: Generating LaTeX files from database ---"
echo ""

cd "$SCRIPT_DIR"
python3 generate_methodology.py \
    --output-dir "$OUTPUT_DIR" \
    --verbose \
    "$@"

echo ""
echo "--- Step 2: Compiling LaTeX to PDF ---"
echo ""

cd "$OUTPUT_DIR"

# Check if pdflatex is available
PDFLATEX=""
if command -v pdflatex &> /dev/null; then
    PDFLATEX="pdflatex"
elif [ -x "$HOME/Library/TinyTeX/bin/universal-darwin/pdflatex" ]; then
    PDFLATEX="$HOME/Library/TinyTeX/bin/universal-darwin/pdflatex"
elif [ -x "$HOME/.TinyTeX/bin/x86_64-linux/pdflatex" ]; then
    PDFLATEX="$HOME/.TinyTeX/bin/x86_64-linux/pdflatex"
fi

if [ -z "$PDFLATEX" ]; then
    echo "ERROR: pdflatex not found"
    echo ""
    echo "Please install TinyTeX via R:"
    echo "  Rscript -e 'install.packages(\"tinytex\"); tinytex::install_tinytex()'"
    echo ""
    echo "Or install a LaTeX distribution:"
    echo "  macOS:   brew install --cask mactex"
    echo "  Ubuntu:  sudo apt-get install texlive-full"
    echo "  Windows: Install MiKTeX from https://miktex.org/"
    echo ""
    echo "LaTeX files have been generated but not compiled."
    echo "You can compile manually with:"
    echo "  cd $OUTPUT_DIR"
    echo "  pdflatex methodology.tex"
    echo "  pdflatex appendix_podcasts.tex"
    exit 1
fi

echo "Using pdflatex: $PDFLATEX"
echo ""

# Install required LaTeX packages via tlmgr if using TinyTeX
TLMGR=""
if [ -x "$HOME/Library/TinyTeX/bin/universal-darwin/tlmgr" ]; then
    TLMGR="$HOME/Library/TinyTeX/bin/universal-darwin/tlmgr"
elif [ -x "$HOME/.TinyTeX/bin/x86_64-linux/tlmgr" ]; then
    TLMGR="$HOME/.TinyTeX/bin/x86_64-linux/tlmgr"
fi

if [ -n "$TLMGR" ]; then
    echo "Installing required LaTeX packages..."
    $TLMGR install booktabs longtable multirow caption float hyperref xcolor 2>/dev/null || true
    echo ""
fi

# Compile methodology.tex (run twice for references)
echo "Compiling methodology.tex..."
$PDFLATEX -interaction=nonstopmode methodology.tex > /dev/null 2>&1 || true
$PDFLATEX -interaction=nonstopmode methodology.tex > /dev/null 2>&1

if [ -f "methodology.pdf" ]; then
    echo "  ✓ methodology.pdf created"
else
    echo "  ✗ Failed to create methodology.pdf"
    echo "    Run '$PDFLATEX methodology.tex' manually to see errors"
fi

# Compile appendix_podcasts.tex (run twice for longtable)
echo "Compiling appendix_podcasts.tex..."
$PDFLATEX -interaction=nonstopmode appendix_podcasts.tex > /dev/null 2>&1 || true
$PDFLATEX -interaction=nonstopmode appendix_podcasts.tex > /dev/null 2>&1

if [ -f "appendix_podcasts.pdf" ]; then
    echo "  ✓ appendix_podcasts.pdf created"
else
    echo "  ✗ Failed to create appendix_podcasts.pdf"
    echo "    Run '$PDFLATEX appendix_podcasts.tex' manually to see errors"
fi

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.lof *.lot 2>/dev/null || true

echo ""
echo "=============================================="
echo "GENERATION COMPLETE"
echo "=============================================="
echo ""
echo "Output files in: $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR"/*.pdf "$OUTPUT_DIR"/*.tex "$OUTPUT_DIR"/*.txt 2>/dev/null || true
echo ""
echo "To view PDFs:"
echo "  open $OUTPUT_DIR/methodology.pdf"
echo "  open $OUTPUT_DIR/appendix_podcasts.pdf"
