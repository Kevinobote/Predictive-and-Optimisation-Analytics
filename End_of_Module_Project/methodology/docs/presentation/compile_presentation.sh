#!/bin/bash

# Beamer Presentation Compilation Script
# Author: Kevin Obote (190696)

echo "=========================================="
echo "Compiling Beamer Presentation"
echo "=========================================="

# Navigate to project directory
cd "$(dirname "$0")"

# Compile the presentation (run twice for references)
echo "First pass..."
pdflatex -interaction=nonstopmode presentation.tex

echo "Second pass (for references)..."
pdflatex -interaction=nonstopmode presentation.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f presentation.aux presentation.log presentation.nav presentation.out presentation.snm presentation.toc presentation.vrb

echo "=========================================="
echo "Compilation complete!"
echo "Output: presentation.pdf"
echo "=========================================="

# Open the PDF (optional - comment out if not needed)
if command -v xdg-open &> /dev/null; then
    xdg-open presentation.pdf
elif command -v open &> /dev/null; then
    open presentation.pdf
fi
