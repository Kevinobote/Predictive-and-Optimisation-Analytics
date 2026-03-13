# Makefile for Methodology Chapter Compilation

# Main document name (without .tex extension)
MAIN = main_methodology

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Output directory
OUT_DIR = output

.PHONY: all clean view help

# Default target
all: $(MAIN).pdf

# Compile the main document
$(MAIN).pdf: $(MAIN).tex references.bib diagrams/*.tex
	@echo "Compiling LaTeX document..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Running BibTeX..."
	$(BIBTEX) $(MAIN)
	@echo "Recompiling for references..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Final compilation..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Compilation complete! Output: $(MAIN).pdf"

# Quick compile (single pass, for testing)
quick: $(MAIN).tex
	@echo "Quick compilation (single pass)..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz
	rm -f *.fdb_latexmk *.fls *.nav *.snm *.vrb
	@echo "Clean complete!"

# Clean everything including PDF
cleanall: clean
	@echo "Removing PDF output..."
	rm -f $(MAIN).pdf
	@echo "All files cleaned!"

# View the PDF (Linux)
view: $(MAIN).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf; \
	elif command -v evince > /dev/null; then \
		evince $(MAIN).pdf; \
	elif command -v okular > /dev/null; then \
		okular $(MAIN).pdf; \
	else \
		echo "No PDF viewer found. Please open $(MAIN).pdf manually."; \
	fi

# Help target
help:
	@echo "Makefile for Methodology Chapter"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Compile the full document with bibliography"
	@echo "  make quick    - Quick single-pass compilation (for testing)"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make view     - Open the PDF with default viewer"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Example workflow:"
	@echo "  1. make          # Compile document"
	@echo "  2. make view     # View the result"
	@echo "  3. make clean    # Clean up auxiliary files"
