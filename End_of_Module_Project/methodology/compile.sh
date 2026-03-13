#!/bin/bash

# Compilation script for Methodology Chapter
# Usage: ./compile.sh [option]
# Options: full, quick, clean, view

set -e  # Exit on error

MAIN="main_methodology"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if pdflatex is installed
check_latex() {
    if ! command -v pdflatex &> /dev/null; then
        print_error "pdflatex not found. Please install LaTeX distribution."
        echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
        echo "  macOS: brew install --cask mactex"
        exit 1
    fi
}

# Full compilation with bibliography
compile_full() {
    print_info "Starting full compilation..."
    
    print_info "First pass..."
    pdflatex -interaction=nonstopmode "$MAIN.tex" || {
        print_error "First compilation failed!"
        exit 1
    }
    
    print_info "Running BibTeX..."
    bibtex "$MAIN" || {
        print_warning "BibTeX warnings (may be normal)"
    }
    
    print_info "Second pass..."
    pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null
    
    print_info "Final pass..."
    pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null
    
    print_info "Compilation complete! Output: ${MAIN}.pdf"
}

# Quick compilation (single pass)
compile_quick() {
    print_info "Quick compilation (single pass)..."
    pdflatex -interaction=nonstopmode "$MAIN.tex" || {
        print_error "Compilation failed!"
        exit 1
    }
    print_info "Quick compilation complete!"
}

# Clean auxiliary files
clean_files() {
    print_info "Cleaning auxiliary files..."
    rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz
    rm -f *.fdb_latexmk *.fls *.nav *.snm *.vrb
    print_info "Clean complete!"
}

# Clean all including PDF
clean_all() {
    clean_files
    print_info "Removing PDF..."
    rm -f "$MAIN.pdf"
    print_info "All files cleaned!"
}

# View PDF
view_pdf() {
    if [ ! -f "$MAIN.pdf" ]; then
        print_error "PDF not found. Compile first!"
        exit 1
    fi
    
    print_info "Opening PDF..."
    if command -v xdg-open &> /dev/null; then
        xdg-open "$MAIN.pdf"
    elif command -v open &> /dev/null; then
        open "$MAIN.pdf"
    elif command -v evince &> /dev/null; then
        evince "$MAIN.pdf"
    else
        print_warning "No PDF viewer found. Please open $MAIN.pdf manually."
    fi
}

# Show help
show_help() {
    echo "Methodology Chapter Compilation Script"
    echo ""
    echo "Usage: ./compile.sh [option]"
    echo ""
    echo "Options:"
    echo "  full      - Full compilation with bibliography (default)"
    echo "  quick     - Quick single-pass compilation"
    echo "  clean     - Remove auxiliary files"
    echo "  cleanall  - Remove all generated files including PDF"
    echo "  view      - Open the PDF with default viewer"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./compile.sh          # Full compilation"
    echo "  ./compile.sh quick    # Quick compilation"
    echo "  ./compile.sh view     # View PDF"
}

# Main script logic
check_latex

case "${1:-full}" in
    full)
        compile_full
        ;;
    quick)
        compile_quick
        ;;
    clean)
        clean_files
        ;;
    cleanall)
        clean_all
        ;;
    view)
        view_pdf
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

exit 0
