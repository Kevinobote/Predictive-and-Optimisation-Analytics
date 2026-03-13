# Presentation Files

This folder contains all files related to the Beamer presentation for the Predictive and Optimisation Analytics project.

## Files

### Main Presentation
- **presentation.tex** - LaTeX Beamer source code (19 slides)
- **presentation.pdf** - Compiled presentation (252 KB)
- **compile_presentation.sh** - Automated compilation script

### Documentation
- **PRESENTATION_README.md** - Compilation instructions and usage guide
- **PRESENTATION_NOTES.md** - Slide-by-slide talking points and Q&A preparation
- **PRESENTATION_FINAL.md** - Final summary and checklist
- **QUICK_REFERENCE_UPDATED.md** - One-page cheat sheet with actual results
- **QUICK_REFERENCE.md** - Original quick reference (deprecated)
- **DIAGRAM_SIZING.md** - Diagram scaling adjustments reference
- **VERIFIED_RESULTS.md** - Actual results from notebooks 01-06

## Quick Start

### Compile Presentation
```bash
cd methodology/docs/presentation
./compile_presentation.sh
```

### View Presentation
```bash
xdg-open presentation.pdf
```

## Key Information

**Student**: Kevin Obote  
**Student No**: 190696  
**Duration**: 5 minutes  
**Slides**: 19  

## Actual Results Used

All metrics in the presentation are verified from notebooks:

- **Dataset**: 26,614 samples (18,629 train / 3,992 val / 3,993 test)
- **ASR WER**: 13.60%
- **ASR CER**: 8.85%
- **Sentiment F1**: 0.6125 (62% accuracy)
- **Bias Detection**: 55% accuracy, AUC=0.5588
- **Gender DPD**: 3.99%
- **Compression**: 1.31x (23.9% reduction)
- **Speedup**: 5.19x
- **Clusters**: 10 optimal

## Diagrams Included

All diagrams are sourced from `methodology/diagrams/`:
- Pseudo-labeling pipeline
- ML pipeline
- Data flow architecture
- System architecture (not used)

## Links

- **Live API**: https://viviannyamoraa--tubonge-fastapi-app.modal.run/docs
- **GitHub**: https://github.com/Kevinobote/Predictive-and-Optimisation-Analytics/tree/main/End_of_Module_Project/web_app

## Notes

- All placeholder values have been replaced with actual results
- Diagrams are scaled to fit slides properly
- Presentation is ready for research defense
- Use QUICK_REFERENCE_UPDATED.md for key metrics
