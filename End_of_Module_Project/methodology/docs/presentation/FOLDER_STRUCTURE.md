# Presentation Folder Structure

```
methodology/docs/presentation/
├── README.md                        # This folder overview
├── FOLDER_STRUCTURE.md              # Folder organization guide
│
├── presentation.tex                 # Main LaTeX source (19 slides)
├── presentation.pdf                 # Compiled presentation (252 KB)
├── compile_presentation.sh          # Build script
│
├── VERIFIED_RESULTS.md              # Actual results from notebooks 01-06
├── QUICK_REFERENCE_UPDATED.md       # Cheat sheet with actual values
├── PRESENTATION_NOTES.md            # Slide-by-slide talking points
├── PRESENTATION_README.md           # Compilation & usage guide
├── PRESENTATION_FINAL.md            # Final summary & checklist
├── DIAGRAM_SIZING.md                # Diagram scaling reference
└── QUICK_REFERENCE.md               # Original reference (deprecated)
```

## File Categories

### Essential Files (Use These)
1. **presentation.pdf** - The final presentation
2. **QUICK_REFERENCE_UPDATED.md** - Key metrics cheat sheet
3. **PRESENTATION_NOTES.md** - Talking points for each slide
4. **VERIFIED_RESULTS.md** - Source of truth for all metrics

### Source Files
- **presentation.tex** - LaTeX source code
- **compile_presentation.sh** - Compilation script

### Supporting Documentation
- **PRESENTATION_README.md** - How to compile and use
- **PRESENTATION_FINAL.md** - Pre-presentation checklist
- **DIAGRAM_SIZING.md** - Technical notes on diagram scaling

### Deprecated
- **QUICK_REFERENCE.md** - Old version with placeholder values

## Related Folders

### Diagrams Source
```
../../diagrams/
├── pseudo_labeling.tex      # Used in Slide 6
├── ml_pipeline.tex          # Used in Slide 7
├── data_flow.tex            # Used in Slide 11
└── system_architecture.tex  # Not used
```

### Methodology PDF
```
../../main_methodology.pdf   # Full methodology document
```

## Usage

### To Compile
```bash
cd methodology/docs/presentation
./compile_presentation.sh
```

### To Present
1. Open `presentation.pdf`
2. Review `QUICK_REFERENCE_UPDATED.md` for key metrics
3. Use `PRESENTATION_NOTES.md` for talking points

### To Verify Metrics
Check `VERIFIED_RESULTS.md` for actual notebook outputs

## Key Metrics (Quick Reference)

- Dataset: 26,614 samples
- ASR WER: 13.60%
- Sentiment F1: 0.6125
- Bias Acc: 55%
- Compression: 1.31x
- Speedup: 5.19x
- Clusters: 10

All values verified from notebooks 01-06.
