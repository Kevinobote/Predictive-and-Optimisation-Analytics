# Tubonge Speech Analytics - Beamer Presentation

**Student**: Kevin Obote  
**Student No**: 190696  
**Course**: Predictive and Optimisation Analytics  
**Duration**: 5 minutes

## Overview

This Beamer presentation covers the complete Tubonge Speech Analytics project, including:

- Data Collection & Preprocessing
- Model Selection & Development (Logistic Regression, DistilBERT, T5, K-Means)
- Optimization Techniques (Quantization, Knowledge Distillation, Data Augmentation)
- Evaluation Metrics & Results
- Implementation & Code Quality
- Deployment (Modal + A100 GPU)

## Files

- `presentation.tex` - Main Beamer LaTeX source
- `compile_presentation.sh` - Compilation script
- `methodology/diagrams/` - TikZ flow diagrams

## Compilation

### Option 1: Using the Script (Recommended)

```bash
./compile_presentation.sh
```

### Option 2: Manual Compilation

```bash
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for references
```

### Option 3: Using Makefile

```bash
make presentation
```

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages:
  - beamer
  - tikz
  - hyperref
  - listings
  - xcolor
  - booktabs

## Presentation Structure

1. **Title & Overview** (30 seconds)
2. **Data Collection & Preprocessing** (45 seconds)
3. **Model Selection & Development** (60 seconds)
   - Includes pseudo-labeling diagram
   - ML pipeline visualization
4. **Optimization Techniques** (45 seconds)
5. **Evaluation Metrics & Results** (45 seconds)
6. **Implementation & Code Quality** (30 seconds)
7. **Deployment** (30 seconds)
8. **Conclusion & Demo** (15 seconds)

**Total**: ~5 minutes

## Key Highlights

### Models Used
- **Logistic Regression**: Bias prediction (82.1% accuracy)
- **DistilBERT**: Sentiment analysis (87.4% accuracy)
- **T5 (MT5-small)**: Text summarization (ROUGE-1: 0.42)
- **K-Means**: Topic clustering (Silhouette: 0.67)

### Optimization Techniques
- **INT8 Quantization**: 4x compression
- **Knowledge Distillation**: 40% size reduction
- **Data Augmentation**: 30% synthetic data increase
- **GPU Acceleration**: 5-7x speedup (A100)

### Results
- **ASR WER**: 18.3%
- **Sentiment F1**: 0.86
- **Deployment Latency**: <2s for 10s audio
- **API Uptime**: 99.5%

## Live Deployment

- **API Docs**: https://viviannyamoraa--tubonge-fastapi-app.modal.run/docs
- **GitHub**: https://github.com/Kevinobote/Predictive-and-Optimisation-Analytics/tree/main/End_of_Module_Project/web_app

## Presentation Tips

1. **Timing**: Allocate 1 minute per major section
2. **Flow Diagrams**: Briefly explain each diagram (15-20 seconds each)
3. **Results**: Emphasize the 87.4% sentiment accuracy and novel pseudo-labeling approach
4. **Demo**: Have the live API ready in a browser tab
5. **Questions**: Prepare for questions on:
   - Pseudo-labeling methodology
   - Model optimization trade-offs
   - Deployment architecture
   - Bias detection approach

## Troubleshooting

### Missing Packages

If compilation fails due to missing packages:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-fonts-recommended

# macOS (with Homebrew)
brew install --cask mactex

# Or install missing packages via tlmgr
tlmgr install beamer tikz hyperref listings xcolor booktabs
```

### TikZ Diagram Issues

If diagrams don't render:
1. Ensure `methodology/diagrams/` folder is in the same directory
2. Check that all `.tex` files are present in the diagrams folder
3. Verify TikZ libraries are loaded

### PDF Not Opening

The script attempts to auto-open the PDF. If it fails:
- Manually open `presentation.pdf` from the file browser
- Comment out the auto-open section in `compile_presentation.sh`

## Contact

**Kevin Obote**  
Student No: 190696  
Email: kevin.obote@strathmore.edu

---

**Note**: This presentation is designed for a 5-minute time slot. Practice timing to ensure smooth delivery within the allocated time.
