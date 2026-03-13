# Methodology Chapter - Kiswahili Speech Analytics System

## Overview
This folder contains the comprehensive methodology chapter for the dissertation on "Kiswahili Speech Analytics: An End-to-End System for Automatic Speech Recognition, Sentiment Analysis, and Summarization."

## Structure

```
methodology/
├── main_methodology.tex          # Main LaTeX document
├── references.bib                # Bibliography file
├── diagrams/                     # TikZ diagram files
│   ├── agile_process.tex
│   ├── system_architecture.tex
│   ├── data_flow.tex
│   ├── ml_pipeline.tex
│   ├── pseudo_labeling.tex
│   ├── component_integration.tex
│   └── deployment_architecture.tex
├── figures/                      # Additional figures (if any)
└── README.md                     # This file
```

## Compilation Instructions

### Prerequisites
Ensure you have a LaTeX distribution installed:
- **Linux**: TeX Live (`sudo apt-get install texlive-full`)
- **macOS**: MacTeX
- **Windows**: MiKTeX or TeX Live

Required LaTeX packages:
- tikz
- algorithm
- algpseudocode
- listings
- hyperref
- graphicx
- booktabs
- subcaption

### Compile the Document

#### Method 1: Using pdflatex (Recommended)
```bash
cd methodology
pdflatex main_methodology.tex
bibtex main_methodology
pdflatex main_methodology.tex
pdflatex main_methodology.tex
```

#### Method 2: Using latexmk (Automated)
```bash
cd methodology
latexmk -pdf main_methodology.tex
```

#### Method 3: Using Overleaf
1. Upload all files to Overleaf
2. Set main document to `main_methodology.tex`
3. Compile automatically

### Output
The compilation will generate `main_methodology.pdf` containing the complete methodology chapter.

## Chapter Contents

### 1. Introduction
- Research design overview
- Alignment with research objectives
- Methodology justification

### 2. Research Design
- Overall approach
- Software development methodology (Agile)
- Justification for chosen methodology

### 3. System Analysis
- Requirements analysis (functional and non-functional)
- Feasibility study
- System constraints

### 4. System Design
- System architecture (3-layer design)
- Data flow design
- ML pipeline architecture
- API design
- Component design (ASR, Sentiment, Summarization)

### 5. Implementation
- Development environment and tools
- Data preparation methodology
- Model development (transfer learning, pseudo-labeling)
- System implementation (backend, frontend, integration)

### 6. Testing and Validation
- Testing methodology (unit, integration, system)
- Model validation approach
- Performance benchmarking

### 7. Deployment Methodology
- Deployment architecture
- Deployment process
- Monitoring and maintenance

### 8. Ethical Considerations
- Data privacy
- Model bias mitigation

### 9. Limitations
- Acknowledged limitations of the methodology

## Key Diagrams

1. **Agile Process Diagram**: Iterative development cycle
2. **System Architecture**: 3-layer architecture (Presentation, Application, Model)
3. **Data Flow Diagram**: End-to-end data processing flow
4. **ML Pipeline**: Complete machine learning workflow
5. **Pseudo-Labeling Workflow**: Novel approach for sentiment labeling
6. **Component Integration**: How system components interact
7. **Deployment Architecture**: Production deployment setup

## Customization

### Adding New Sections
Add new sections in `main_methodology.tex`:
```latex
\section{Your New Section}
\subsection{Subsection}
Content here...
```

### Adding New Diagrams
1. Create a new `.tex` file in `diagrams/` folder
2. Use TikZ syntax to create the diagram
3. Include in main document:
```latex
\begin{figure}[H]
    \centering
    \input{diagrams/your_diagram.tex}
    \caption{Your Caption}
    \label{fig:your_label}
\end{figure}
```

### Adding References
Add new entries to `references.bib`:
```bibtex
@article{author2024title,
  title={Article Title},
  author={Author Name},
  journal={Journal Name},
  year={2024}
}
```

Cite in text:
```latex
\cite{author2024title}
```

## Alignment with Dissertation Guidelines

This methodology chapter follows the guidelines by:

✅ Aligning with research objectives from Chapter 1
✅ Explaining chosen methodologies with justification
✅ Focusing on "HOW" rather than theoretical explanations
✅ Using software development methodology (Agile)
✅ Covering all development phases (Analysis, Design, Implementation, Testing, Validation)
✅ Including proper diagrams (UML-style, data flow, architecture)
✅ Referencing tools and technologies used
✅ Detailing testing and validation methodologies
✅ Providing justification for methodology choices
✅ Avoiding "original" work (descriptive, not prescriptive)

## Integration with Other Chapters

This methodology chapter should be integrated with:
- **Chapter 1**: Introduction and Research Objectives
- **Chapter 2**: Literature Review
- **Chapter 3**: (This chapter - Methodology)
- **Chapter 4**: Implementation and Results
- **Chapter 5**: Evaluation and Discussion
- **Chapter 6**: Conclusion and Future Work

## Notes for Dissertation Writing

1. **Tables to Complete**: Fill in "TBD" values in performance tables with actual results
2. **Figures**: Add screenshots or additional figures to `figures/` folder if needed
3. **Validation**: Update validation results based on actual testing
4. **References**: Add any additional references used in your research
5. **Customization**: Adjust content to match your specific implementation details

## Troubleshooting

### Common Issues

**Issue**: Missing TikZ package
```bash
sudo apt-get install texlive-pictures
```

**Issue**: Bibliography not showing
- Run bibtex after first pdflatex compilation
- Compile pdflatex twice more

**Issue**: Figures not appearing
- Check file paths in `\input{}` commands
- Ensure all diagram files exist in `diagrams/` folder

**Issue**: Compilation errors
- Check for unmatched braces `{}`
- Verify all `\begin{}` have matching `\end{}`
- Check for special characters that need escaping

## Contact and Support

For questions about this methodology chapter structure, refer to:
- LaTeX documentation: https://www.latex-project.org/help/documentation/
- TikZ manual: https://tikz.dev/
- Overleaf tutorials: https://www.overleaf.com/learn

## Version History

- **v1.0** (2024): Initial comprehensive methodology chapter
  - Complete system analysis and design
  - All architectural diagrams
  - Testing and validation methodology
  - Deployment architecture

## License

This methodology chapter template is part of the Kiswahili Speech Analytics dissertation project.
