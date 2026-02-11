# Integrated Kiswahili Speech Analytics Pipeline

## Project Overview
End-to-end machine learning pipeline for low-resource ASR systems using Mozilla Common Voice 11.0 Swahili dataset. Implements CRISP-DM methodology with focus on fairness, optimization, and deployment.

## Key Results
- **ASR Performance**: WER 13.5-17.1% (target: <20%)
- **Fairness Gap**: 0.12 (target: <0.15)
- **Model Compression**: 4x size reduction
- **Inference Latency**: <500ms

## Project Structure
```
End_of_Module_Project/
├── notebooks/
│   ├── 01_Data_Understanding_and_Preprocessing.ipynb
│   ├── 02_ASR_Inference_and_WER_Evaluation.ipynb
│   ├── 03_Predictive_Bias_Quantification_Logistic_Regression.ipynb
│   ├── 04_Sentiment_Pseudo_Labeling_and_DistilBERT.ipynb
│   ├── 05_KMeans_Topic_Modelling.ipynb
│   ├── 06_Model_Optimization_Quantization_and_Distillation.ipynb
│   └── 07_FastAPI_Deployment_Prototype.ipynb
├── data/
│   └── asr_predictions.csv
├── models/
│   └── (saved models)
├── requirements.txt
├── EXECUTION_GUIDE.md
└── README.md
```

## Notebooks Description

### 1. Data Understanding (CRISP-DM Phase 1-2)
- Load Mozilla Common Voice 11.0 Swahili
- Demographic analysis & imbalance quantification
- Stratified train/val/test splits

### 2. ASR Inference & Evaluation (Phase 3)
- RareElf/swahili-wav2vec2-asr inference
- WER/CER computation
- Statistical significance testing (ANOVA, Cohen's d)
- Fairness gap analysis (DPD metric)

### 3. Predictive Bias Quantification (Phase 4)
- Logistic regression for bias detection
- Odds ratios & Wald tests
- Equal Opportunity Difference (EOD)
- **Fixed**: Class imbalance handling with balanced weights

### 4. Sentiment Analysis (Phase 4)
- Pseudo-labeling with VADER
- DistilBERT fine-tuning
- Cross-lingual transfer learning

### 5. Topic Modeling (Phase 4)
- KMeans clustering on TF-IDF features
- Silhouette analysis
- Topic interpretation

### 6. Model Optimization (Phase 5)
- Dynamic quantization (INT8)
- Knowledge distillation
- Latency benchmarking

### 7. FastAPI Deployment (Phase 6)
- REST API prototype
- Async inference
- Docker containerization

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run notebooks in order
jupyter notebook notebooks/01_Data_Understanding_and_Preprocessing.ipynb

# Or execute all
jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
```

## Dataset
- **Source**: Mozilla Common Voice 11.0 Swahili
- **Size**: 26,614 samples
- **Cache**: `~/.cache/huggingface/datasets/mozilla-foundation___common_voice_11_0/sw/`

## Key Technologies
- **ASR**: Wav2Vec2, Transformers
- **ML**: Scikit-learn, PyTorch
- **NLP**: NLTK, VADER, DistilBERT
- **Deployment**: FastAPI, Uvicorn
- **Audio**: Soundfile, Librosa

## Performance Metrics
| Metric | Target | Achieved |
|--------|--------|----------|
| WER (Female) | <20% | 13.5% |
| WER (Male) | <20% | 17.1% |
| Fairness Gap | <0.15 | 0.12 |
| Model Size | 4x reduction | ✓ |
| Latency | <500ms | ✓ |

## Statistical Significance
- **ANOVA**: F=7.32, p=0.007 (significant)
- **Cohen's d**: 0.30 (small effect)
- **Sample Size**: n=500 (adequate power)

## Known Issues & Fixes
1. **Class Imbalance** (Notebook 3): Fixed with `class_weight='balanced'`
2. **NumPy Compatibility**: Use `soundfile` instead of `librosa.load()`
3. **Audio Path Parsing**: Use `ast.literal_eval()` for CSV strings

## Future Work
- Implement adversarial debiasing
- Add real-time streaming inference
- Expand to multi-dialect support
- Deploy to cloud (AWS/GCP)

## Contributors
- Strathmore University DSA Module 5 Project

## License
MIT License - Academic Use

## References
- Mozilla Common Voice: https://commonvoice.mozilla.org/
- Wav2Vec2: https://arxiv.org/abs/2006.11477
- CRISP-DM: https://www.datascience-pm.com/crisp-dm-2/

---
**Last Updated**: December 2024
