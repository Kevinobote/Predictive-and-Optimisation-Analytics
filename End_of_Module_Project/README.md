# Tubonge: Kiswahili Speech Analytics Pipeline

**Student**: Kevin Obote | **Student No**: 190696  
**Course**: Predictive and Optimisation Analytics  
**Institution**: Strathmore University

---

## 🎯 Project Overview

End-to-end machine learning pipeline for Swahili speech analytics implementing:
- Automatic Speech Recognition (ASR)
- Sentiment Analysis via Pseudo-Labeling
- Bias Detection & Quantification
- Topic Modeling
- Model Optimization & Deployment

**Dataset**: Mozilla Common Voice 11.0 Swahili (26,614 samples)  
**Methodology**: CRISP-DM with focus on fairness, optimization, and production deployment

---

## 📊 Key Results (Verified from Notebooks)

| Metric | Value | Details |
|--------|-------|---------|
| **ASR WER** | 13.60% | Word Error Rate (86.4% accuracy) |
| **ASR CER** | 8.85% | Character Error Rate |
| **Sentiment F1** | 0.6125 | Weighted F1-score (62% accuracy) |
| **Bias Detection** | 55% acc | Logistic Regression (AUC=0.5588) |
| **Gender DPD** | 3.99% | Demographic Parity Difference |
| **Cohen's d** | 0.298 | Small effect size (p=0.007) |
| **Compression** | 1.31x | INT8 quantization (23.9% reduction) |
| **Speedup** | 5.19x | INT8 vs FP32 inference |
| **Clusters** | 10 | Optimal K-Means clusters |

---

## 📁 Project Structure

```
End_of_Module_Project/
├── notebooks/              # Jupyter notebooks (01-07)
│   ├── 01_Data_Understanding_and_Preprocessing.ipynb
│   ├── 02_ASR_Inference_and_WER_Evaluation.ipynb
│   ├── 03_Predictive_Bias_Quantification_Logistic_Regression.ipynb
│   ├── 04_Sentiment_Pseudo_Labeling_and_DistilBERT.ipynb
│   ├── 05_KMeans_Topic_Modelling.ipynb
│   ├── 06_Model_Optimization_Quantization_and_Distillation.ipynb
│   └── 07_FastAPI_Deployment_Prototype.ipynb
│
├── data/                   # Processed datasets
│   ├── train.csv, val.csv, test.csv
│   ├── asr_predictions.csv
│   ├── asr_metrics.json
│   ├── train_translated.csv
│   └── clustered_data.csv
│
├── models/                 # Trained models
│   ├── distilbert_sentiment/
│   ├── distilbert_sentiment_final/
│   └── distilbert_int8.pth
│
├── methodology/            # LaTeX methodology & presentation
│   ├── diagrams/          # TikZ flow diagrams
│   ├── docs/
│   │   └── presentation/  # Beamer presentation (19 slides)
│   ├── figures/
│   ├── main_methodology.pdf
│   └── main_methodology.tex
│
├── web_app/               # Production web application
│   ├── config/           # Modal deployment config
│   ├── docs/             # Deployment documentation
│   ├── scripts/          # Deployment scripts
│   ├── static/           # Frontend assets
│   ├── main.py           # FastAPI backend
│   └── index.html        # Web interface
│
├── docs/                  # Project documentation
│   ├── project_docs/     # Guides and summaries
│   └── project_materials/ # Course materials & proposal
│
├── deployment/            # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── scripts/               # Utility scripts
│   ├── run_docker.sh
│   └── test_audio_load.py
│
├── app/                   # Legacy app (deprecated)
├── src/                   # Source code modules
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd End_of_Module_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Execute notebooks in order (01-07)
```

### 3. Run Web Application

```bash
cd web_app
python main.py
# Visit http://localhost:8000
```

### 4. Deploy to Modal (Production)

```bash
cd web_app
conda run -n audio_ml modal deploy config/modal_app.py
```

---

## 📓 Notebooks Description

### Notebook 01: Data Understanding & Preprocessing
- Load Mozilla Common Voice 11.0 Swahili (26,614 samples)
- Demographic analysis (68% male, 32% female)
- Train/Val/Test split: 18,629 / 3,992 / 3,993
- Audio normalization and feature extraction

### Notebook 02: ASR Inference & WER Evaluation
- Wav2Vec2 ASR model inference
- **WER**: 13.60% | **CER**: 8.85%
- Gender bias analysis (ANOVA: p=0.007, Cohen's d=0.298)
- Demographic Parity Difference: 3.99%

### Notebook 03: Predictive Bias Quantification
- Logistic Regression for bias detection
- **Accuracy**: 55% | **AUC-ROC**: 0.5588
- Odds ratios for demographic features
- Fairness metrics (DPD, EOD)

### Notebook 04: Sentiment Pseudo-Labeling
- Translation-based pseudo-labeling (NLLB-200-600M)
- DistilBERT fine-tuning on Kiswahili
- **F1-Score**: 0.6125 | **Accuracy**: 62%
- Novel approach for low-resource languages

### Notebook 05: KMeans Topic Modeling
- TF-IDF feature extraction
- **Optimal K**: 10 clusters
- Topic interpretation and visualization
- 2,000 transcripts clustered

### Notebook 06: Model Optimization
- INT8 quantization: 516 MB → 393 MB (1.31x compression)
- **Speedup**: 5.19x (49.32ms → 9.51ms)
- DistilBERT vs BERT: 23.9% size reduction
- Knowledge distillation evaluation

### Notebook 07: FastAPI Deployment
- REST API implementation
- Async audio processing
- Docker containerization
- Production-ready prototype

---

## 🌐 Live Deployment

**Production API**: https://viviannyamoraa--tubonge-fastapi-app.modal.run/docs  
**GitHub Repository**: https://github.com/Kevinobote/Predictive-and-Optimisation-Analytics

**Features**:
- Live audio recording
- File upload (MP3, WAV, OGG, M4A, FLAC, AAC, WEBM)
- Real-time transcription
- Sentiment analysis
- Text summarization
- Multi-language UI (English/Kiswahili)

**Infrastructure**:
- Platform: Modal (Serverless)
- GPU: NVIDIA A100 (40GB)
- Auto-scaling: 0-20 containers
- Uptime: 99.5%

---

## 🔧 Key Technologies

### Machine Learning
- **ASR**: Wav2Vec2 (RareElf/swahili-wav2vec2-asr)
- **Sentiment**: DistilBERT (distilbert-base-uncased)
- **Summarization**: T5 (google/mt5-small)
- **Translation**: NLLB-200-distilled-600M
- **Clustering**: K-Means (scikit-learn)

### Frameworks
- **ML**: PyTorch, Transformers, Scikit-learn
- **Web**: FastAPI, Uvicorn
- **Audio**: Librosa, Soundfile
- **Deployment**: Modal, Docker

### Optimization
- INT8 Quantization
- Knowledge Distillation
- Data Augmentation (pitch shift, time stretch, noise injection)
- GPU Acceleration (A100, FP16, TF32)

---

## 📈 Statistical Significance

- **ANOVA Test**: F=7.32, p=0.007 (significant gender difference in WER)
- **Cohen's d**: 0.298 (small effect size)
- **Sample Size**: n=500 (adequate statistical power)
- **Chi-Square**: χ²=29013.06, p<0.0001 (gender-age association)

---

## 📚 Documentation

### Methodology
- **PDF**: `methodology/main_methodology.pdf`
- **LaTeX Source**: `methodology/main_methodology.tex`
- **Diagrams**: `methodology/diagrams/` (TikZ)

### Presentation
- **Location**: `methodology/docs/presentation/`
- **Slides**: 19 slides, 5-minute presentation
- **Format**: Beamer LaTeX
- **Quick Reference**: `methodology/docs/presentation/QUICK_REFERENCE_UPDATED.md`

### Deployment Guides
- **General**: `docs/project_docs/DEPLOYMENT.md`
- **Modal**: `web_app/docs/MODAL_DEPLOY.md`
- **Edge**: `docs/project_docs/EDGE_DEPLOYMENT.md`
- **Quick Start**: `docs/project_docs/QUICK_START.md`

---

## 🎓 Academic Context

**Course**: Predictive and Optimisation Analytics  
**Module**: 5 - Advanced Machine Learning  
**Institution**: Strathmore University  
**Methodology**: CRISP-DM  

**Rubric Coverage**:
- ✅ Data Collection & Preprocessing
- ✅ Feature Engineering & Selection
- ✅ Model Selection & Development (4 models)
- ✅ Optimization Techniques (Quantization, Distillation, Augmentation)
- ✅ Evaluation Metrics & Results
- ✅ Implementation & Code Quality
- ✅ Deployment (API Endpoint)
- ✅ Presentation & Communication

---

## 🔬 Innovation

**Novel Contribution**: Pseudo-labeling pipeline for sentiment analysis in low-resource languages

**Approach**:
1. Translate Kiswahili → English (NLLB-200-600M)
2. Apply English sentiment model
3. Map labels back to original Kiswahili text
4. Fine-tune DistilBERT on pseudo-labeled data

**Result**: F1-score of 0.6125 without manual labeling

---

## 🐛 Known Issues & Fixes

1. **Class Imbalance** (Notebook 03): Fixed with `class_weight='balanced'`
2. **NumPy Compatibility**: Use `soundfile` instead of `librosa.load()`
3. **Audio Path Parsing**: Use `ast.literal_eval()` for CSV strings
4. **Modal Deprecations**: Updated to `scaledown_window` and `max_containers`

---

## 🔮 Future Work

- Fine-tune larger models (BERT-large, T5-base)
- Expand to multi-dialect Swahili support
- Mobile app development (iOS/Android)
- Edge device optimization (Raspberry Pi)
- Real-time streaming inference
- Speaker diarization
- Emotion detection

---

## 📄 License

MIT License - Academic Use

---

## 📞 Contact

**Kevin Obote**  
Student No: 190696  
Email: kevin.obote@strathmore.edu  
GitHub: https://github.com/Kevinobote

---

## 🙏 Acknowledgments

- Mozilla Common Voice for the Swahili dataset
- Hugging Face for pre-trained models
- Strathmore University DSA Program
- Modal for serverless GPU infrastructure

---

**Last Updated**: March 2024  
**Status**: ✅ Production Ready
