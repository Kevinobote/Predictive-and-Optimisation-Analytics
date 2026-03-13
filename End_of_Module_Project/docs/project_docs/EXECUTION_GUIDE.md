# Project Execution Guide
## Integrated Kiswahili Speech Analytics Pipeline

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux/macOS/Windows
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **GPU**: Optional (CUDA-compatible for faster training)
- **Python**: 3.8 or higher

### Software Requirements
- Python 3.8+
- pip package manager
- Jupyter Notebook
- Git (optional)

---

## 🔧 Installation Steps

### Step 1: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Note:** Installation may take 10-15 minutes depending on internet speed.

### Step 3: Download Dataset

1. Visit: https://commonvoice.mozilla.org/en/datasets
2. Download **Common Voice Corpus 11.0 - Swahili (sw)**
3. Extract the archive
4. Copy `validated.tsv` to `data/` directory
5. Copy `clips/` folder to `data/clips/`

**Dataset Structure:**
```
data/
├── validated.tsv
└── clips/
    ├── common_voice_sw_12345.mp3
    ├── common_voice_sw_12346.mp3
    └── ...
```

---

## 🚀 Execution Workflow

### Phase 1: Data Preparation (Notebook 1)

**Estimated Time:** 30-45 minutes

```bash
jupyter notebook notebooks/01_Data_Understanding_and_Preprocessing.ipynb
```

**What it does:**
- Loads and explores dataset
- Analyzes demographic distributions
- Engineers quality features
- Creates train/val/test splits
- Defines augmentation strategies

**Outputs:**
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

**Key Decisions:**
- Quality threshold: Median validation score
- Split ratio: 70/15/15
- Stratification: Gender + Age

---

### Phase 2: ASR Evaluation (Notebook 2)

**Estimated Time:** 1-2 hours (depends on dataset size)

```bash
jupyter notebook notebooks/02_ASR_Inference_and_WER_Evaluation.ipynb
```

**What it does:**
- Loads pretrained ASR model (RareElf/swahili-wav2vec2-asr)
- Performs batch inference
- Computes WER and CER
- Conducts stratified error analysis
- Quantifies fairness gaps

**Outputs:**
- `data/asr_predictions.csv`
- `data/asr_metrics.json`

**Expected Results:**
- Overall WER: 15-20%
- Demographic disparities identified
- Statistical significance confirmed

**Optimization Tip:**
- Use GPU for 5-10x speedup
- Process in batches of 32-64 samples
- Sample subset for quick testing (n=500)

---

### Phase 3: Bias Quantification (Notebook 3)

**Estimated Time:** 20-30 minutes

```bash
jupyter notebook notebooks/03_Predictive_Bias_Quantification_Logistic_Regression.ipynb
```

**What it does:**
- Trains logistic regression on demographic features
- Computes odds ratios
- Performs Wald test
- Calculates fairness metrics
- Provides prescriptive recommendations

**Outputs:**
- Coefficient analysis
- ROC curve
- Fairness metrics report

**Key Insights:**
- Which demographics predict ASR failure
- Where to apply weighted loss
- Data collection priorities

---

### Phase 4: Sentiment Analysis (Notebook 4)

**Estimated Time:** 2-3 hours (training time)

```bash
jupyter notebook notebooks/04_Sentiment_Pseudo_Labeling_and_DistilBERT.ipynb
```

**What it does:**
- Translates Kiswahili → English
- Generates pseudo-labels
- Fine-tunes DistilBERT
- Evaluates sentiment model

**Outputs:**
- `models/distilbert_sentiment_final/`
- Training metrics
- Evaluation report

**Training Configuration:**
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: AdamW

**Expected Results:**
- F1-Score: 68-72%
- Training time: ~30 minutes (GPU)

---

### Phase 5: Topic Modeling (Notebook 5)

**Estimated Time:** 15-20 minutes

```bash
jupyter notebook notebooks/05_KMeans_Topic_Modelling.ipynb
```

**What it does:**
- Preprocesses Kiswahili text
- Applies TF-IDF vectorization
- Performs KMeans clustering
- Visualizes topics with PCA/UMAP

**Outputs:**
- `data/clustered_data.csv`
- Topic visualizations
- Keyword extraction

**Optimal K Selection:**
- Elbow method
- Silhouette score
- Typically: 4-6 clusters

---

### Phase 6: Model Optimization (Notebook 6)

**Estimated Time:** 30-40 minutes

```bash
jupyter notebook notebooks/06_Model_Optimization_Quantization_and_Distillation.ipynb
```

**What it does:**
- Applies INT8 quantization
- Benchmarks latency
- Compares BERT vs DistilBERT
- Measures compression ratios

**Outputs:**
- `models/distilbert_int8.pth`
- Performance benchmarks
- Optimization report

**Expected Improvements:**
- 4x memory reduction
- 2x inference speedup
- <5% accuracy loss

---

### Phase 7: API Deployment (Notebook 7)

**Estimated Time:** 30 minutes

```bash
jupyter notebook notebooks/07_FastAPI_Deployment_Prototype.ipynb
```

**What it does:**
- Creates FastAPI application
- Implements full pipeline
- Generates Docker configuration
- Provides deployment guides

**Outputs:**
- `app/main.py`
- `Dockerfile`
- `DEPLOYMENT.md`
- `EDGE_DEPLOYMENT.md`

**Testing the API:**

1. Start server:
```bash
cd app
python main.py
```

2. Test endpoint:
```bash
python test_client.py
```

3. Benchmark performance:
```bash
python benchmark.py
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size
batch_size = 8  # instead of 16

# Clear cache
torch.cuda.empty_cache()
```

### Issue 2: Audio Loading Errors

**Solution:**
```bash
# Install ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/
```

### Issue 3: Slow Inference

**Solution:**
- Use GPU if available
- Reduce sample size for testing
- Apply quantization
- Use smaller models

### Issue 4: Import Errors

**Solution:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

---

## 📊 Expected Outputs Summary

| Notebook | Key Outputs | File Size |
|----------|-------------|-----------|
| 1 | train.csv, val.csv, test.csv | ~50MB |
| 2 | asr_predictions.csv, metrics.json | ~10MB |
| 3 | Fairness report, ROC curves | ~1MB |
| 4 | distilbert_sentiment_final/ | ~250MB |
| 5 | clustered_data.csv | ~20MB |
| 6 | distilbert_int8.pth | ~65MB |
| 7 | FastAPI app, Docker config | ~5MB |

**Total Storage Required:** ~400MB (excluding dataset)

---

## ⏱️ Time Estimates

| Phase | CPU Time | GPU Time |
|-------|----------|----------|
| Notebook 1 | 45 min | 30 min |
| Notebook 2 | 2 hours | 30 min |
| Notebook 3 | 30 min | 20 min |
| Notebook 4 | 3 hours | 45 min |
| Notebook 5 | 20 min | 15 min |
| Notebook 6 | 40 min | 30 min |
| Notebook 7 | 30 min | 30 min |
| **Total** | **7-8 hours** | **3-4 hours** |

---

## 🎯 Success Criteria

### Notebook 1
- ✅ All splits created
- ✅ Demographic imbalance quantified
- ✅ No missing critical features

### Notebook 2
- ✅ WER < 20%
- ✅ Statistical significance confirmed (p < 0.05)
- ✅ Fairness gaps documented

### Notebook 3
- ✅ AUC-ROC > 0.70
- ✅ Odds ratios interpretable
- ✅ Prescriptive insights generated

### Notebook 4
- ✅ F1-Score > 65%
- ✅ Model converges (loss decreases)
- ✅ No overfitting (val loss stable)

### Notebook 5
- ✅ Optimal K identified
- ✅ Silhouette score > 0.3
- ✅ Topics interpretable

### Notebook 6
- ✅ Compression ratio > 3x
- ✅ Speedup > 1.5x
- ✅ Accuracy degradation < 5%

### Notebook 7
- ✅ API responds successfully
- ✅ Latency < 500ms (GPU)
- ✅ Docker builds without errors

---

## 📝 Best Practices

### 1. Version Control
```bash
git init
git add .
git commit -m "Initial commit"
```

### 2. Experiment Tracking
- Document hyperparameters
- Save model checkpoints
- Log metrics to CSV/JSON

### 3. Reproducibility
- Set random seeds (SEED=42)
- Pin dependency versions
- Document environment

### 4. Code Quality
- Use meaningful variable names
- Add docstrings to functions
- Follow PEP 8 style guide

### 5. Resource Management
- Close file handles
- Clear GPU memory
- Use context managers

---

## 🚀 Next Steps After Completion

### Academic
1. Write research paper
2. Prepare presentation
3. Submit to conference (e.g., INTERSPEECH, ACL)

### Production
1. Deploy to cloud (AWS, GCP, Azure)
2. Set up monitoring (Prometheus, Grafana)
3. Implement CI/CD pipeline

### Research Extensions
1. Multi-lingual support (Swahili + English)
2. Real-time streaming inference
3. Active learning for data collection
4. Federated learning for privacy

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review notebook markdown cells
3. Consult README.md
4. Create GitHub issue

---

## ✅ Completion Checklist

- [ ] Environment set up
- [ ] Dependencies installed
- [ ] Dataset downloaded
- [ ] Notebook 1 executed
- [ ] Notebook 2 executed
- [ ] Notebook 3 executed
- [ ] Notebook 4 executed
- [ ] Notebook 5 executed
- [ ] Notebook 6 executed
- [ ] Notebook 7 executed
- [ ] API tested
- [ ] Documentation reviewed
- [ ] Results validated

---

**Good luck with your project! 🎓**
