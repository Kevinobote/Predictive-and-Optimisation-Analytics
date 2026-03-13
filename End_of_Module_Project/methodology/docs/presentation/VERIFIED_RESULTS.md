# ACTUAL RESULTS FROM NOTEBOOKS - VERIFIED VALUES

## Notebook 01: Data Understanding and Preprocessing
- **Total Dataset Size**: 26,614 samples
- **Train Set**: 18,629 samples
- **Validation Set**: 3,992 samples  
- **Test Set**: 3,993 samples
- **Gender Distribution**: 68% male, 32% female
- **Chi-Square Test (Gender vs Age)**: χ² = 29013.06, p-value < 0.0001

## Notebook 02: ASR Inference and WER Evaluation
- **Overall WER**: 13.60%
- **Overall CER**: 8.85%
- **WER by Gender**:
  - Female: 13.54% (mean), std=0.112, n=190
  - Male: 17.09% (mean), std=0.128, n=145
- **ANOVA Test (WER ~ Gender)**:
  - F-statistic: 7.3169
  - P-value: 0.007182
  - Significant: Yes
- **Cohen's d (Effect Size)**: 0.2983 (Small effect)
- **Demographic Parity Difference (Gender)**: 3.99%

## Notebook 03: Predictive Bias Quantification - Logistic Regression
- **AUC-ROC**: 0.5588
- **Overall Accuracy**: 55%
- **Classification Report**:
  - Class 0 (Low Quality): Precision=0.12, Recall=0.54, F1=0.19
  - Class 1 (High Quality): Precision=0.91, Recall=0.55, F1=0.69
  - Weighted Avg: Precision=0.84, Recall=0.55, F1=0.64
- **Logistic Regression Coefficients**:
  - Intercept: -0.586 (Odds Ratio: 0.557)
  - Gender: -0.110 (Odds Ratio: 0.896)
  - Age: -0.015 (Odds Ratio: 0.985)
  - Validation Score: 0.412 (Odds Ratio: 1.510)

## Notebook 04: Sentiment Pseudo-Labeling and DistilBERT
- **F1-Score (Weighted)**: 0.6125 (61.25%)
- **Overall Accuracy**: 62%
- **Classification Report**:
  - Class 0 (Negative): Precision=0.53, Recall=0.40, F1=0.45
  - Class 1 (Positive): Precision=0.67, Recall=0.77, F1=0.71
  - Macro Avg: Precision=0.60, Recall=0.58, F1=0.58
  - Weighted Avg: Precision=0.61, Recall=0.62, F1=0.61
- **Training Time**: 587.05 seconds (~9.8 minutes)
- **Training Loss**: 0.6203
- **Training Speed**: 4.088 samples/second

## Notebook 05: KMeans Topic Modelling
- **Optimal K**: 10 clusters
- **Total Samples Clustered**: 2,000
- **Cluster Distribution**:
  - Cluster 0: 25 samples
  - Cluster 1: 49 samples
  - Cluster 2: 1,595 samples (largest)
  - Cluster 3: 32 samples
  - Cluster 4: 58 samples
  - Cluster 5: 65 samples
  - Cluster 6: 33 samples
  - Cluster 7: 67 samples
  - Cluster 8: 48 samples
  - Cluster 9: 28 samples
- **Note**: Silhouette score not explicitly printed in notebook output

## Notebook 06: Model Optimization - Quantization and Distillation
- **FP32 Model Size**: 516.26 MB
- **INT8 Model Size**: 393.10 MB
- **Compression Ratio**: 1.31x (23.9% reduction)
- **FP32 Latency**: 49.32 ± 203.16 ms
- **INT8 Latency**: 9.51 ± 8.52 ms
- **Speedup**: 5.19x
- **BERT Size**: 678.53 MB
- **DistilBERT Size**: 516.26 MB
- **Size Reduction (BERT → DistilBERT)**: 23.9%

## Summary of Key Metrics for Presentation

### Data
- 26,614 total samples → 18,629 train / 3,992 val / 3,993 test

### ASR Performance
- WER: 13.60%
- CER: 8.85%
- Gender bias (DPD): 3.99%
- Cohen's d: 0.298 (small effect)

### Sentiment Analysis
- F1-Score: 0.6125 (61.25%)
- Accuracy: 62%

### Bias Quantification (Logistic Regression)
- AUC-ROC: 0.5588
- Accuracy: 55%

### Topic Modeling
- Optimal clusters: 10
- Largest cluster: 1,595 samples (79.75%)

### Model Optimization
- Compression: 1.31x (INT8 quantization)
- Speedup: 5.19x
- DistilBERT reduction: 23.9% vs BERT

## IMPORTANT NOTES

1. **Sentiment F1 is 0.6125, NOT 0.86 or 0.87** - this was a placeholder
2. **Logistic Regression accuracy is 55%, NOT 82%** - this was a placeholder
3. **No explicit Silhouette score** was printed in notebook 05
4. **Compression ratio is 1.31x, NOT 4x** - the 4x was aspirational
5. **DistilBERT is 23.9% smaller than BERT, NOT 40%** - actual measured value
6. **No ROUGE scores** were computed for summarization (T5 was used but not evaluated)
7. **Test sample size for ASR was 500**, not the full 3,993

## What to Remove/Adjust in Presentation

❌ Remove: "87.4% sentiment accuracy" → Use: "62% accuracy, F1=0.6125"
❌ Remove: "82.1% bias detection accuracy" → Use: "55% accuracy, AUC=0.5588"
❌ Remove: "4x compression" → Use: "1.31x compression (23.9% reduction)"
❌ Remove: "40% size reduction" → Use: "23.9% size reduction"
❌ Remove: "Silhouette score: 0.67" → Use: "10 optimal clusters identified"
❌ Remove: "ROUGE-1: 0.42" → T5 was used but not evaluated
❌ Remove: "5-7x GPU speedup" → Use: "5.19x speedup (INT8 vs FP32)"
❌ Adjust: "18,629 samples" is correct for training set only
