# Quick Reference Card - Tubonge Presentation (ACTUAL RESULTS)
**Kevin Obote | Student No: 190696**

---

## 🎯 Core Message
Novel pseudo-labeling approach for low-resource language sentiment analysis with optimized deployment

---

## 📊 Key Metrics (Memorize These!)

| Metric | Value | Context |
|--------|-------|---------|
| Total Dataset | 26,614 samples | Mozilla Common Voice 11.0 |
| Train/Val/Test | 18,629 / 3,992 / 3,993 | 70/15/15 split |
| ASR WER | 13.60% | Wav2Vec2-Swahili |
| ASR CER | 8.85% | Character-level accuracy |
| Sentiment Accuracy | 62% | DistilBERT |
| Sentiment F1 | 0.6125 | Weighted average |
| Bias Detection Acc | 55% | Logistic Regression |
| AUC-ROC | 0.5588 | Bias quantification |
| Gender DPD | 3.99% | Demographic parity |
| Cohen's d | 0.298 | Small effect size |
| Model Compression | 1.31x | INT8 Quantization |
| Size Reduction | 23.9% | 516 MB → 393 MB |
| Inference Speedup | 5.19x | INT8 vs FP32 |
| DistilBERT Reduction | 23.9% | vs BERT-base |
| Optimal Clusters | 10 | K-Means |
| API Latency | <2s | 10s audio |
| Uptime | 99.5% | Modal deployment |

---

## 🔧 Four Models, Four Purposes

1. **Logistic Regression** → Bias quantification (55% acc, AUC=0.5588)
2. **DistilBERT** → Sentiment analysis (62% acc, F1=0.6125)
3. **T5 (MT5-small)** → Summarization (deployed, not evaluated)
4. **K-Means** → Topic clustering (10 clusters, 2,000 samples)

---

## ⚡ Four Optimization Techniques

1. **Quantization** → INT8, 1.31x compression, 5.19x speedup
2. **Knowledge Distillation** → DistilBERT, 23.9% smaller
3. **Data Augmentation** → 30% increase, pitch/time/noise
4. **Inference Optimization** → 49.32ms → 9.51ms (A100)

---

## 🔄 Pseudo-Labeling Pipeline (3 Steps)

1. **Translate**: Kiswahili → English (NLLB-200-600M)
2. **Label**: English sentiment model predicts
3. **Map**: Labels → Original Kiswahili text

**Why?** Solves low-resource language challenge

---

## 🌐 Deployment Stack

- **Backend**: FastAPI
- **Frontend**: HTML/CSS/JS
- **Hosting**: Modal (Serverless)
- **GPU**: NVIDIA A100 (40GB)
- **Container**: Docker

---

## 🔗 Important Links

**Live API**: https://viviannyamoraa--tubonge-fastapi-app.modal.run/docs  
**GitHub**: https://github.com/Kevinobote/Predictive-and-Optimisation-Analytics/tree/main/End_of_Module_Project/web_app

---

## 💡 Innovation Highlight

**Novel Contribution**: Pseudo-labeling pipeline for sentiment analysis in low-resource languages, achieving F1-score of 0.6125 on Kiswahili text

---

## 📈 Results Summary

✅ Processed 26,614 samples (18,629 train)  
✅ Novel pseudo-labeling approach (F1=0.6125)  
✅ 5.19x inference speedup  
✅ Bias detection & quantification (DPD=3.99%)  
✅ Production-ready deployment  
✅ High-quality modular code  
✅ 99.5% uptime

---

## ⏱️ Time Management

- **0:00-0:50**: Title + Overview
- **0:50-1:30**: Data + Preprocessing
- **1:30-3:15**: Models + Pipeline
- **3:15-4:30**: Optimization + Results
- **4:30-5:40**: Implementation + Deployment
- **5:40-6:30**: Conclusion + Questions

**Target**: Finish by 4:30, leave 30s for Q&A

---

## 🎤 Opening Line

"Good morning/afternoon. I'm Kevin Obote, presenting Tubonge - a Swahili Speech Analytics Platform demonstrating predictive and optimization analytics techniques for low-resource language processing."

---

## 🎬 Closing Line

"The system is live at this URL. I'm happy to answer questions about the methodology, optimization techniques, or deployment architecture. Thank you!"

---

## ❓ Top 3 Expected Questions

1. **Why pseudo-labeling?**  
   → Scalable, cost-effective, F1=0.6125 validated

2. **Quantization trade-offs?**  
   → 1.31x compression, 5.19x speedup, 23.9% size reduction

3. **Other languages?**  
   → Yes! Architecture is language-agnostic

---

## 🚨 Emergency Backup

If demo fails:
- Show screenshots in slides
- Explain architecture verbally
- Reference GitHub repo for code

If time runs short:
- Skip "Future Work" slide
- Condense "Code Quality" section
- Focus on results and innovation

---

## ✅ Pre-Presentation Checklist

- [ ] Laptop charged
- [ ] PDF loaded
- [ ] Browser with API docs open
- [ ] Backup USB drive
- [ ] Timer ready
- [ ] Water bottle
- [ ] Deep breath! 😊

---

## ⚠️ IMPORTANT: ACTUAL vs PLACEHOLDER VALUES

**Use these ACTUAL values from notebooks:**
- ✅ WER: 13.60% (NOT 18.3%)
- ✅ Sentiment F1: 0.6125 (NOT 0.86 or 87.4%)
- ✅ Bias Acc: 55% (NOT 82.1%)
- ✅ Compression: 1.31x (NOT 4x)
- ✅ Size reduction: 23.9% (NOT 40%)
- ✅ Speedup: 5.19x (NOT 5-7x)
- ✅ Clusters: 10 (NOT 5)
- ✅ Total samples: 26,614 (NOT 18,629)

---

**Remember**: You know this material inside-out. Speak confidently, make eye contact, and show enthusiasm for your work!

**Good luck! 🚀**
