# Quick Reference Card - Tubonge Presentation
**Kevin Obote | Student No: 190696**

---

## 🎯 Core Message
Novel pseudo-labeling approach for low-resource language sentiment analysis with optimized deployment

---

## 📊 Key Metrics (Memorize These!)

| Metric | Value | Context |
|--------|-------|---------|
| Dataset Size | 18,629 samples | Mozilla Common Voice |
| Sentiment Accuracy | 87.4% | DistilBERT |
| ASR WER | 18.3% | Wav2Vec2-Swahili |
| Model Compression | 4x | INT8 Quantization |
| Size Reduction | 40% | Knowledge Distillation |
| GPU Speedup | 5-7x | A100 vs CPU |
| API Latency | <2s | 10s audio |
| Uptime | 99.5% | Modal deployment |
| Bias Disparity | 15% | Gender in ASR |

---

## 🔧 Four Models, Four Purposes

1. **Logistic Regression** → Bias prediction (82.1% acc)
2. **DistilBERT** → Sentiment analysis (87.4% acc)
3. **T5 (MT5-small)** → Summarization (ROUGE-1: 0.42)
4. **K-Means** → Topic clustering (Silhouette: 0.67)

---

## ⚡ Four Optimization Techniques

1. **Quantization** → INT8, 4x compression, <2% loss
2. **Knowledge Distillation** → DistilBERT, 40% smaller
3. **Data Augmentation** → 30% increase, pitch/time/noise
4. **GPU Acceleration** → FP16, TF32, A100

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

**Novel Contribution**: Pseudo-labeling pipeline for sentiment analysis in low-resource languages, validated at 87.4% accuracy

---

## 📈 Results Summary

✅ Effective data preprocessing (18K+ samples)  
✅ Novel pseudo-labeling approach  
✅ 4x model compression  
✅ Bias detection & quantification  
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
   → Scalable, cost-effective, 87.4% validated accuracy

2. **Quantization trade-offs?**  
   → 4x compression, 2-3x speedup, <2% accuracy loss

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

**Remember**: You know this material inside-out. Speak confidently, make eye contact, and show enthusiasm for your work!

**Good luck! 🚀**
