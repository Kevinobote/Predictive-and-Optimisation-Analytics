
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path
import tempfile
import time

app = FastAPI(title="Kiswahili Speech Analytics API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - handle both local and Docker paths
if Path("static").exists():
    static_path = Path("static")
elif Path("app/static").exists():
    static_path = Path("app/static")
else:
    raise FileNotFoundError("Static directory not found")

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Load models at startup
print("Loading models...")

# ASR Model
asr_processor = AutoProcessor.from_pretrained("RareElf/swahili-wav2vec2-asr")
asr_model = AutoModelForCTC.from_pretrained("RareElf/swahili-wav2vec2-asr")
device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = asr_model.to(device)

# Sentiment Model - handle both local and Docker paths
if Path("models/distilbert_sentiment_final").exists():
    MODEL_DIR = Path("models/distilbert_sentiment_final")
elif Path("../models/distilbert_sentiment_final").exists():
    MODEL_DIR = Path("../models/distilbert_sentiment_final")
else:
    raise FileNotFoundError("Model directory not found. Check models/distilbert_sentiment_final path.")

sentiment_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
sentiment_model = sentiment_model.to(device)

# T5 Summarization Model
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = t5_model.to(device)
    summarizer_available = True
    print("T5 summarization model loaded")
except Exception as e:
    summarizer_available = False
    print(f"T5 model not available: {e}")

print(f"Models loaded on {device}")

@app.get("/")
async def root():
    return {"message": "Kiswahili Speech Analytics API", "status": "active"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file: ASR → Sentiment → Summarization
    """
    start_time = time.time()
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Step 1: ASR
        speech, sr = librosa.load(tmp_path, sr=16000)
        inputs = asr_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = asr_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_processor.batch_decode(predicted_ids)[0]
        
        # Step 2: Sentiment
        sentiment_inputs = sentiment_tokenizer(transcription, return_tensors="pt", padding=True, truncation=True, max_length=128)
        sentiment_inputs = {k: v.to(device) for k, v in sentiment_inputs.items()}
        
        with torch.no_grad():
            sentiment_outputs = sentiment_model(**sentiment_inputs)
        sentiment_pred = torch.argmax(sentiment_outputs.logits, dim=1).item()
        sentiment_label = "positive" if sentiment_pred == 1 else "negative"
        sentiment_score = torch.softmax(sentiment_outputs.logits, dim=1)[0][sentiment_pred].item()
        
        # Step 3: T5 Summarization with fallback to extractive
        word_count = len(transcription.split())
        
        if word_count <= 15:
            # Short text - return as is
            summary = transcription
        elif word_count > 15 and summarizer_available:
            # Use T5 for summarization
            try:
                input_text = f"summarize: {transcription}"
                inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                summary_ids = t5_model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    min_length=20,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"T5 summarization failed: {e}")
                # Fallback to extractive
                sentences = [s.strip() for s in transcription.split('.') if s.strip()]
                if len(sentences) <= 2:
                    summary = transcription
                else:
                    summary = f"{sentences[0]}. {sentences[-1]}."
        else:
            # Fallback extractive summarization
            sentences = [s.strip() for s in transcription.split('.') if s.strip()]
            if len(sentences) <= 2:
                summary = transcription
            else:
                mid_idx = len(sentences) // 2
                summary = f"{sentences[0]}. {sentences[mid_idx]}. {sentences[-1]}."
        
        # Ensure summary isn't too long
        if len(summary) > 250:
            summary = summary[:247] + "..."
        
        # Clean up
        Path(tmp_path).unlink()
        
        latency = (time.time() - start_time) * 1000
        
        return JSONResponse({
            "transcription": transcription,
            "sentiment": {
                "label": sentiment_label,
                "confidence": float(sentiment_score)
            },
            "summary": summary,
            "latency_ms": round(latency, 2),
            "audio_duration_sec": round(len(speech) / sr, 2)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Alias for /analyze endpoint
    """
    return await analyze_audio(file)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
