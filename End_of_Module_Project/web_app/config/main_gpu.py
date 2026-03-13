"""
Tubonge - GPU-Optimized FastAPI Backend for Modal
Enhanced version with A100 GPU acceleration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import tempfile
import time
from datetime import datetime
import logging
import numpy as np

# GPU Optimization Setup
try:
    from gpu_config import setup_gpu_env, GPU_CONFIG
    setup_gpu_env()
except ImportError:
    GPU_CONFIG = {"use_gpu": False}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize FastAPI app
app = FastAPI(
    title="Tubonge API (GPU-Accelerated)",
    description="Advanced Speech Analytics with A100 GPU Acceleration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Configuration =====
ASR_MODEL_NAME = "RareElf/swahili-wav2vec2-asr"
SENTIMENT_MODEL_PATH = "../models/distilbert_sentiment_final"
SUMMARIZATION_MODEL_NAME = "google/mt5-small"
USE_REAL_ASR = True
USE_T5_SUMMARIZATION = True

# Global model storage
models = {
    "asr_processor": None,
    "asr_model": None,
    "sentiment_model": None,
    "sentiment_tokenizer": None,
    "summarization_model": None,
    "summarization_tokenizer": None,
    "device": None
}

# ===== GPU-Optimized Model Loading =====
def load_models():
    """Load models with GPU optimizations"""
    global models
    
    try:
        from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        import torch
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models["device"] = device
        
        logger.info(f"🚀 Loading models on {device}")
        if device == "cuda":
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Determine dtype for GPU optimization
        dtype = torch.float16 if device == "cuda" and GPU_CONFIG.get("torch_dtype") == "float16" else torch.float32
        
        # Load ASR model with GPU optimization
        logger.info(f"📥 Loading ASR model: {ASR_MODEL_NAME}")
        models["asr_processor"] = AutoProcessor.from_pretrained(
            ASR_MODEL_NAME,
            cache_dir=GPU_CONFIG.get("cache_dir")
        )
        models["asr_model"] = AutoModelForCTC.from_pretrained(
            ASR_MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=GPU_CONFIG.get("low_cpu_mem_usage", True),
            cache_dir=GPU_CONFIG.get("cache_dir")
        )
        models["asr_model"] = models["asr_model"].to(device)
        
        # Enable eval mode for inference
        models["asr_model"].eval()
        
        logger.info(f"✅ ASR model loaded on {device} ({dtype})")
        
        # Load T5 Summarization model with GPU optimization
        if USE_T5_SUMMARIZATION:
            try:
                logger.info(f"📥 Loading T5 summarization: {SUMMARIZATION_MODEL_NAME}")
                models["summarization_tokenizer"] = AutoTokenizer.from_pretrained(
                    SUMMARIZATION_MODEL_NAME,
                    cache_dir=GPU_CONFIG.get("cache_dir")
                )
                models["summarization_model"] = AutoModelForSeq2SeqLM.from_pretrained(
                    SUMMARIZATION_MODEL_NAME,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=GPU_CONFIG.get("low_cpu_mem_usage", True),
                    cache_dir=GPU_CONFIG.get("cache_dir")
                )
                models["summarization_model"] = models["summarization_model"].to(device)
                models["summarization_model"].eval()
                logger.info(f"✅ T5 model loaded on {device} ({dtype})")
            except Exception as e:
                logger.error(f"❌ Error loading T5: {str(e)}")
        
        # Try to load sentiment model
        sentiment_path = os.path.join(BASE_DIR, SENTIMENT_MODEL_PATH)
        if os.path.exists(sentiment_path):
            logger.info(f"📥 Loading sentiment model from {sentiment_path}")
            models["sentiment_tokenizer"] = AutoTokenizer.from_pretrained(sentiment_path)
            models["sentiment_model"] = AutoModelForSequenceClassification.from_pretrained(
                sentiment_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=GPU_CONFIG.get("low_cpu_mem_usage", True)
            )
            models["sentiment_model"] = models["sentiment_model"].to(device)
            models["sentiment_model"].eval()
            logger.info(f"✅ Sentiment model loaded on {device}")
        else:
            logger.warning(f"⚠️  Sentiment model not found at {sentiment_path}")
        
        # Warm up models with dummy input (improves first request speed)
        if device == "cuda":
            logger.info("🔥 Warming up models...")
            warmup_models()
            logger.info("✅ Models warmed up")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def warmup_models():
    """Warm up models with dummy inputs for faster first inference"""
    try:
        import torch
        
        # Warm up ASR
        if models["asr_model"] is not None:
            dummy_audio = torch.randn(1, 16000).to(models["device"])
            with torch.no_grad():
                inputs = models["asr_processor"](
                    dummy_audio.cpu().numpy().squeeze(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
                _ = models["asr_model"](**inputs)
        
        # Warm up summarization
        if models["summarization_model"] is not None:
            dummy_text = "summarize: This is a test."
            with torch.no_grad():
                inputs = models["summarization_tokenizer"](
                    dummy_text,
                    return_tensors="pt",
                    max_length=50,
                    truncation=True
                )
                inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
                _ = models["summarization_model"].generate(inputs["input_ids"], max_length=20)
        
        logger.info("✅ Model warmup complete")
    except Exception as e:
        logger.warning(f"⚠️  Warmup failed: {e}")

# ===== Models =====
class AnalysisResponse(BaseModel):
    transcript: str
    summary: str
    keywords: List[str]
    sentiment: str
    sentiment_emoji: str
    sentiment_confidence: float
    sentiment_distribution: dict
    metrics: dict
    processing_time: float
    gpu_used: bool
    language: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    gpu_available: bool
    gpu_name: str
    models_loaded: dict

# ===== Helper Functions =====
def transcribe_audio(audio_path: str, language: str = "sw") -> str:
    """GPU-accelerated transcription"""
    try:
        import torch
        import soundfile as sf
        
        # Load audio
        speech, sample_rate = sf.read(audio_path)
        
        # Convert stereo to mono
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        
        # Normalize
        speech = speech.astype(np.float32)
        if speech.max() > 1.0:
            speech = speech / 32768.0
        
        # Process with GPU
        inputs = models["asr_processor"](
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
        
        # Fast inference with no_grad
        with torch.no_grad():
            logits = models["asr_model"](**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = models["asr_processor"].batch_decode(predicted_ids)[0]
        
        logger.info(f"✅ Transcription: {transcription[:50]}...")
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        return get_fallback_transcript(language)

def generate_summary(text: str, language: str = "en") -> str:
    """GPU-accelerated summarization"""
    try:
        if models["summarization_model"] is not None and USE_T5_SUMMARIZATION:
            import torch
            
            input_text = f"summarize: {text}"
            
            inputs = models["summarization_tokenizer"](
                input_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
            
            # Fast generation with GPU
            with torch.no_grad():
                summary_ids = models["summarization_model"].generate(
                    inputs["input_ids"],
                    max_length=150,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            summary = models["summarization_tokenizer"].decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            if summary and len(summary.strip()) >= 10:
                logger.info(f"✅ Summary: {summary[:50]}...")
                return summary.strip()
            
    except Exception as e:
        logger.error(f"❌ Summarization error: {str(e)}")
    
    # Fallback
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) <= 2:
        return text
    return f"{sentences[0]}. {sentences[-1]}."

def analyze_sentiment(text: str, language: str = "en"):
    """GPU-accelerated sentiment analysis"""
    try:
        if models["sentiment_model"] is not None:
            import torch
            
            inputs = models["sentiment_tokenizer"](
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = models["sentiment_model"](**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item() * 100
            
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_map.get(predicted_class, "neutral")
            
            distribution = {
                "negative": float(probs[0][0]) * 100,
                "neutral": float(probs[0][1]) * 100 if probs.shape[1] > 1 else 0,
                "positive": float(probs[0][2]) * 100 if probs.shape[1] > 2 else float(probs[0][1]) * 100
            }
            
            return sentiment, confidence, distribution
            
    except Exception as e:
        logger.error(f"❌ Sentiment error: {str(e)}")
    
    return get_fallback_sentiment(text)

# Import remaining helper functions from original main.py
def get_fallback_transcript(language: str) -> str:
    fallbacks = {
        "en": "This is a sample transcription. The audio processing system is currently using fallback mode.",
        "sw": "Hii ni nakala ya mfano. Mfumo wa usindikaji wa sauti unatumia hali ya msaada."
    }
    return fallbacks.get(language, fallbacks["en"])

def get_fallback_sentiment(text: str):
    import random
    positive_words = ["good", "great", "excellent", "happy", "nzuri", "vizuri", "furaha"]
    negative_words = ["bad", "poor", "terrible", "sad", "mbaya", "huzuni"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment, confidence = "positive", 75
    elif neg_count > pos_count:
        sentiment, confidence = "negative", 75
    else:
        sentiment, confidence = "neutral", 65
    
    distribution = {
        "positive": confidence if sentiment == "positive" else (100 - confidence) * 0.4,
        "neutral": confidence if sentiment == "neutral" else (100 - confidence) * 0.5,
        "negative": confidence if sentiment == "negative" else (100 - confidence) * 0.4
    }
    
    return sentiment, confidence, distribution

def extract_keywords(text: str, language: str = "en") -> List[str]:
    import re
    from collections import Counter
    
    stop_words = {
        "en": {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for", "of", "as", "by"},
        "sw": {"na", "ya", "wa", "ni", "kwa", "katika", "la", "za", "au", "lakini"}
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if len(w) > 3 and w not in stop_words.get(language, stop_words["en"])]
    
    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(5)]
    
    return keywords if keywords else ["speech", "audio", "analysis"]

def calculate_metrics(transcript: str, duration: float):
    import re
    
    words = transcript.split()
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        "word_count": len(words),
        "char_count": len(transcript),
        "sentence_count": len(sentences),
        "paragraph_count": max(1, len(transcript.split('\n\n'))),
        "speaking_rate": round((len(words) / duration) * 60) if duration > 0 else 0,
        "duration": duration
    }

def get_sentiment_data(sentiment: str, confidence: float, distribution: dict):
    emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😔"}
    return {
        "emoji": emoji_map.get(sentiment, "😐"),
        "label": sentiment.capitalize(),
        "confidence": confidence,
        "distribution": distribution
    }

def get_audio_duration(file_path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    except:
        return 30.0

# ===== Static File Routes =====
@app.get("/")
async def root():
    html_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(html_path)

@app.get("/styles.css")
async def get_styles():
    css_path = os.path.join(BASE_DIR, "styles.css")
    return FileResponse(css_path, media_type="text/css")

@app.get("/app.js")
async def get_app_js():
    js_path = os.path.join(BASE_DIR, "app.js")
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/translations.js")
async def get_translations_js():
    js_path = os.path.join(BASE_DIR, "translations.js")
    return FileResponse(js_path, media_type="application/javascript")

# ===== API Endpoints =====
@app.get("/health", response_model=HealthResponse)
async def health_check():
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0-gpu",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "models_loaded": {
            "asr": models["asr_model"] is not None,
            "sentiment": models["sentiment_model"] is not None,
            "summarization": models["summarization_model"] is not None
        }
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    audio: UploadFile = File(...),
    language: str = Form("en")
):
    """GPU-accelerated audio analysis"""
    start_time = time.time()
    
    try:
        import torch
        
        # Save uploaded file
        suffix = os.path.splitext(audio.filename)[1] if audio.filename else '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Convert WebM if needed
        converted_path = None
        if suffix == '.webm' or audio.content_type == 'audio/webm':
            try:
                import subprocess
                converted_path = temp_path.replace('.webm', '.wav')
                subprocess.run(
                    ['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-y', converted_path],
                    capture_output=True,
                    timeout=30
                )
                processing_path = converted_path if os.path.exists(converted_path) else temp_path
            except:
                processing_path = temp_path
        else:
            processing_path = temp_path
        
        try:
            duration = get_audio_duration(processing_path)
            
            # GPU-accelerated processing
            transcript = transcribe_audio(processing_path, language) if models["asr_model"] else get_fallback_transcript(language)
            summary = generate_summary(transcript, language)
            keywords = extract_keywords(transcript, language)
            sentiment, confidence, distribution = analyze_sentiment(transcript, language)
            sentiment_data = get_sentiment_data(sentiment, confidence, distribution)
            metrics = calculate_metrics(transcript, duration)
            
            processing_time = time.time() - start_time
            
            response = {
                "transcript": transcript,
                "summary": summary,
                "keywords": keywords,
                "sentiment": sentiment,
                "sentiment_emoji": sentiment_data["emoji"],
                "sentiment_confidence": sentiment_data["confidence"],
                "sentiment_distribution": sentiment_data["distribution"],
                "metrics": metrics,
                "processing_time": round(processing_time, 2),
                "gpu_used": torch.cuda.is_available(),
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Processed in {processing_time:.2f}s (GPU: {torch.cuda.is_available()})")
            
            return response
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/api/analyze-recording", response_model=AnalysisResponse)
async def analyze_recording(audio: UploadFile = File(...), language: str = Form("en")):
    return await analyze_audio(audio, language)

@app.get("/api/languages")
async def get_languages():
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "sw", "name": "Kiswahili"}
        ]
    }

@app.get("/api/stats")
async def get_stats():
    import torch
    return {
        "asr_model": ASR_MODEL_NAME,
        "summarization_model": SUMMARIZATION_MODEL_NAME,
        "gpu_enabled": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "models_loaded": {
            "asr": models["asr_model"] is not None,
            "sentiment": models["sentiment_model"] is not None,
            "summarization": models["summarization_model"] is not None
        }
    }

# ===== Startup Event =====
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Tubonge API (GPU-Accelerated)...")
    load_models()
    logger.info("✅ Startup complete")

# ===== Main =====
if __name__ == "__main__":
    print("=" * 60)
    print("🎙️  Tubonge - Speech Analytics API (GPU-Accelerated)")
    print("=" * 60)
    print(f"📍 Server: http://localhost:8000")
    print(f"🚀 GPU: A100")
    print("=" * 60)
    
    uvicorn.run("main_gpu:app", host="0.0.0.0", port=8000, reload=True)
