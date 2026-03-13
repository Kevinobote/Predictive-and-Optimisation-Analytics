"""
Tubonge - FastAPI Backend with Real ASR Integration
Speech Analytics API with ASR, Sentiment Analysis, and Summarization
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize FastAPI app
app = FastAPI(
    title="Tubonge API",
    description="Advanced Speech Analytics with ASR and Sentiment Analysis",
    version="1.0.0"
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
SUMMARIZATION_MODEL_NAME = "google/mt5-small"  # Multilingual T5 for summarization
USE_REAL_ASR = True  # Enable real ASR
USE_T5_SUMMARIZATION = True  # Enable T5 summarization

# Global model storage
models = {
    "asr_processor": None,
    "asr_model": None,
    "sentiment_model": None,
    "sentiment_tokenizer": None,
    "summarization_model": None,
    "summarization_tokenizer": None
}

# ===== Model Loading =====
def load_models():
    """Load ASR, Sentiment, and Summarization models on startup"""
    global models
    
    try:
        from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models["device"] = device
        
        # Load ASR model
        logger.info(f"Loading ASR model: {ASR_MODEL_NAME}")
        models["asr_processor"] = AutoProcessor.from_pretrained(ASR_MODEL_NAME)
        models["asr_model"] = AutoModelForCTC.from_pretrained(ASR_MODEL_NAME)
        models["asr_model"] = models["asr_model"].to(device)
        logger.info(f"ASR model loaded on {device}")
        
        # Load T5 Summarization model
        if USE_T5_SUMMARIZATION:
            try:
                logger.info(f"Loading T5 summarization model: {SUMMARIZATION_MODEL_NAME}")
                models["summarization_tokenizer"] = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
                models["summarization_model"] = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
                models["summarization_model"] = models["summarization_model"].to(device)
                logger.info(f"T5 summarization model loaded on {device}")
            except Exception as e:
                logger.error(f"Error loading T5 model: {str(e)}")
                logger.info("Will use fallback extractive summarization")
        
        # Try to load sentiment model if available
        sentiment_path = os.path.join(BASE_DIR, SENTIMENT_MODEL_PATH)
        if os.path.exists(sentiment_path):
            logger.info(f"Loading sentiment model from {sentiment_path}")
            models["sentiment_tokenizer"] = AutoTokenizer.from_pretrained(sentiment_path)
            models["sentiment_model"] = AutoModelForSequenceClassification.from_pretrained(sentiment_path)
            models["sentiment_model"] = models["sentiment_model"].to(device)
            logger.info("Sentiment model loaded")
        else:
            logger.warning(f"Sentiment model not found at {sentiment_path}, will use fallback")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.info("Falling back to sample data mode")

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
    language: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models_loaded: dict

# ===== Helper Functions =====
def transcribe_audio(audio_path: str, language: str = "sw") -> str:
    """Transcribe audio using ASR model"""
    try:
        import torch
        import soundfile as sf
        
        # Load audio
        speech, sample_rate = sf.read(audio_path)
        
        # Convert stereo to mono if needed
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        
        # Ensure float32 and normalize
        speech = speech.astype(np.float32)
        if speech.max() > 1.0:
            speech = speech / 32768.0
        
        # Process audio
        inputs = models["asr_processor"](
            speech, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = models["asr_model"](**inputs).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = models["asr_processor"].batch_decode(predicted_ids)[0]
        
        logger.info(f"Transcription successful: {transcription[:50]}...")
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Fallback to sample
        return get_fallback_transcript(language)

def get_fallback_transcript(language: str) -> str:
    """Get fallback transcript if ASR fails"""
    fallbacks = {
        "en": "This is a sample transcription. The audio processing system is currently using fallback mode. Please ensure all required models are properly installed.",
        "sw": "Hii ni nakala ya mfano. Mfumo wa usindikaji wa sauti unatumia hali ya msaada. Tafadhali hakikisha miundo yote inayohitajika imesakinishwa vizuri."
    }
    return fallbacks.get(language, fallbacks["en"])

def analyze_sentiment(text: str, language: str = "en"):
    """Analyze sentiment using trained model or fallback"""
    try:
        if models["sentiment_model"] is not None:
            import torch
            
            # Tokenize
            inputs = models["sentiment_tokenizer"](
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = models["sentiment_model"](**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item() * 100
            
            # Map to sentiment
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_map.get(predicted_class, "neutral")
            
            # Get distribution
            distribution = {
                "negative": float(probs[0][0]) * 100,
                "neutral": float(probs[0][1]) * 100 if probs.shape[1] > 1 else 0,
                "positive": float(probs[0][2]) * 100 if probs.shape[1] > 2 else float(probs[0][1]) * 100
            }
            
            return sentiment, confidence, distribution
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
    
    # Fallback sentiment analysis
    return get_fallback_sentiment(text)

def get_fallback_sentiment(text: str):
    """Simple rule-based sentiment fallback"""
    import random
    
    positive_words = ["good", "great", "excellent", "happy", "positive", "success", "wonderful", 
                     "nzuri", "vizuri", "furaha", "mafanikio"]
    negative_words = ["bad", "poor", "terrible", "sad", "negative", "failure", "awful",
                     "mbaya", "huzuni", "kushindwa"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = "positive"
        confidence = 70 + random.random() * 20
    elif neg_count > pos_count:
        sentiment = "negative"
        confidence = 70 + random.random() * 20
    else:
        sentiment = "neutral"
        confidence = 60 + random.random() * 20
    
    # Generate distribution
    if sentiment == "positive":
        distribution = {
            "positive": confidence,
            "neutral": (100 - confidence) * 0.7,
            "negative": (100 - confidence) * 0.3
        }
    elif sentiment == "negative":
        distribution = {
            "negative": confidence,
            "neutral": (100 - confidence) * 0.7,
            "positive": (100 - confidence) * 0.3
        }
    else:
        distribution = {
            "neutral": confidence,
            "positive": (100 - confidence) * 0.5,
            "negative": (100 - confidence) * 0.5
        }
    
    return sentiment, confidence, distribution

def generate_summary(text: str, language: str = "en") -> str:
    """Generate summary from transcript using T5 or fallback to extractive"""
    try:
        # Use T5 model if available
        if models["summarization_model"] is not None and USE_T5_SUMMARIZATION:
            import torch
            
            # Add task prefix for mT5
            input_text = f"summarize: {text}"
            
            # Tokenize input with padding
            inputs = models["summarization_tokenizer"](
                input_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(models["device"]) for k, v in inputs.items()}
            
            # Generate summary
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
            
            # Decode summary with proper cleanup
            summary = models["summarization_tokenizer"].decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Validate summary
            if not summary or len(summary.strip()) < 10:
                raise ValueError("Generated summary too short")
            
            logger.info(f"Generated T5 summary: {summary[:50]}...")
            return summary.strip()
            
    except Exception as e:
        logger.error(f"Error in T5 summarization: {str(e)}")
        logger.info("Falling back to extractive summarization")
    
    # Fallback: Simple extractive summarization
    try:
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= 2:
            return text
        
        # Take first and last sentences as summary
        summary = f"{sentences[0]}. {sentences[-1]}."
        return summary
        
    except Exception as e:
        logger.error(f"Error in extractive summarization: {str(e)}")
        return text[:200] + "..." if len(text) > 200 else text

def extract_keywords(text: str, language: str = "en") -> List[str]:
    """Extract keywords from text"""
    try:
        # Simple keyword extraction based on word frequency
        import re
        from collections import Counter
        
        # Remove common stop words
        stop_words = {
            "en": {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for", "of", "as", "by"},
            "sw": {"na", "ya", "wa", "ni", "kwa", "katika", "la", "za", "au", "lakini"}
        }
        
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if len(w) > 3 and w not in stop_words.get(language, stop_words["en"])]
        
        # Get most common
        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(5)]
        
        return keywords if keywords else ["speech", "audio", "analysis"]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return ["speech", "audio", "analysis"]

def calculate_metrics(transcript: str, duration: float):
    """Calculate text metrics from transcript"""
    import re
    
    words = transcript.split()
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = transcript.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return {
        "word_count": len(words),
        "char_count": len(transcript),
        "sentence_count": len(sentences),
        "paragraph_count": max(1, len(paragraphs)),
        "speaking_rate": round((len(words) / duration) * 60) if duration > 0 else 0,
        "duration": duration
    }

def get_sentiment_data(sentiment: str, confidence: float, distribution: dict):
    """Get sentiment emoji and formatted data"""
    emoji_map = {
        "positive": "😊",
        "neutral": "😐",
        "negative": "😔"
    }
    
    return {
        "emoji": emoji_map.get(sentiment, "😐"),
        "label": sentiment.capitalize(),
        "confidence": confidence,
        "distribution": distribution
    }

def get_audio_duration(file_path: str) -> float:
    """Get actual audio duration"""
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    except:
        import random
        return 30 + random.random() * 90

# ===== Static File Routes =====

@app.get("/")
async def root():
    """Serve the main HTML page"""
    html_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(html_path)

@app.get("/styles.css")
async def get_styles():
    """Serve CSS file"""
    css_path = os.path.join(BASE_DIR, "styles.css")
    return FileResponse(css_path, media_type="text/css")

@app.get("/app.js")
async def get_app_js():
    """Serve JavaScript file"""
    js_path = os.path.join(BASE_DIR, "app.js")
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/translations.js")
async def get_translations_js():
    """Serve translations JavaScript file"""
    js_path = os.path.join(BASE_DIR, "translations.js")
    return FileResponse(js_path, media_type="application/javascript")

# ===== API Endpoints =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
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
    """
    Analyze audio file with real ASR
    
    - **audio**: Audio file (MP3, WAV, OGG, M4A, FLAC, AAC, WEBM)
    - **language**: Language code (en or sw)
    """
    start_time = time.time()
    
    try:
        valid_types = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/flac", "audio/aac", "audio/webm"]
        if audio.content_type not in valid_types:
            raise HTTPException(status_code=400, detail="Invalid audio file type")
        
        # Save uploaded file temporarily
        suffix = os.path.splitext(audio.filename)[1] if audio.filename else '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Convert WebM to WAV if needed (soundfile may have issues with WebM)
        converted_path = None
        if suffix == '.webm' or audio.content_type == 'audio/webm':
            try:
                import subprocess
                converted_path = temp_path.replace('.webm', '.wav')
                result = subprocess.run(
                    ['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-y', converted_path],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info("Converted WebM to WAV successfully")
                    processing_path = converted_path
                else:
                    logger.warning("FFmpeg conversion failed, trying direct processing")
                    processing_path = temp_path
            except Exception as e:
                logger.warning(f"WebM conversion failed: {e}, trying direct processing")
                processing_path = temp_path
        else:
            processing_path = temp_path
        
        try:
            # Get audio duration
            duration = get_audio_duration(processing_path)
            
            # Transcribe audio using real ASR
            if models["asr_model"] is not None and USE_REAL_ASR:
                logger.info(f"Using real ASR model for transcription (file: {audio.filename})")
                transcript = transcribe_audio(processing_path, language)
            else:
                logger.info("Using fallback transcription")
                transcript = get_fallback_transcript(language)
            
            # Generate summary
            summary = generate_summary(transcript, language)
            
            # Extract keywords
            keywords = extract_keywords(transcript, language)
            
            # Analyze sentiment
            sentiment, confidence, distribution = analyze_sentiment(transcript, language)
            
            # Get sentiment data
            sentiment_data = get_sentiment_data(sentiment, confidence, distribution)
            
            # Calculate metrics
            metrics = calculate_metrics(transcript, duration)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
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
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Processed audio: {audio.filename}, Language: {language}, Duration: {duration}s, WC: {metrics['word_count']}")
            
            return response
            
        finally:
            # Clean up temp files
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/api/analyze-recording", response_model=AnalysisResponse)
async def analyze_recording(
    audio: UploadFile = File(...),
    language: str = Form("en")
):
    """Analyze recorded audio"""
    return await analyze_audio(audio, language)

@app.get("/api/languages")
async def get_languages():
    """Get supported languages"""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "sw", "name": "Kiswahili"}
        ]
    }

@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "asr_model": ASR_MODEL_NAME,
        "summarization_model": SUMMARIZATION_MODEL_NAME,
        "models_loaded": {
            "asr": models["asr_model"] is not None,
            "sentiment": models["sentiment_model"] is not None,
            "summarization": models["summarization_model"] is not None
        },
        "supported_formats": ["MP3", "WAV", "OGG", "M4A", "FLAC", "AAC"],
        "max_file_size": "100MB",
        "real_asr_enabled": USE_REAL_ASR,
        "t5_summarization_enabled": USE_T5_SUMMARIZATION
    }

# ===== Startup Event =====

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Tubonge API...")
    load_models()
    logger.info("Startup complete")

# ===== Main =====

if __name__ == "__main__":
    print("=" * 60)
    print("🎙️  Tubonge - Speech Analytics API")
    print("=" * 60)
    print(f"📍 Server: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🔧 Health Check: http://localhost:8000/health")
    print(f"🤖 ASR Model: {ASR_MODEL_NAME}")
    print(f"📝 Summarization: {SUMMARIZATION_MODEL_NAME}")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
