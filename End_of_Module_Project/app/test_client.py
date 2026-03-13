
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.json()}")

def test_analyze(audio_path):
    with open(audio_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        start = time.time()
        response = requests.post(f"{API_URL}/analyze", files=files)
        latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTranscription: {result['transcription']}")
        print(f"Sentiment: {result['sentiment']['label']} ({result['sentiment']['confidence']:.2f})")
        print(f"Summary: {result['summary']}")
        print(f"API Latency: {latency:.2f} ms")
        print(f"Processing Latency: {result['latency_ms']} ms")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_health()
    
    # Test with sample audio
    audio_path = Path("../data/clips/sample.wav")
    if audio_path.exists():
        test_analyze(audio_path)
    else:
        print("Sample audio not found. Place a test file at data/clips/sample.wav")
