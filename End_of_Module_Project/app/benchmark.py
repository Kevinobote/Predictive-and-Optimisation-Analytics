
import requests
import time
import numpy as np
from pathlib import Path

API_URL = "http://localhost:8000"
AUDIO_PATH = Path("../data/clips/sample.wav")
N_REQUESTS = 50

def benchmark_api():
    latencies = []
    
    print(f"Running {N_REQUESTS} requests...")
    for i in range(N_REQUESTS):
        with open(AUDIO_PATH, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            start = time.time()
            response = requests.post(f"{API_URL}/analyze", files=files)
            latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{N_REQUESTS}")
    
    print(f"\nBenchmark Results:")
    print(f"Mean Latency: {np.mean(latencies):.2f} ms")
    print(f"Median Latency: {np.median(latencies):.2f} ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"Throughput: {1000 / np.mean(latencies):.2f} req/sec")

if __name__ == "__main__":
    benchmark_api()
