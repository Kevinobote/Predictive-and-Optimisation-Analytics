"""
Tubonge - Modal Deployment (CPU-only)
Minimal cost deployment without GPU
Use this if GPU costs are too high for demo
"""

import modal

app = modal.App("tubonge-cpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "pydantic==2.5.0",
        "transformers>=4.35.0",
        "torch>=2.2.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "numpy>=1.24.3",
        "scipy>=1.11.4",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    ])
    .apt_install(["ffmpeg"])
)

@app.function(
    image=image,
    cpu=2,  # CPU-only - much cheaper
    memory=4096,
    timeout=600,
    allow_concurrent_inputs=5,
)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from main import app as fastapi_app
    return fastapi_app
