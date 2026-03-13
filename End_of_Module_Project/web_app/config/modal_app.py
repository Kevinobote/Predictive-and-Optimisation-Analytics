"""
Tubonge - Modal Deployment (A100 GPU Optimized)
Maximum speed deployment with GPU acceleration
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("tubonge")

# Get the directory containing this file
web_app_dir = Path(__file__).parent

# Create image with dependencies optimized for GPU
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "pydantic==2.5.0",
        "transformers>=4.35.0",
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "numpy>=1.24.3",
        "scipy>=1.11.4",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "accelerate>=0.25.0",
    ])
    .apt_install(["ffmpeg"])
    .add_local_file(str(web_app_dir / "main.py"), "/root/main.py")
    .add_local_file(str(web_app_dir / "index.html"), "/root/index.html")
    .add_local_file(str(web_app_dir / "styles.css"), "/root/styles.css")
    .add_local_file(str(web_app_dir / "app.js"), "/root/app.js")
    .add_local_file(str(web_app_dir / "translations.js"), "/root/translations.js")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    memory=16384,
    max_containers=20,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    
    from main import app as fastapi_app
    return fastapi_app
