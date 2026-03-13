"""
GPU Optimization Configuration for Modal Deployment
Add this to enable faster processing with A100
"""

import os

# GPU Optimization Settings
GPU_CONFIG = {
    # Enable GPU acceleration
    "use_gpu": True,
    
    # Model loading optimizations
    "torch_dtype": "float16",  # Use FP16 for faster inference on A100
    "device_map": "auto",  # Automatic device mapping
    
    # Batch processing
    "batch_size": 8,  # Process multiple requests in parallel
    
    # Model caching
    "cache_dir": "/tmp/model_cache",  # Cache models to avoid re-downloading
    
    # Inference optimizations
    "use_flash_attention": True,  # Faster attention mechanism
    "torch_compile": True,  # PyTorch 2.0 compilation for speed
    
    # Memory optimizations
    "gradient_checkpointing": False,  # Not needed for inference
    "low_cpu_mem_usage": True,  # Reduce CPU memory usage
}

# Environment variables for GPU optimization
def setup_gpu_env():
    """Set environment variables for optimal GPU performance"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "8"
    
    # Enable TF32 for A100 (faster matmul)
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"✅ GPU Optimizations Enabled: {torch.cuda.get_device_name(0)}")
        print(f"   - TF32: Enabled")
        print(f"   - cuDNN Benchmark: Enabled")
        print(f"   - Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
