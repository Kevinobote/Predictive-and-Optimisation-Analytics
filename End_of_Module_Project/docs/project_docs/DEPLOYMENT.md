
# Kiswahili Speech Analytics API - Deployment Guide

## Local Deployment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Server
```bash
cd app
python main.py
```

### 3. Test API
```bash
python test_client.py
```

## Docker Deployment

### 1. Build Image
```bash
docker build -t kiswahili-speech-api .
```

### 2. Run Container
```bash
docker run -p 8000:8000 kiswahili-speech-api
```

## API Endpoints

### GET /
Root endpoint

### GET /health
Health check

### POST /analyze
Analyze audio file
- Input: Audio file (WAV format)
- Output: JSON with transcription, sentiment, summary

## Performance Optimization

### 1. Use Quantized Models
Replace FP32 models with INT8 versions for 2x speedup

### 2. Batch Processing
Process multiple requests in batches

### 3. Caching
Cache model outputs for repeated requests

### 4. GPU Acceleration
Deploy on GPU-enabled instances for 5-10x speedup

## Edge Deployment (Raspberry Pi)

### Requirements
- Raspberry Pi 4 (4GB+ RAM)
- Use quantized models only
- Disable summarization for lower latency

### Installation
```bash
# Install PyTorch for ARM
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### Expected Performance
- Latency: 1-2 seconds per request
- Memory: ~1.5GB RAM
- Throughput: 1-2 requests/second

## Production Considerations

1. **Load Balancing**: Use Nginx or AWS ALB
2. **Monitoring**: Implement Prometheus + Grafana
3. **Logging**: Use structured logging (JSON)
4. **Authentication**: Add API key validation
5. **Rate Limiting**: Prevent abuse
6. **HTTPS**: Use SSL certificates
