# Modal Deployment Guide

## Setup

1. **Activate your conda environment:**
```bash
conda activate audio_ml
```

2. **Install Modal (if not already installed):**
```bash
pip install modal
```

3. **Authenticate with Modal:**
```bash
modal token new
```

## Deploy

### Deploy to Modal:
```bash
cd "/home/obote/Documents/Strathmore DSA/Module 5/Predictive and Optimisation Analytics/End_of_Module_Project/web_app"
modal deploy modal_app.py
```

### Run locally with Modal (for testing):
```bash
modal serve modal_app.py
```

## Configuration

- **GPU**: A100 (your available GPU - excellent performance)
- **Timeout**: 600 seconds
- **Concurrent requests**: 20
- **Container idle timeout**: 300s (keeps container warm to reduce cold starts)

## GPU Usage Tips

### For Demo (Cost-Conscious):
- Use `modal serve` for development (runs locally)
- Deploy only when showing the demo
- Stop the app when not in use: `modal app stop tubonge`
- A100 processes requests very fast, reducing total GPU time

### Alternative: CPU-only for testing
If you want to test without GPU:
```bash
modal deploy modal_app_cpu.py
```

## Cost Optimization

The A100 deployment:
- **On-demand billing** - Only charged when processing requests
- **Auto-scaling** - Scales to zero when idle
- **Fast inference** - A100 processes audio much faster than smaller GPUs
- **Container idle timeout** - Keeps warm for 5 min to avoid cold starts during demos

## Access

After deployment, Modal will provide a URL like:
```
https://your-username--tubonge-fastapi-app.modal.run
```

Update your frontend `app.js` to use this URL:
```javascript
const AppState = {
    apiBaseUrl: 'https://your-username--tubonge-fastapi-app.modal.run'
};
```

## Monitor

View logs and metrics:
```bash
modal app logs tubonge
```

Check GPU usage:
```bash
modal app stats tubonge
```

## Stop/Remove

Stop the app when not demoing:
```bash
modal app stop tubonge
```

Remove completely:
```bash
modal app delete tubonge
```

## Quick Commands

```bash
# Deploy
modal deploy modal_app.py

# Check status
modal app list

# View logs
modal app logs tubonge --follow

# Stop (important for cost savings!)
modal app stop tubonge
```
