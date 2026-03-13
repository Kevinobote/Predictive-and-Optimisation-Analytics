# Tubonge Modal Deployment - File Structure

## Created Files

```
web_app/
├── modal_app.py           # Main Modal deployment (A100 GPU)
├── modal_app_cpu.py       # Alternative CPU-only deployment
├── MODAL_DEPLOY.md        # Detailed deployment guide
└── deploy_modal.sh        # Quick deployment script
```

## Quick Start

### 1. Activate Environment
```bash
conda activate audio_ml
```

### 2. Deploy (Choose One)

**Option A: Use deployment script (easiest)**
```bash
cd "/home/obote/Documents/Strathmore DSA/Module 5/Predictive and Optimisation Analytics/End_of_Module_Project/web_app"
./deploy_modal.sh
```

**Option B: Manual deployment**
```bash
# With A100 GPU (recommended for demo)
modal deploy modal_app.py

# Or CPU-only (for testing)
modal deploy modal_app_cpu.py
```

### 3. Get Your URL
After deployment, Modal provides a URL like:
```
https://your-username--tubonge-fastapi-app.modal.run
```

### 4. Update Frontend (Optional)
If you want to use the deployed API, update `app.js`:
```javascript
const AppState = {
    apiBaseUrl: 'https://your-modal-url-here.modal.run'
};
```

### 5. Stop When Done (Important!)
```bash
modal app stop tubonge
```

## Deployment Options Comparison

| Feature | A100 GPU | CPU-only |
|---------|----------|----------|
| Speed | Very Fast | Slower |
| Cost | Higher per hour | Lower per hour |
| Best for | Live demos | Testing |
| Concurrent requests | 20 | 5 |
| Model loading | Fast | Slower |

## Cost Management Tips

1. **Use `modal serve` for development** - Runs locally, no cloud costs
2. **Deploy only for demos** - Don't leave running 24/7
3. **Stop after demos** - `modal app stop tubonge`
4. **A100 is fast** - Processes requests quickly, reducing total GPU time
5. **Container idle timeout** - Set to 300s to balance cold starts vs. idle costs

## File Descriptions

### modal_app.py
- Main deployment configuration
- Uses A100 GPU
- Optimized for your available hardware
- 20 concurrent requests
- 5-minute idle timeout

### modal_app_cpu.py
- CPU-only alternative
- Much cheaper
- Good for testing
- Slower inference

### MODAL_DEPLOY.md
- Complete deployment guide
- All commands and options
- Monitoring and troubleshooting

### deploy_modal.sh
- Interactive deployment script
- Handles authentication
- Easy menu-driven interface

## Common Commands

```bash
# Deploy
modal deploy modal_app.py

# Check status
modal app list

# View logs
modal app logs tubonge --follow

# Check GPU usage
modal app stats tubonge

# Stop (save costs!)
modal app stop tubonge

# Delete completely
modal app delete tubonge
```

## Next Steps

1. ✅ Files created
2. ⏭️ Run `./deploy_modal.sh` or `modal deploy modal_app.py`
3. ⏭️ Get your deployment URL
4. ⏭️ Test the API
5. ⏭️ Stop when done to save costs

## Support

- Modal Docs: https://modal.com/docs
- Check logs: `modal app logs tubonge`
- Test locally first: `modal serve modal_app.py`
