# Deployment

Docker deployment configuration files.

## Files

- **Dockerfile** - Docker image definition
- **docker-compose.yml** - Docker Compose configuration
- **.dockerignore** - Files to exclude from Docker build

## Usage

### Build Docker Image
```bash
docker build -t tubonge-app .
```

### Run with Docker Compose
```bash
docker-compose up
```

### Run Docker Container
```bash
docker run -p 8000:8000 tubonge-app
```

## Configuration

The Docker setup includes:
- Python 3.9+ runtime
- All required dependencies from requirements.txt
- FastAPI application
- Audio processing libraries (librosa, soundfile)
- ML models (transformers, torch)

## Environment Variables

Set these in docker-compose.yml or pass with `-e`:
- `PORT` - Server port (default: 8000)
- `HOST` - Server host (default: 0.0.0.0)

## Production Deployment

For production deployment, see:
- `../docs/project_docs/DEPLOYMENT.md` - General deployment guide
- `../web_app/docs/MODAL_DEPLOY.md` - Modal serverless deployment
- `../docs/project_docs/EDGE_DEPLOYMENT.md` - Edge device deployment

## Notes

- Docker deployment is for local/traditional hosting
- For serverless deployment, use Modal (see web_app/config/modal_app.py)
- For edge devices, see EDGE_DEPLOYMENT.md for optimization strategies
