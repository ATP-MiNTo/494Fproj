# Deployment Guide

## Local Development

### 1. Setup

```bash
# Clone repository
git clone <repository-url>
cd project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Export Models

```bash
python scripts/export_models_simple.py
```

This generates optimized model files in `model/`:
- `model_original.pt` - PyTorch checkpoint
- `model.onnx` - ONNX format (primary)
- `model_quantized.onnx` - INT8 quantized (attempted, fallback handled)

### 3. Run Locally

```bash
uvicorn app.main:app --reload
```

API available at: `http://127.0.0.1:8000`

### 4. Test Endpoints

**Health Check:**
```bash
curl http://127.0.0.1:8000/healthz
```

**Prediction:**
```bash
curl -X POST \
  -F "file=@image.jpg" \
  http://127.0.0.1:8000/predict
```

---

## Docker Deployment

### Build Image

```bash
docker build -t image-classifier .
```

### Run Container

```bash
docker run -p 8000:8000 image-classifier
```

### Verify

```bash
curl http://localhost:8000/healthz
```

---

## Hugging Face Spaces Deployment

### Prerequisites

1. **Hugging Face Account**: https://huggingface.co/
2. **Create Space**: 
   - Go to https://huggingface.co/new-space
   - Choose "Docker" as the space type
   - Set name and visibility

3. **Create Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create "write" token
   - Copy token value

### Manual Deployment

```bash
# Clone HF Spaces repo
git clone https://huggingface.co/spaces/<username>/<space-name> hf-deploy
cd hf-deploy

# Copy project files
cp -r ../app .
cp -r ../model .
cp ../Dockerfile .
cp ../requirements.txt .
cp ../pytest.ini .

# Commit and push
git add .
git commit -m "Deploy image classifier"
git push https://oauth2:<HF_TOKEN>@huggingface.co/spaces/<username>/<space-name>.git
```

### Automatic Deployment (CI/CD)

1. **Set GitHub Secrets**:
   - Go to GitHub repo → Settings → Secrets
   - Add `HF_TOKEN`: Your Hugging Face write token
   - Add `HF_REPO`: `<username>/<space-name>`

2. **Workflow Triggers**:
   - On every push to `main` branch
   - Runs tests → builds Docker → deploys to HF Spaces
   - Only deploys if all tests pass (100% pass rate)

### Monitor Deployment

- **GitHub Actions**: https://github.com/your-repo/actions
- **HF Spaces**: https://huggingface.co/spaces/your-username/your-space

---

## Testing

### Unit Tests

```bash
pytest
```

Tests validate:
- Endpoint returns correct JSON structure
- Invalid files return appropriate HTTP codes
- Model predictions are reasonable

### Load Testing

```bash
# Install Apache Bench
# macOS: brew install httpd
# Ubuntu: sudo apt-get install apache2-utils
# Windows: Use JMeter or similar

# Single request
ab -n 1 -c 1 http://localhost:8000/predict

# Load test: 100 requests, 5 concurrent
ab -n 100 -c 5 -p image.jpg -T image/jpeg http://localhost:8000/predict
```

---

## Troubleshooting

### Models not found

```bash
python scripts/export_models_simple.py
```

### Permission denied (Linux/Mac)

```bash
chmod +x scripts/*.py
```

### Port already in use

```bash
# Change port
uvicorn app.main:app --port 8001
```

### Out of memory (Docker)

Increase Docker memory limit in preferences.

---

## Performance Tuning

### CPU Optimization

- **ProcessPoolExecutor Workers**: Edit `app/main.py` max_workers
- Default: 4 workers (adjust based on CPU cores)

### Memory Optimization

- Model uses ONNX (lighter than PyTorch)
- Image validation prevents huge uploads (5MB max)

### Inference Speed

- ONNX model: ~400-600ms per image (CPU)
- ProcessPoolExecutor enables concurrent requests

---

## Production Checklist

- [ ] Model files exported and verified
- [ ] All tests passing locally
- [ ] Docker image builds successfully
- [ ] GitHub secrets configured for CI/CD
- [ ] HF Spaces token valid
- [ ] Environment variables set (if needed)
- [ ] Load tests show acceptable throughput
- [ ] Error responses properly formatted

---

**Last Updated**: May 9, 2026
