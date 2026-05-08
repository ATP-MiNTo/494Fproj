# FastAPI Image Classifier

A small CPU-friendly image classification service built around a Hugging Face model, exported to ONNX, and quantized for faster inference.

## What is included

- Hugging Face source model: `timm/mobilenetv4_conv_medium.e500_r224_in1k`
- Optimization scripts for PyTorch, ONNX, and INT8 quantized ONNX
- Async FastAPI `POST /predict` endpoint
- ProcessPoolExecutor-based CPU offload
- Pytest coverage for API behavior and validation errors
- Dockerfile for container packaging
- GitHub Actions workflow for CI and Hugging Face Spaces deployment

## Project structure

- `app/` FastAPI application and model helpers
- `model/` exported model artifacts
- `scripts/` export, benchmark, and deployment utilities
- `tests/` pytest suite
- `.github/workflows/` CI/CD workflow

## Setup

```bash
python -m pip install -r requirements.txt
```

## Export model artifacts

```bash
python scripts/export_models.py
```

This generates:

- `model/model_original.pt`
- `model/model.onnx`
- `model/model_quantized.onnx`

## Run the API locally

```bash
uvicorn app.main:app --reload
```

## Predict with curl

```bash
curl -X POST \
  -F "file=@cat.jpg" \
  http://127.0.0.1:8000/predict
```

Example response:

```json
{
  "label": "cat",
  "confidence": 0.98
}
```

## Cloud API Usage

If you deployed the app to Hugging Face Spaces, replace `<username>/<space-name>` with your Space URL:

```bash
https://<username>-<space-name>.hf.space
```

Health check:

```bash
curl https://<username>-<space-name>.hf.space/healthz
```

Prediction:

```bash
curl -X POST \
  -F "file=@cat.jpg" \
  https://<username>-<space-name>.hf.space/predict
```

If your Space is private, add a bearer token:

```bash
curl -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -F "file=@cat.jpg" \
  https://<username>-<space-name>.hf.space/predict
```

## Run tests

```bash
pytest
```

## Docker

```bash
docker build -t image-classifier .
docker run -p 8000:8000 image-classifier
```

## Benchmarking

Run the benchmark script after exporting the models:

```bash
python scripts/benchmark_models.py --image path/to/sample.jpg
```

The results are written to `docs/benchmark_results.json`.

## Hugging Face Spaces deployment

Set these secrets before the deployment job runs:

- `HF_TOKEN`
- `HF_SPACE_REPO`

The workflow uploads the repository to a Docker-based Space.

