# Model Artifacts

This folder is expected to contain the exported model files used by the API and benchmark scripts:

- `model_original.pt`
- `model.onnx`
- `model_quantized.onnx`

Generate them with:

```bash
python scripts/export_models.py
```

The Docker image copies this directory so the API can start without exporting the model at runtime.
