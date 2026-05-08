from __future__ import annotations

import io
import json
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

from app.config import Settings, get_settings
from app.label_map import coarse_label


# =========================
# RESULT TYPE
# =========================
@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float


# =========================
# TORCH MODEL
# =========================
class _TorchExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


class TorchImageClassifier:
    def __init__(self, model_name: str) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        idx = int(torch.argmax(probs).item())
        raw_label = self.id2label.get(idx, str(idx))

        return PredictionResult(
            label=coarse_label(raw_label),
            confidence=float(probs[idx].item()),
        )

    def export_onnx(self, output_path: Path, input_size: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = Image.new("RGB", (input_size, input_size))
        inputs = self.processor(images=dummy, return_tensors="pt")["pixel_values"]

        wrapper = _TorchExportWrapper(self.model)

        torch.onnx.export(
            wrapper,
            inputs,
            output_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )


# =========================
# ONNX MODEL
# =========================
class OnnxImageClassifier:
    def __init__(self, model_path: Path, model_name: str) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.id2label = AutoConfig.from_pretrained(model_name).id2label
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="np")

        logits = self.session.run(None, {
            self.input_name: inputs["pixel_values"].astype(np.float32)
        })[0][0]

        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()

        idx = int(np.argmax(probs))
        raw_label = self.id2label.get(idx, str(idx))

        return PredictionResult(
            label=coarse_label(raw_label),
            confidence=float(probs[idx]),
        )


# =========================
# LOADERS
# =========================
@lru_cache(maxsize=1)
def get_torch_classifier(model_name: str | None = None):
    settings = get_settings()
    return TorchImageClassifier(model_name or settings.hf_model_name)


@lru_cache(maxsize=1)
def get_onnx_classifier(model_path: str, model_name: str):
    return OnnxImageClassifier(Path(model_path), model_name)


# =========================
# SELECT BACKEND
# =========================
def get_best_classifier(settings: Settings | None = None) -> Callable[[bytes], PredictionResult]:
    config = settings or get_settings()

    if config.quantized_onnx_path.exists():
        return get_onnx_classifier(str(config.quantized_onnx_path), config.hf_model_name).predict

    if config.onnx_path.exists():
        return get_onnx_classifier(str(config.onnx_path), config.hf_model_name).predict

    return get_torch_classifier(config.hf_model_name).predict


def predict_image_bytes(image_bytes: bytes, settings: Settings | None = None) -> PredictionResult:
    return get_best_classifier(settings)(image_bytes)


# =========================
# EXPORT + QUANTIZE
# =========================
def export_model_assets(settings: Settings | None = None) -> None:
    config = settings or get_settings()
    config.model_dir.mkdir(parents=True, exist_ok=True)

    model = get_torch_classifier(config.hf_model_name)

    if not config.onnx_path.exists():
        model.export_onnx(config.onnx_path, config.input_size)

    if not config.quantized_onnx_path.exists():
        quantize_dynamic(
            str(config.onnx_path),
            str(config.quantized_onnx_path),
            weight_type=QuantType.QInt8,
        )


# =========================
# BENCHMARK
# =========================
def benchmark_predictions(
    predictor: Callable[[bytes], PredictionResult],
    image_bytes: bytes,
    runs: int = 20,
    warmup_runs: int = 5,
):
    for _ in range(warmup_runs):
        predictor(image_bytes)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predictor(image_bytes)
        times.append(time.perf_counter() - t0)

    return {
        "latency_ms": statistics.median(times) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "throughput_rps": 1 / statistics.mean(times),
    }


def serialize_benchmark_report(rows):
    return json.dumps(rows, indent=2)