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

from .config import Settings, get_settings
from .label_map import coarse_label


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float


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
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        top_index = int(torch.argmax(probabilities).item())
        raw_label = self.id2label.get(top_index, str(top_index))
        return PredictionResult(
            label=coarse_label(raw_label),
            confidence=float(probabilities[top_index].item()),
        )

    def export_checkpoint(self, output_path: Path, model_name: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": model_name,
            "state_dict": self.model.state_dict(),
            "id2label": self.id2label,
        }
        torch.save(payload, output_path)

    def export_onnx(self, output_path: Path, input_size: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_image = Image.new("RGB", (input_size, input_size), color=(255, 255, 255))
        dummy_inputs = self.processor(images=dummy_image, return_tensors="pt")["pixel_values"]
        wrapper = _TorchExportWrapper(self.model)
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )


class OnnxImageClassifier:
    def __init__(self, model_path: Path, model_name: str) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.id2label = AutoConfig.from_pretrained(model_name).id2label
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
        logits = self.session.run(None, {self.input_name: pixel_values})[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities = probabilities / probabilities.sum()
        top_index = int(np.argmax(probabilities))
        raw_label = self.id2label.get(top_index, str(top_index))
        return PredictionResult(label=coarse_label(raw_label), confidence=float(probabilities[top_index]))


@lru_cache(maxsize=1)
def get_torch_classifier(model_name: str | None = None) -> TorchImageClassifier:
    settings = get_settings()
    return TorchImageClassifier(model_name or settings.hf_model_name)


@lru_cache(maxsize=1)
def get_onnx_classifier(model_path: str, model_name: str) -> OnnxImageClassifier:
    return OnnxImageClassifier(Path(model_path), model_name)


def get_best_classifier(settings: Settings | None = None) -> Callable[[bytes], PredictionResult]:
    config = settings or get_settings()
    if config.quantized_onnx_path.exists():
        backend = get_onnx_classifier(str(config.quantized_onnx_path), config.hf_model_name)
        return backend.predict
    if config.onnx_path.exists():
        backend = get_onnx_classifier(str(config.onnx_path), config.hf_model_name)
        return backend.predict
    backend = get_torch_classifier(config.hf_model_name)
    return backend.predict


def predict_image_bytes(image_bytes: bytes, settings: Settings | None = None) -> PredictionResult:
    classifier = get_best_classifier(settings)
    return classifier(image_bytes)


def export_model_assets(settings: Settings | None = None) -> None:
    config = settings or get_settings()
    torch_classifier = get_torch_classifier(config.hf_model_name)
    torch_classifier.export_checkpoint(config.torch_weights_path, config.hf_model_name)
    torch_classifier.export_onnx(config.onnx_path, config.input_size)
    
    # Try unsigned INT8 quantization
    try:
        quantize_dynamic(str(config.onnx_path), str(config.quantized_onnx_path), weight_type=QuantType.QUInt8)
        print("✓ QUInt8 quantization successful")
    except Exception as e:
        print(f"⚠ Quantization failed: {type(e).__name__} - {str(e)[:100]}")


def benchmark_predictions(
    predictor: Callable[[bytes], PredictionResult],
    image_bytes: bytes,
    runs: int = 20,
    warmup_runs: int = 5,
) -> dict[str, float]:
    for _ in range(warmup_runs):
        predictor(image_bytes)

    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        predictor(image_bytes)
        timings.append(time.perf_counter() - start)

    average_seconds = statistics.mean(timings)
    return {
        "latency_ms": statistics.median(timings) * 1000.0,
        "throughput_rps": 1.0 / average_seconds if average_seconds else 0.0,
    }


def serialize_benchmark_report(rows: list[dict[str, float | str]]) -> str:
    return json.dumps(rows, indent=2)
