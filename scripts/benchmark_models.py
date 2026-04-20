from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.model_backend import (
    benchmark_predictions,
    get_onnx_classifier,
    get_torch_classifier,
    serialize_benchmark_report,
)


def load_sample_image(image_path: Path) -> bytes:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        from io import BytesIO

        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG")
        return buffer.getvalue()


def model_size_mb(model_path: Path) -> float:
    return model_path.stat().st_size / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark original, ONNX, and quantized model variants.")
    parser.add_argument("--image", type=Path, required=True, help="Sample image used for benchmarking")
    args = parser.parse_args()

    settings = get_settings()
    image_bytes = load_sample_image(args.image)

    torch_backend = get_torch_classifier(settings.hf_model_name)
    onnx_backend = get_onnx_classifier(str(settings.onnx_path), settings.hf_model_name)
    quantized_backend = get_onnx_classifier(str(settings.quantized_onnx_path), settings.hf_model_name)

    rows = [
        {
            "variant": "Original",
            "model_size_mb": model_size_mb(settings.torch_weights_path),
            **benchmark_predictions(torch_backend.predict, image_bytes),
        },
        {
            "variant": "ONNX",
            "model_size_mb": model_size_mb(settings.onnx_path),
            **benchmark_predictions(onnx_backend.predict, image_bytes),
        },
        {
            "variant": "Quantized",
            "model_size_mb": model_size_mb(settings.quantized_onnx_path),
            **benchmark_predictions(quantized_backend.predict, image_bytes),
        },
    ]

    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.benchmark_output_path.write_text(serialize_benchmark_report(rows), encoding="utf-8")
    print(serialize_benchmark_report(rows))


if __name__ == "__main__":
    main()
