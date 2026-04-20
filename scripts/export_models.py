from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.model_backend import export_model_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch, ONNX, and quantized ONNX model artifacts.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Override the default model directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.model_dir is not None:
        settings = settings.model_copy(update={"model_dir": args.model_dir})
    export_model_assets(settings)


if __name__ == "__main__":
    main()
