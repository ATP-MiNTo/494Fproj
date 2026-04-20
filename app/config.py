from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    hf_model_name: str = "timm/mobilenetv4_conv_medium.e500_r224_in1k"
    input_size: int = 224
    max_upload_mb: int = 5
    accepted_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    accepted_content_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp", "image/bmp")
    worker_processes: int = 2
    model_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "model")
    docs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "docs")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def torch_weights_path(self) -> Path:
        return self.model_dir / "model_original.pt"

    @property
    def onnx_path(self) -> Path:
        return self.model_dir / "model.onnx"

    @property
    def quantized_onnx_path(self) -> Path:
        return self.model_dir / "model_quantized.onnx"

    @property
    def benchmark_output_path(self) -> Path:
        return self.docs_dir / "benchmark_results.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
