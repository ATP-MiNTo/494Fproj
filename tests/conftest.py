from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import create_app
from app.model_backend import PredictionResult


def make_jpeg_bytes(color: tuple[int, int, int] = (180, 140, 100), size: tuple[int, int] = (64, 64)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
async def async_client() -> AsyncClient:
    async def fake_predictor(image_bytes: bytes) -> PredictionResult:
        return PredictionResult(label="dog", confidence=0.98)

    app = create_app(use_process_pool=False, predictor=fake_predictor)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def dog_image_bytes() -> bytes:
    return make_jpeg_bytes()


@pytest.fixture
def bad_text_bytes() -> bytes:
    return b"not an image"
