from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_predict_endpoint_returns_json(async_client, dog_image_bytes):
    response = await async_client.post(
        "/predict",
        files={"file": ("dog.jpg", dog_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "dog"
    assert payload["confidence"] == 0.98


@pytest.mark.asyncio
async def test_invalid_file_type_returns_415(async_client, bad_text_bytes):
    response = await async_client.post(
        "/predict",
        files={"file": ("notes.txt", bad_text_bytes, "text/plain")},
    )

    assert response.status_code == 415


@pytest.mark.asyncio
async def test_corrupted_image_returns_422(async_client, bad_text_bytes):
    response = await async_client.post(
        "/predict",
        files={"file": ("broken.jpg", bad_text_bytes, "image/jpeg")},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_oversized_file_returns_413(async_client):
    payload = b"x" * (5 * 1024 * 1024 + 1)
    response = await async_client.post(
        "/predict",
        files={"file": ("large.jpg", payload, "image/jpeg")},
    )

    assert response.status_code == 413
