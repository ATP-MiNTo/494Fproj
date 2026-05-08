from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from app.config import Settings


class ImageUploadMetadata(BaseModel):
    filename: str
    content_type: str
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_metadata(self) -> "ImageUploadMetadata":
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        allowed_content_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
        suffix = Path(self.filename).suffix.lower()
        if suffix not in allowed_extensions:
            raise ValueError("Unsupported file type")
        if self.content_type not in allowed_content_types:
            raise ValueError("Unsupported file type")
        return self


async def read_and_validate_image(file: UploadFile, settings: Settings) -> bytes:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File name is required")

    data = await file.read(settings.max_upload_bytes + 1)
    metadata = ImageUploadMetadata(
        filename=file.filename,
        content_type=file.content_type or "",
        size_bytes=len(data),
    )

    if metadata.size_bytes > settings.max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds 5 MB limit")

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unable to decode image") from exc

    return data
