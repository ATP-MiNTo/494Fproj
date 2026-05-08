from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from app.config import Settings, get_settings
from app.image_validation import read_and_validate_image
from app.model_backend import predict_image_bytes   # ✅ เอาแค่นี้พอ
from app.schemas import PredictionResponse


def _predict_in_worker(image_bytes: bytes, settings_dict: dict) -> dict:
    settings = Settings(**settings_dict)
    result = predict_image_bytes(image_bytes, settings)
    return {"label": result.label, "confidence": result.confidence}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    executor = ProcessPoolExecutor(max_workers=settings.worker_processes)
    app.state.executor = executor
    app.state.settings = settings

    async def process_pool_predictor(image_bytes: bytes) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            _predict_in_worker,
            image_bytes,
            settings.model_dump(mode="json"),
        )

    app.state.predictor = process_pool_predictor

    try:
        yield
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def create_app(use_process_pool: bool = True, predictor=None) -> FastAPI:
    app = FastAPI(
        title="Image Classifier API",
        version="1.0.0",
        lifespan=lifespan if use_process_pool else None,
    )

    if not use_process_pool:
        app.state.settings = get_settings()
        app.state.predictor = predictor or (
            lambda image_bytes: {
                "label": predict_image_bytes(image_bytes, app.state.settings).label,
                "confidence": predict_image_bytes(image_bytes, app.state.settings).confidence,
            }
        )
    elif predictor is not None:
        app.state.predictor = predictor

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        settings = request.app.state.settings

        image_bytes = await read_and_validate_image(file, settings)

        predictor_callable = getattr(request.app.state, "predictor", None)
        if predictor_callable is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Predictor is not initialized",
            )

        # async / sync รองรับทั้งคู่
        if asyncio.iscoroutinefunction(predictor_callable):
            result = await asyncio.wait_for(predictor_callable(image_bytes), timeout=10)
        else:
            result = predictor_callable(image_bytes)

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app(use_process_pool=False)


# ✅ สำคัญมากสำหรับ Windows
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()