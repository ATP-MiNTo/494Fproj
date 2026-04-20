from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from .config import Settings, get_settings
from .image_validation import read_and_validate_image
from .model_backend import PredictionResult, predict_image_bytes
from .schemas import PredictionResponse


def _predict_in_worker(image_bytes: bytes, settings_dict: dict[str, str]) -> PredictionResult:
    settings = Settings(**settings_dict)
    return predict_image_bytes(image_bytes, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    executor = ProcessPoolExecutor(max_workers=settings.worker_processes)
    app.state.executor = executor
    app.state.settings = settings

    async def process_pool_predictor(image_bytes: bytes) -> PredictionResult:
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
    app = FastAPI(title="Image Classifier API", version="1.0.0", lifespan=lifespan if use_process_pool else None)

    if not use_process_pool:
        app.state.settings = get_settings()
        app.state.predictor = predictor or (lambda image_bytes: predict_image_bytes(image_bytes, app.state.settings))
    elif predictor is not None:
        app.state.predictor = predictor

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        settings = get_settings()
        image_bytes = await read_and_validate_image(file, settings)
        app_instance = request.app
        predictor_callable = getattr(app_instance.state, "predictor", None)
        if predictor_callable is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Predictor is not initialized")

        result = predictor_callable(image_bytes)
        if asyncio.iscoroutine(result):
            result = await result

        return PredictionResponse(label=result.label, confidence=result.confidence)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
