from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class BenchmarkResult(BaseModel):
    variant: str
    model_size_mb: float
    latency_ms: float
    throughput_rps: float