FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim

ENV PYTHONPATH=/install/lib/python3.11/site-packages

WORKDIR /app

COPY --from=builder /install /install
COPY app ./app
COPY model ./model
COPY scripts ./scripts
COPY README.md ./README.md

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
