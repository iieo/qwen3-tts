FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON=python3.13

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126

COPY . .

EXPOSE 3000

CMD ["uv", "run", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "3000"]
