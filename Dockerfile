# GPU runtime for the online retrieval application.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000 \
    AIC_WORKSPACE_ROOT=/workspace \
    AIC_5FPS_DATA=/five-fps \
    AIC_CACHE_DIR=/runtime \
    AIC_SIGLIP_DEVICE=cpu \
    HF_HOME=/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CUDA-enabled PyTorch pair before the remaining dependencies.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.7.1 torchvision==0.22.1

COPY requirements.docker.txt ./
RUN pip install -r requirements.docker.txt

# Keep preprocessing tools, artifacts, model caches, and Git metadata out of the image.
COPY app.py aic_config.py l23_scenes.py retrieval.py preview.py submission.py ./
COPY templates ./templates
COPY docker/entrypoint.py /usr/local/bin/aic-entrypoint.py
RUN mkdir -p /app/data /runtime/indexes /runtime/extracted /supplied-indexes /workspace /five-fps /hf \
    && chmod 755 /usr/local/bin/aic-entrypoint.py

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:5000/api/status || exit 1

ENTRYPOINT ["python", "/usr/local/bin/aic-entrypoint.py"]
CMD ["python", "app.py"]
