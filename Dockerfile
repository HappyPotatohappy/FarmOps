# syntax=docker/dockerfile:1
FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BEEOPS_SERVICE_PYTHON=/opt/venv-service/bin/python \
    BEEOPS_TIREX_PYTHON=/opt/venv-tirex/bin/python \
    BEEOPS_RUNTIME=/app/runtime \
    BEEOPS_HORIZON_MODELS=/app/runtime/horizon_models \
    BEEOPS_DASHBOARD_DATASET=observed \
    BEEOPS_PORT=8012 \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 beeops \
    && python -m venv /opt/venv-service \
    && python -m venv /opt/venv-tirex

WORKDIR /tmp/requirements
COPY requirements-service.lock.txt requirements-tirex.lock.txt \
     requirements-container-service.txt requirements-container-tirex.txt ./

# Installing the explicit CPU wheels first avoids the much larger CUDA runtime
# dependencies pulled by default Linux Torch wheels. Both amd64 and arm64 use
# the same Python/model versions; the wheel index chooses the native platform.
RUN /opt/venv-service/bin/python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu' \
    && /opt/venv-tirex/bin/python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
RUN /opt/venv-service/bin/python -m pip install --no-cache-dir -r requirements-container-service.txt \
    && /opt/venv-tirex/bin/python -m pip install --no-cache-dir -r requirements-container-tirex.txt \
    && /opt/venv-service/bin/python -m pip check \
    && /opt/venv-tirex/bin/python -m pip check

WORKDIR /app
# MLflow creates ./mlruns even with a SQLite tracking URI. Keep that default
# path writable and persistent without relying on a developer's local folder.
RUN mkdir -p /app/runtime/mlruns /app/.cache/matplotlib \
    && chown -R beeops:beeops /app/runtime /app/.cache \
    && ln -s /app/runtime/mlruns /app/mlruns
COPY --chown=beeops:beeops . /app
USER beeops

# GitHub clone/ZIP contains the checkpoint metadata and license. Fetch the exact
# official revision and verify its SHA-256 during the image build, before any
# runtime volume is mounted. Normal prediction/training then works offline.
RUN /opt/venv-service/bin/python scripts/fetch_model.py

EXPOSE 8012
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD /opt/venv-service/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8012/workspaces', timeout=8).read()" || exit 1
ENTRYPOINT ["sh", "/app/container-entrypoint.sh"]
