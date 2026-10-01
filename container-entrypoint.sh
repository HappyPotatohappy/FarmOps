#!/bin/sh
set -eu
cd /app

# Existing volumes do not receive new image directories through copy-up.
mkdir -p "$BEEOPS_RUNTIME/mlruns"

# One named volume preserves uploads, model versions, and training history.
# The image's seed remains available independently of that writable volume.
"$BEEOPS_SERVICE_PYTHON" scripts/initialize_models.py \
  --seed /app/data/horizon_models --models "$BEEOPS_HORIZON_MODELS"
"$BEEOPS_SERVICE_PYTHON" scripts/initialize_runtime.py --runtime "$BEEOPS_RUNTIME"
exec "$BEEOPS_SERVICE_PYTHON" -m uvicorn app.main:create_app --factory \
  --host 0.0.0.0 --port 8012
