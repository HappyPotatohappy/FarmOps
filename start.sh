#!/bin/sh
set -eu
BUNDLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BUNDLE_ROOT"
SERVICE_PYTHON=${BEEOPS_SERVICE_PYTHON:-"$BUNDLE_ROOT/.venv/bin/python"}
export BEEOPS_TIREX_PYTHON=${BEEOPS_TIREX_PYTHON:-"$BUNDLE_ROOT/.venv-tirex/bin/python"}
export BEEOPS_RUNTIME=${BEEOPS_RUNTIME:-"$BUNDLE_ROOT/runtime"}
export BEEOPS_HORIZON_MODELS=${BEEOPS_HORIZON_MODELS:-"$BEEOPS_RUNTIME/horizon_models"}
export BEEOPS_DASHBOARD_DATASET=observed
export BEEOPS_PORT=${BEEOPS_PORT:-8012}
export MPLCONFIGDIR="$BUNDLE_ROOT/.cache/matplotlib"
if [ ! -x "$SERVICE_PYTHON" ] || [ ! -x "$BEEOPS_TIREX_PYTHON" ]; then
  echo '실행 환경이 없습니다. 이 폴더에서 먼저 sh setup.sh 를 실행하세요.' >&2
  exit 1
fi
"$SERVICE_PYTHON" - <<'PY'
import os,socket
port=int(os.environ['BEEOPS_PORT'])
if not 1<=port<=65535: raise SystemExit('BEEOPS_PORT는 1~65535 범위여야 합니다.')
with socket.socket() as sock:
    try: sock.bind(('127.0.0.1',port))
    except OSError: raise SystemExit(f'{port} 포트가 사용 중입니다. 이미 열린 BeeOPS를 사용하거나 BEEOPS_PORT=8014 sh start.sh 로 실행하세요.')
PY
"$SERVICE_PYTHON" scripts/fetch_model.py
"$SERVICE_PYTHON" scripts/initialize_models.py --seed "$BUNDLE_ROOT/data/horizon_models" --models "$BEEOPS_HORIZON_MODELS"
"$SERVICE_PYTHON" scripts/initialize_runtime.py --runtime "$BEEOPS_RUNTIME"
exec "$SERVICE_PYTHON" -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port "$BEEOPS_PORT"
