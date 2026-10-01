#!/bin/sh
set -eu
BUNDLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BUNDLE_ROOT"
if command -v uv >/dev/null 2>&1; then
  [ -x .venv/bin/python ] || uv venv .venv --python 3.11
  [ -x .venv-tirex/bin/python ] || uv venv .venv-tirex --python 3.11
  uv pip install --python .venv/bin/python -r requirements-service.lock.txt
  uv pip install --python .venv-tirex/bin/python -r requirements-tirex.lock.txt
else
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo 'Python 3.11 또는 uv가 필요합니다. 설치 후 sh setup.sh를 다시 실행하세요.' >&2
    exit 1
  fi
  [ -x .venv/bin/python ] || python3.11 -m venv .venv
  [ -x .venv-tirex/bin/python ] || python3.11 -m venv .venv-tirex
  .venv/bin/python -m pip install -r requirements-service.lock.txt
  .venv-tirex/bin/python -m pip install -r requirements-tirex.lock.txt
fi
.venv/bin/python -c 'import tensorflow, mlflow, fastapi; print("서비스 환경 준비 완료")'
.venv-tirex/bin/python -c 'import tirex2, torch; print("TiRex-2 환경 준비 완료")'
.venv/bin/python scripts/fetch_model.py
.venv/bin/python scripts/check_bundle.py
echo '설치 완료. sh start.sh 실행 후 http://127.0.0.1:8012 를 여세요.'
