#!/bin/sh
set -eu
BUNDLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$BUNDLE_ROOT"
SERVICE_PYTHON=${BEEOPS_SERVICE_PYTHON:-"$BUNDLE_ROOT/.venv/bin/python"}
if [ ! -x "$SERVICE_PYTHON" ]; then echo '먼저 sh setup.sh 를 실행하세요.' >&2; exit 1; fi
"$SERVICE_PYTHON" scripts/check_bundle.py
"$SERVICE_PYTHON" -m pytest tests -q
if command -v node >/dev/null 2>&1; then
  node --test tests/ui/*.cjs
else
  echo 'Node.js가 없어 프런트엔드 검사만 생략했습니다. Node.js 20 이상에서 sh verify.sh 를 다시 실행하세요.'
fi
