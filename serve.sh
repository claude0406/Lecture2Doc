#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
  echo "실행 환경(.venv)이 없습니다. 프로젝트 설치를 먼저 진행하세요."
  exit 1
fi
exec .venv/bin/python -m lecture2doc.launcher "$@"
