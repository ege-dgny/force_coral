#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-forte_phase1}"
BACKEND="${2:-cpu}"

cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10 pip ffmpeg
fi

case "$BACKEND" in
  cpu)
    REQUIREMENTS="requirements-cpu.txt"
    ;;
  mps|macos|macos-mps)
    REQUIREMENTS="requirements-macos-mps.txt"
    ;;
  cuda|linux-cuda)
    REQUIREMENTS="requirements-linux-cuda.txt"
    ;;
  *)
    echo "unknown backend '$BACKEND' (expected: cpu | mps | cuda)" >&2
    exit 1
    ;;
esac

conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade pip
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$REQUIREMENTS"

echo "Environment '$ENV_NAME' is ready using backend '$BACKEND'."
