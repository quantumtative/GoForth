#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

HOST="${RNA_WORKBENCH_HOST:-127.0.0.1}"
PORT="${RNA_WORKBENCH_PORT:-7860}"
DEVICE="${RNA_WORKBENCH_DEVICE:-auto}"
PYTHON_BIN="${RNA_WORKBENCH_PYTHON:-python}"
CORS_ORIGIN="${RNA_WORKBENCH_CORS_ORIGIN:-}"

args=(
  apps/rna_workbench/server.py
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE"
)

if [[ -n "$CORS_ORIGIN" ]]; then
  args+=(--cors-origin "$CORS_ORIGIN")
fi

exec "$PYTHON_BIN" "${args[@]}"
