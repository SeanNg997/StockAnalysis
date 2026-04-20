#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

echo "=============================================="
echo "  StockAnalysis Web Console"
echo "  地址: http://$HOST:$PORT"
echo "=============================================="

"$PYTHON_BIN" -m uvicorn webapp.server:app --host "$HOST" --port "$PORT"
