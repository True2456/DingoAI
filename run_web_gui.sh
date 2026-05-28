#!/bin/bash
# Launch the local web GUI for generation, training, resume, and reports.

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8765}

"$ROOT_DIR/setup_venv.sh"
cd "$ROOT_DIR"

echo "Opening DingoAI console at http://$HOST:$PORT"
open "http://$HOST:$PORT" >/dev/null 2>&1 || true
"$ROOT_DIR/mlx_foundation/venv/bin/python" "$ROOT_DIR/web/server.py" --host "$HOST" --port "$PORT"
