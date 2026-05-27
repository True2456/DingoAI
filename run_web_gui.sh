#!/bin/bash
# Launch the local web GUI for generation, training, resume, and reports.

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8765}

echo "Opening MLX Self Training Console at http://$HOST:$PORT"
open "http://$HOST:$PORT" >/dev/null 2>&1 || true
./mlx_foundation/venv/bin/python web/server.py --host "$HOST" --port "$PORT"
