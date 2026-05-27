#!/bin/bash
# Run a full production-scale training run of the MLX Self-Training & Distillation Pipeline
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$ROOT_DIR/setup_venv.sh"
cd "$ROOT_DIR"
echo "Launching Agentic Distillation in FULL PRODUCTION mode..."
"$ROOT_DIR/mlx_foundation/venv/bin/python" "$ROOT_DIR/mlx_foundation/src/main.py" --mode full "$@"
