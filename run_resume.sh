#!/bin/bash
# Automatically detects the latest completed training iteration and resumes distillation from there.

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$ROOT_DIR/setup_venv.sh"
cd "$ROOT_DIR"

BASE_DIR="$ROOT_DIR/models/mlx_self_training"
PYTHON_BIN="$ROOT_DIR/mlx_foundation/venv/bin/python"
MAIN="$ROOT_DIR/mlx_foundation/src/main.py"

if [ ! -d "$BASE_DIR" ]; then
    echo "No training directory found at $BASE_DIR."
    echo "Starting from scratch in full mode..."
    "$PYTHON_BIN" -u "$MAIN" --mode full "$@"
    exit 0
fi

# Find the highest iteration subdirectory that actually contains adapters.safetensors
LATEST_ITER=""
for dir in $(ls -d $BASE_DIR/iteration_* 2>/dev/null | sort -Vr); do
    if [ -f "$dir/adapters.safetensors" ]; then
        LATEST_ITER="${dir#$ROOT_DIR/}"
        break
    fi
done

if [ -z "$LATEST_ITER" ]; then
    echo "No valid checkpoints (adapters.safetensors) found in $BASE_DIR."
    echo "Starting from scratch in full mode..."
    "$PYTHON_BIN" -u "$MAIN" --mode full "$@"
else
    echo "Found latest checkpoint: $LATEST_ITER"
    echo "Resuming Agentic Distillation in FULL PRODUCTION mode..."
    "$PYTHON_BIN" -u "$MAIN" --mode full --resume "$LATEST_ITER" "$@"
fi
