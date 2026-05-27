#!/bin/bash
# Automatically detects the latest completed training iteration and resumes distillation from there.

BASE_DIR="models/mlx_self_training"

if [ ! -d "$BASE_DIR" ]; then
    echo "No training directory found at $BASE_DIR."
    echo "Starting from scratch in full mode..."
    ./mlx_foundation/venv/bin/python -u mlx_foundation/src/main.py --mode full "$@"
    exit 0
fi

# Find the highest iteration subdirectory that actually contains adapters.safetensors
LATEST_ITER=""
for dir in $(ls -d $BASE_DIR/iteration_* 2>/dev/null | sort -Vr); do
    if [ -f "$dir/adapters.safetensors" ]; then
        LATEST_ITER="$dir"
        break
    fi
done

if [ -z "$LATEST_ITER" ]; then
    echo "No valid checkpoints (adapters.safetensors) found in $BASE_DIR."
    echo "Starting from scratch in full mode..."
    ./mlx_foundation/venv/bin/python -u mlx_foundation/src/main.py --mode full "$@"
else
    echo "Found latest checkpoint: $LATEST_ITER"
    echo "Resuming Agentic Distillation in FULL PRODUCTION mode..."
    ./mlx_foundation/venv/bin/python -u mlx_foundation/src/main.py --mode full --resume "$LATEST_ITER" "$@"
fi
