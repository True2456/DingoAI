#!/bin/bash
# Run a full production-scale training run of the MLX Self-Training & Distillation Pipeline
echo "Launching Agentic Distillation in FULL PRODUCTION mode..."
./mlx_foundation/venv/bin/python mlx_foundation/src/main.py --mode full "$@"
