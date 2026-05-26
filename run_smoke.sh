#!/bin/bash
# Run a quick smoke test of the MLX Self-Training & Distillation Pipeline
echo "Launching Agentic Distillation in SMOKE TEST mode..."
./mlx_foundation/venv/bin/python mlx_foundation/src/main.py --mode smoke
