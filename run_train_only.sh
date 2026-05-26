#!/bin/bash
# run_train_only.sh
# ─────────────────────────────────────────────────────────────────────────────
# Run this on your PRIMARY machine to train on a pre-generated JSONL batch file.
# No teacher models are loaded — only the student model (Gemma-4-26B MoE).
#
# Usage:
#   ./run_train_only.sh data/batch.jsonl                  # default 200 iters
#   ./run_train_only.sh data/batch.jsonl 150              # custom iter count
#   ./run_train_only.sh data/batch.jsonl 200 models/my_adapter  # custom output dir
#
# To resume from an existing adapter:
#   ./run_train_only.sh data/batch.jsonl 200 models/my_adapter models/mlx_self_training/iteration_1
# ─────────────────────────────────────────────────────────────────────────────

DATA=${1:-""}
ITERS=${2:-200}
OUTPUT_DIR=${3:-"models/mlx_self_training/train_only"}
RESUME=${4:-""}

if [ -z "$DATA" ]; then
    echo "ERROR: Please provide a path to a generated JSONL file."
    echo "Usage: ./run_train_only.sh <data.jsonl> [iters] [output_dir] [resume_adapter]"
    exit 1
fi

if [ ! -f "$DATA" ]; then
    echo "ERROR: File not found: $DATA"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "$DATA" | tr -d ' ')
RATIO=$(echo "scale=1; $ITERS / $SAMPLE_COUNT" | bc)

echo "======================================================"
echo "  TRAIN-ONLY MODE"
echo "  Data:       $DATA ($SAMPLE_COUNT samples)"
echo "  Iterations: $ITERS"
echo "  Ratio:      ${RATIO}x  (safe if <3.0)"
echo "  Output:     $OUTPUT_DIR"
if [ -n "$RESUME" ]; then
    echo "  Resuming:   $RESUME"
fi
echo "======================================================"

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
fi

./mlx_foundation/venv/bin/python mlx_foundation/src/main.py \
    --mode train-only \
    --data "$DATA" \
    --train-iters "$ITERS" \
    --train-output "$OUTPUT_DIR" \
    $RESUME_FLAG
