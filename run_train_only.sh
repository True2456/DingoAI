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
#
# To overwrite a partial/failed output dir (5th arg: overwrite):
#   ./run_train_only.sh data/batch.jsonl 290 models/mlx_self_training/pilot_v1 "" overwrite
# ─────────────────────────────────────────────────────────────────────────────

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$ROOT_DIR/setup_venv.sh"
cd "$ROOT_DIR"

DATA=${1:-""}
ITERS=${2:-200}
OUTPUT_DIR=${3:-"models/mlx_self_training/train_only"}
RESUME=${4:-""}
OVERWRITE_FLAG=${5:-""}
WIRE_FORMAT=${6:-""}

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
echo "  Max seq:    8192 (default; pass --max-seq-length via python for other)"
if [ -n "$RESUME" ]; then
    echo "  Resuming:   $RESUME"
fi
if [ "$OVERWRITE_FLAG" = "overwrite" ]; then
    echo "  Overwrite:  yes"
fi
echo "======================================================"

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
fi
OVERWRITE_ARG=""
if [ "$OVERWRITE_FLAG" = "overwrite" ]; then
    OVERWRITE_ARG="--overwrite"
fi
WIRE_ARG=""
if [ -n "$WIRE_FORMAT" ]; then
    WIRE_ARG="--wire-format $WIRE_FORMAT"
fi

"$ROOT_DIR/mlx_foundation/venv/bin/python" "$ROOT_DIR/mlx_foundation/src/main.py" \
    --mode train-only \
    --data "$DATA" \
    --train-iters "$ITERS" \
    --train-output "$OUTPUT_DIR" \
    $RESUME_FLAG \
    $OVERWRITE_ARG \
    $WIRE_ARG
