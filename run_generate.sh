#!/bin/bash
# run_generate.sh
# ─────────────────────────────────────────────────────────────────────────────
# Run this on your second machine (e.g. M3 Max 64GB) to generate agentic coding
# trajectories using the teacher models. No student model is needed.
#
# Usage:
#   ./run_generate.sh                         # generates 100 trajectories
#   ./run_generate.sh 200                     # generates 200 trajectories
#   ./run_generate.sh 100 data/batch_002.jsonl  # custom output path
#
# After it finishes, copy the output JSONL to your training machine and run:
#   ./run_train_only.sh data/<batch_file>.jsonl
# ─────────────────────────────────────────────────────────────────────────────

SAMPLES=${1:-100}
OUTPUT=${2:-"data/generated_trajectories_$(date +%Y%m%d_%H%M%S).jsonl"}

echo "======================================================"
echo "  GENERATE-ONLY MODE"
echo "  Samples:  $SAMPLES"
echo "  Output:   $OUTPUT"
echo "======================================================"

./mlx_foundation/venv/bin/python mlx_foundation/src/main.py \
    --mode generate-only \
    --samples "$SAMPLES" \
    --output "$OUTPUT"

echo ""
echo "Done! Transfer this file to your training machine:"
echo "  $OUTPUT"
echo ""
echo "Then on the training machine run:"
echo "  ./run_train_only.sh $OUTPUT"
