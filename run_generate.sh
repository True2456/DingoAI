#!/bin/bash
# run_generate.sh
# ─────────────────────────────────────────────────────────────────────────────
# Run this on your second machine (e.g. M3 Max 64GB) to generate agentic coding
# trajectories using the teacher models. No student model is needed.
#
# Usage:
#   ./run_generate.sh                         # generates 100 trajectories
#   ./run_generate.sh 200                     # generates 200 trajectories
#   ./run_generate.sh 100 data/my_batch.jsonl   # custom output path
#   ./run_generate.sh 100 --qwen              # Qwen Coder Next generates task/questions
#   ./run_generate.sh 100 --gemma             # Gemma generates task/questions
#   ./run_generate.sh 100 data/batch.jsonl --qwen
#
# After it finishes, copy the output JSONL to your training machine and run:
#   ./run_train_only.sh data/<batch_file>.jsonl
# ─────────────────────────────────────────────────────────────────────────────

SAMPLES=100
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    SAMPLES="$1"
    shift
fi

GENERATOR_LABEL="auto"
for arg in "$@"; do
    case "$arg" in
        --qwen)
            GENERATOR_LABEL="qwen-coder"
            ;;
        --gemma)
            GENERATOR_LABEL="gemma-31b"
            ;;
    esac
done

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")
OUTPUT="data/generated/claude_tool_traces_${GENERATOR_LABEL}_${SAMPLES}_samples_${TIMESTAMP}.jsonl"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    OUTPUT="$1"
    shift
fi

EXTRA_ARGS=("$@")

echo "======================================================"
echo "  GENERATE-ONLY MODE"
echo "  Samples:  $SAMPLES"
echo "  Output:   $OUTPUT"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Extra:    ${EXTRA_ARGS[*]}"
fi
echo "======================================================"

./mlx_foundation/venv/bin/python mlx_foundation/src/main.py \
    --mode generate-only \
    --samples "$SAMPLES" \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done! Transfer this file to your training machine:"
echo "  $OUTPUT"
echo ""
echo "Then on the training machine run:"
echo "  ./run_train_only.sh $OUTPUT"
