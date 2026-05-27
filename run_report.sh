#!/bin/bash
# Generate a self-contained HTML dashboard for one or more trajectory JSONL files.
#
# Usage:
#   ./run_report.sh data/generated/example.jsonl
#   ./run_report.sh data/generated/a.jsonl data/generated/b.jsonl

if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide at least one trajectory JSONL file."
    echo "Usage: ./run_report.sh <trajectory.jsonl> [more.jsonl ...]"
    exit 1
fi

./mlx_foundation/venv/bin/python tools/report_runs.py "$@"
