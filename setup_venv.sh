#!/bin/bash
# Create and populate the local MLX virtual environment if needed.

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/mlx_foundation/venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ "${SKIP_VENV_SETUP:-0}" = "1" ]; then
    exit 0
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
required = ["mlx_lm", "mlx", "transformers", "safetensors"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
sys.exit(1 if missing else 0)
PY
then
    echo "Installing Python dependencies into $VENV_DIR"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
fi
