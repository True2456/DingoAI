#!/usr/bin/env python3
"""Convert Dingo-track curated JSONL to oMLX / Claude Code wire format for training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlx_foundation" / "src"))

from agent_wire_formats import convert_sample_for_omlx_training  # noqa: E402

DEFAULT_IN = ROOT / "data" / "curated" / "all_tool_training.jsonl"
DEFAULT_OUT = ROOT / "data" / "curated" / "all_omlx_tool_training.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: missing input {args.input}")
        print("Run: python3 tools/curate_all.py")
        sys.exit(1)

    converted = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open() as src, args.output.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            sample = json.loads(line)
            if not sample.get("sandbox_success", True):
                skipped += 1
                continue
            out = convert_sample_for_omlx_training(sample)
            if not out.get("actions") or "call:" not in out["actions"]:
                skipped += 1
                continue
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            converted += 1

    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "converted": converted,
        "skipped": skipped,
        "wire_format": "omlx_claude",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print(f"\nTrain with:\n  ./run_train_only.sh {args.output} <iters> models/mlx_self_training/pilot_v4_omlx --wire-format omlx_claude")


if __name__ == "__main__":
    main()
