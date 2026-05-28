#!/usr/bin/env python3
"""Curate data/iteration_* and data/generated_trajectories_* into data/curated/."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_import_samples import (  # noqa: E402
    CURATED_DIR,
    classify,
    prefixed,
    score,
    strip_internal,
    write_jsonl,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PREFIX = "data"
SOURCE_GLOBS = ("iteration_*_trajectories.jsonl", "generated_trajectories_*.jsonl")


def load_rows(data_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths: List[Path] = []
    for pattern in SOURCE_GLOBS:
        paths.extend(sorted(data_dir.glob(pattern)))
    for path in paths:
        if path.name.endswith(".bak"):
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample["_source_file"] = path.name
            sample["_source_line"] = line_no
            rows.append(sample)
    return rows


def pick_best_with_meta(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for sample in rows:
        instruction = sample.get("instruction", "")
        if instruction not in best or score(sample) > score(best[instruction]):
            best[instruction] = sample
    output: List[Dict[str, Any]] = []
    for instruction in sorted(best):
        row = strip_internal(best[instruction])
        row["source_run"] = best[instruction].get("_source_file")
        output.append(row)
    return output


def curate(data_dir: Path, out_dir: Path, prefix: str = DEFAULT_PREFIX) -> Dict[str, Any]:
    rows = load_rows(data_dir)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejects: Counter[str] = Counter()

    for sample in rows:
        label = classify(sample)
        if label.startswith("reject_"):
            rejects[label] += 1
            continue
        buckets[label].append(sample)

    outputs = {
        "good_tool_workflow.jsonl": pick_best_with_meta(buckets["good_tool_workflow"]),
        "good_tool_patch_single_python.jsonl": pick_best_with_meta(buckets["good_tool_patch_single_python"]),
        "good_inline_algorithm.jsonl": pick_best_with_meta(buckets["good_inline_algorithm"]),
        "good_inline_file_task.jsonl": pick_best_with_meta(buckets["good_inline_file_task"]),
    }

    written: Dict[str, int] = {}
    for name, rows_out in outputs.items():
        if not rows_out and name not in (
            "good_tool_workflow.jsonl",
            "good_inline_algorithm.jsonl",
            "good_inline_file_task.jsonl",
        ):
            continue
        path = out_dir / prefixed(prefix, name)
        write_jsonl(path, rows_out)
        written[prefixed(prefix, name)] = len(rows_out)

    by_source: Dict[str, int] = Counter()
    for label_rows in buckets.values():
        for sample in label_rows:
            by_source[sample.get("_source_file", "?")] += 1

    good_counts = {
        key.replace(".jsonl", ""): len(value) for key, value in outputs.items()
    }
    manifest = {
        "prefix": prefix,
        "data_dir": str(data_dir),
        "source_globs": list(SOURCE_GLOBS),
        "total_rows": len(rows),
        "reject_counts": dict(rejects),
        "good_counts": good_counts,
        "written_files": written,
        "rows_by_source_file": dict(by_source),
    }
    with (out_dir / prefixed(prefix, "manifest.json")).open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def merge_tool_training(out_dir: Path, prefix: str = DEFAULT_PREFIX) -> Dict[str, Any]:
    """Dedupe tool workflows across generated + data (+ import) packs."""
    packs = [
        ("generated", out_dir / "generated_recommended_tool_training.jsonl"),
        ("data", out_dir / prefixed(prefix, "good_tool_workflow.jsonl")),
        ("import", out_dir / "import_good_tool_workflow.jsonl"),
    ]
    priority = {"generated": 0, "data": 1, "import": 2}
    by_instruction: Dict[str, Dict[str, Any]] = {}

    for pack_name, path in packs:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = row.get("instruction", "")
            rank = (priority.get(pack_name, 9), 1 if row.get("recovered_from_failed") else 0, -score(row))
            existing = by_instruction.get(instruction)
            if existing is None or rank < existing["_rank"]:
                row = dict(row)
                row["training_pack"] = pack_name
                row["_rank"] = rank
                by_instruction[instruction] = row

    merged = []
    for instruction in sorted(by_instruction):
        row = by_instruction[instruction]
        row.pop("_rank", None)
        merged.append(row)

    out_path = out_dir / "all_tool_training.jsonl"
    write_jsonl(out_path, merged)
    by_pack = Counter(r.get("training_pack") for r in merged)
    return {"path": str(out_path), "count": len(merged), "by_pack": dict(by_pack)}


def main() -> None:
    out_dir = CURATED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = curate(DATA_DIR, out_dir)
    merged = merge_tool_training(out_dir)
    manifest["all_tool_training"] = merged
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
