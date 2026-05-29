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
    action_types,
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


def oversized_skip_instructions(out_dir: Path) -> set[str]:
    """Instructions excluded from all_tool_training (token overflow during train)."""
    skip_path = out_dir / "oversized_training_skipped.jsonl"
    if not skip_path.exists():
        return set()
    instructions: set[str] = set()
    for line in skip_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instruction = row.get("instruction", "")
        if instruction:
            instructions.add(instruction)
    return instructions


def merge_tool_training(out_dir: Path, prefix: str = DEFAULT_PREFIX) -> Dict[str, Any]:
    """Dedupe tool workflows across generated + data (+ import) packs."""
    oversized_skip = oversized_skip_instructions(out_dir)
    packs = [
        ("generated", out_dir / "generated_recommended_tool_training.jsonl"),
        ("data", out_dir / prefixed(prefix, "good_tool_workflow.jsonl")),
        ("import", out_dir / "import_good_tool_workflow.jsonl"),
        ("generated_patch", out_dir / "generated_good_tool_patch_single_python.jsonl"),
    ]
    for kept_name in (
        "dingo_run_tooling_kept.jsonl",
        "dingo_run_toolingv2_kept.jsonl",
        "dingo_run_toolingv3_kept.jsonl",
        "toolingv4_kept.jsonl",
        "Reading_kept.jsonl",
        "dingo_run2_kept.jsonl",
        "reading2_kept.jsonl",
        "Networking_Run_kept.jsonl",
        "dingo_run_networking_kept.jsonl",
        "dingo_run_newmodel_kept.jsonl",
        "dingo_run_Jsonfocus_kept.jsonl",
        "dingo_run_jsonfocus2_kept.jsonl",
        "new_model_test_v1_kept.jsonl",
        "dingo_run3_kept.jsonl",
        "macbook_archive_kept.jsonl",
        "NewModelRun3_kept.jsonl",
        "NewModelRun4_kept.jsonl",
        "NewModelRun5_kept.jsonl",
        "NewModelRun6_kept.jsonl",
        "NewModelRun7-ReadFocus_kept.jsonl",
        "NewModelRun8-ReadFocus_kept.jsonl",
        "dingo_run4_laptop_kept.jsonl",
        "V3_Jsonpatch_kept.jsonl",
    ):
        tooling_kept = out_dir / kept_name
        if tooling_kept.exists():
            packs.append((kept_name.replace(".jsonl", ""), tooling_kept))

    # Partial-run skips: only true rejects from sort_partial_run (see sort report).
    # Do not block an instruction if a *_kept pack has a good trajectory for it.
    kept_good_instructions: set[str] = set()
    for kept_name in (
        "dingo_run_tooling_kept.jsonl",
        "dingo_run_toolingv2_kept.jsonl",
        "dingo_run_toolingv3_kept.jsonl",
        "toolingv4_kept.jsonl",
        "Reading_kept.jsonl",
        "dingo_run2_kept.jsonl",
        "reading2_kept.jsonl",
        "Networking_Run_kept.jsonl",
        "dingo_run_networking_kept.jsonl",
        "dingo_run_newmodel_kept.jsonl",
        "dingo_run_Jsonfocus_kept.jsonl",
        "dingo_run_jsonfocus2_kept.jsonl",
        "new_model_test_v1_kept.jsonl",
        "dingo_run3_kept.jsonl",
        "macbook_archive_kept.jsonl",
        "NewModelRun3_kept.jsonl",
        "NewModelRun4_kept.jsonl",
        "NewModelRun5_kept.jsonl",
        "NewModelRun6_kept.jsonl",
        "NewModelRun7-ReadFocus_kept.jsonl",
        "NewModelRun8-ReadFocus_kept.jsonl",
        "dingo_run4_laptop_kept.jsonl",
        "V3_Jsonpatch_kept.jsonl",
    ):
        kept_path = out_dir / kept_name
        if not kept_path.exists():
            continue
        for line in kept_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not classify(row).startswith("reject_"):
                kept_good_instructions.add(row.get("instruction", ""))

    tooling_skip_instructions: set[str] = set()
    for skip_name in (
        "dingo_run_tooling_skipped.jsonl",
        "dingo_run_toolingv2_skipped.jsonl",
        "dingo_run_toolingv3_skipped.jsonl",
        "toolingv4_skipped.jsonl",
        "Reading_skipped.jsonl",
        "dingo_run2_skipped.jsonl",
        "reading2_skipped.jsonl",
        "Networking_Run_skipped.jsonl",
        "dingo_run_networking_skipped.jsonl",
        "dingo_run_newmodel_skipped.jsonl",
        "dingo_run_Jsonfocus_skipped.jsonl",
        "dingo_run_jsonfocus2_skipped.jsonl",
        "new_model_test_v1_skipped.jsonl",
        "dingo_run3_skipped.jsonl",
    ):
        tooling_skipped = out_dir / skip_name
        if tooling_skipped.exists():
            for line in tooling_skipped.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                instruction = row.get("instruction", "")
                if (
                    classify(row).startswith("reject_")
                    and instruction
                    and instruction not in kept_good_instructions
                ):
                    tooling_skip_instructions.add(instruction)
    priority = {
        "generated": 0,
        "dingo_run_tooling_kept": -1,
        "dingo_run_toolingv2_kept": -1,
        "dingo_run_toolingv3_kept": -1,
        "toolingv4_kept": -1,
        "Reading_kept": -1,
        "dingo_run2_kept": -1,
        "reading2_kept": -2,
        "Networking_Run_kept": -1,
        "dingo_run_networking_kept": -1,
        "dingo_run_newmodel_kept": -1,
        "dingo_run_Jsonfocus_kept": -1,
        "dingo_run_jsonfocus2_kept": -1,
        "new_model_test_v1_kept": -1,
        "dingo_run3_kept": -1,
        "macbook_archive_kept": -1,
        "NewModelRun3_kept": -1,
        "NewModelRun4_kept": -1,
        "NewModelRun5_kept": -1,
        "NewModelRun6_kept": -1,
        "NewModelRun7-ReadFocus_kept": -1,
        "NewModelRun8-ReadFocus_kept": -1,
        "dingo_run4_laptop_kept": -1,
        "V3_Jsonpatch_kept": -1,
        "data": 1,
        "import": 2,
        "generated_patch": 3,
    }
    by_instruction: Dict[str, Dict[str, Any]] = {}

    for pack_name, path in packs:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = row.get("instruction", "")
            if instruction in tooling_skip_instructions or instruction in oversized_skip:
                continue
            types = action_types(row)
            has_read = "read_file" in types
            multi_py = sum(1 for t in types if t == "python") >= 2
            rank = (
                priority.get(pack_name, 9),
                1 if row.get("recovered_from_failed") else 0,
                0 if has_read else 1,
                0 if multi_py else 1,
                -score(row),
            )
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
