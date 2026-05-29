#!/usr/bin/env python3
"""Sort a partial generation JSONL into kept vs skipped tiers for manual review.

Keeps broad agentic tool workflows (Claude Code style): multi-file writes, list_dir,
read_file, python verification, frontend and backend — not only security/CSV patch tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_import_samples import (  # noqa: E402
    CURATED_DIR,
    action_types,
    classify,
    score,
    strip_internal,
    write_jsonl,
)


def tier_row(
    row: Dict[str, Any],
    existing_instructions: set[str],
    existing_by_instruction: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    instruction = row.get("instruction", "")
    label = classify(row)
    types = action_types(row)

    if label.startswith("reject"):
        return "skip", label

    if instruction in existing_instructions:
        existing = existing_by_instruction.get(instruction)
        if existing is not None and score(row) >= score(existing):
            return "keep", "upgrade_over_existing_in_all_tool_training"
        return "skip", "duplicate_instruction_in_all_tool_training"

    if not types or types[-1] != "none":
        return "skip", "incomplete_not_none_terminal"

    if "write_file" not in types or "python" not in types:
        return "skip", "missing_write_file_or_python"

    # Broad agentic keep: any verified file workflow
    if label in ("good_tool_workflow", "good_tool_patch_single_python"):
        py = sum(1 for t in types if t == "python")
        has_read = "read_file" in types
        has_list = "list_dir" in types
        if has_read and py >= 1:
            return "keep", "agentic_with_read_file"
        if py >= 2:
            return "keep", "agentic_multipy"
        if has_list and py >= 1:
            return "keep", "agentic_list_dir_package"
        return "keep", "agentic_file_workflow"

    return "skip", label


def sort_run(
    input_path: Path,
    out_dir: Path,
    stem: str,
    training_path: Path,
) -> Dict[str, Any]:
    existing_by_instruction: Dict[str, Dict[str, Any]] = {}
    if training_path.exists():
        for line in training_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = row.get("instruction", "")
            if instruction:
                existing_by_instruction[instruction] = row
    existing = set(existing_by_instruction)

    rows = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]

    kept: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    kept_by_instruction: Dict[str, Dict[str, Any]] = {}

    for index, row in enumerate(rows, 1):
        decision, reason = tier_row(row, existing, existing_by_instruction)
        entry = {
            "index": index,
            "decision": decision,
            "reason": reason,
            "classify": classify(row),
            "instruction": row.get("instruction", "")[:120],
        }
        report.append(entry)
        if decision == "keep":
            out = strip_internal(row)
            out["source_run"] = input_path.name
            out["training_pack"] = f"{stem}_kept"
            instruction = row.get("instruction", "")
            prev = kept_by_instruction.get(instruction)
            if prev is None or score(out) >= score(prev):
                kept_by_instruction[instruction] = out
            existing.add(instruction)
        else:
            skipped.append(row)

    kept = [
        kept_by_instruction[instruction]
        for instruction in sorted(kept_by_instruction)
    ]

    kept_path = out_dir / f"{stem}_kept.jsonl"
    skipped_path = out_dir / f"{stem}_skipped.jsonl"
    report_path = out_dir / f"{stem}_sort_report.json"

    write_jsonl(kept_path, kept)
    write_jsonl(skipped_path, skipped)
    with report_path.open("w") as handle:
        json.dump(
            {
                "input": str(input_path),
                "kept_path": str(kept_path),
                "skipped_path": str(skipped_path),
                "kept_count": len(kept),
                "skipped_count": len(skipped),
                "rows": report,
            },
            handle,
            indent=2,
        )

    return {
        "kept_path": str(kept_path),
        "skipped_path": str(skipped_path),
        "report_path": str(report_path),
        "kept_count": len(kept),
        "skipped_count": len(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated/dingo_run_tooling.jsonl"),
    )
    parser.add_argument("--stem", type=str, default="dingo_run_tooling")
    parser.add_argument("--out-dir", type=Path, default=CURATED_DIR)
    parser.add_argument(
        "--training",
        type=Path,
        default=CURATED_DIR / "all_tool_training.jsonl",
    )
    args = parser.parse_args()
    print(json.dumps(sort_run(args.input, args.out_dir, args.stem, args.training), indent=2))


if __name__ == "__main__":
    main()
