#!/usr/bin/env python3
"""Curate trajectory JSONL directories into tiered training-ready outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_import_samples import (  # noqa: E402
    CURATED_DIR,
    classify,
    is_json_abort,
    prefixed,
    score,
    strip_internal,
    write_jsonl,
)

GENERATED_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"
CURATED_DIR = Path(__file__).resolve().parents[1] / "data" / "curated"
DEFAULT_PREFIX = "generated"


def oversized_skip_instructions() -> set[str]:
    from curate_data_trajectories import oversized_skip_instructions as load_skip

    return load_skip(CURATED_DIR)


def parse_legacy_actions(actions_str: str) -> List[Tuple[str, str]]:
    parts = str(actions_str).split(" | ")
    parsed: List[Tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            action_type, action_input = part.split(":", 1)
            parsed.append((action_type.strip(), action_input.strip()))
        else:
            parsed.append((part, ""))
    return parsed


def action_types(sample: Dict[str, Any]) -> List[str]:
    turns = sample.get("turns")
    if isinstance(turns, list) and turns:
        return [(turn.get("action") or {}).get("type") for turn in turns]
    return [action_type for action_type, _ in parse_legacy_actions(sample.get("actions", ""))]


def last_python_passed(traj: Dict[str, Any]) -> bool:
    last = None
    for turn in traj.get("turns") or []:
        if (turn.get("action") or {}).get("type") == "python":
            last = turn
    if not last:
        return False
    obs = last.get("observation") or {}
    combined = (obs.get("stdout") or "") + (obs.get("stderr") or "")
    if any(m in combined for m in ("FAILED", "AssertionError", "Traceback (most recent call last):")):
        return False
    return obs.get("success") or obs.get("verification_passed") or "OK" in combined or "Verified" in combined


def recoverable_from_failed(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    traj = row.get("trajectory") or {}
    if not traj.get("turns"):
        return None
    if is_json_abort(traj):
        return None
    types = action_types(traj)
    if "write_file" not in types or "python" not in types or types[-1] != "none":
        return None
    if not last_python_passed(traj):
        return None
    out = dict(traj)
    out["sandbox_success"] = True
    out["recovered_from_failed"] = row.get("_source_file")
    out["recovery_attempt"] = row.get("attempt")
    return out


def load_success_rows(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        if path.name.endswith(".bak") or path.name.endswith("_failed_attempts.jsonl"):
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


def load_recoverable_failed(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*_failed_attempts.jsonl")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_source_file"] = path.name
            row["_source_line"] = line_no
            recovered = recoverable_from_failed(row)
            if recovered:
                recovered["_source_file"] = path.name
                recovered["_source_line"] = line_no
                rows.append(recovered)
    return rows


SOURCE_PRIORITY = (
    "newprompt3.jsonl",
    "New Prompt1.jsonl",
    "New Prompt2.jsonl",
    "qwencoder9.jsonl",
    "qwencoder8.jsonl",
    "qwencoder7.jsonl",
    "qwencoder6.jsonl",
    "Test1 Claude agentic.jsonl",
    "Test2 Claude agentic.jsonl",
    "gui_claude_tool_traces.jsonl",
    "Qwencoder3.jsonl",
)


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
        if best[instruction].get("recovered_from_failed"):
            row["recovered_from_failed"] = best[instruction]["recovered_from_failed"]
        output.append(row)
    return output


def build_recommended_pack(tool_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer overnight newprompt3 successes; add unique instructions from other runs."""
    by_instruction: Dict[str, Dict[str, Any]] = {}
    priority_rank = {name: idx for idx, name in enumerate(SOURCE_PRIORITY)}

    def rank(sample: Dict[str, Any]) -> Tuple[int, int, int]:
        source = sample.get("source_run") or ""
        recovered = 1 if sample.get("recovered_from_failed") else 0
        return (priority_rank.get(source, 99), recovered, -score(sample))

    for sample in sorted(tool_rows, key=rank):
        instruction = sample.get("instruction", "")
        if instruction not in by_instruction:
            by_instruction[instruction] = sample
    return [by_instruction[key] for key in sorted(by_instruction)]


def curate(
    input_dir: Path,
    out_dir: Path,
    include_failed_recovery: bool,
    prefix: str = DEFAULT_PREFIX,
) -> Dict[str, Any]:
    rows = load_success_rows(input_dir)
    if include_failed_recovery:
        rows.extend(load_recoverable_failed(input_dir))
    skip_instructions = oversized_skip_instructions()

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejects: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    for sample in rows:
        if sample.get("instruction", "") in skip_instructions:
            continue
        by_source[sample.get("_source_file", "?")] += 1
        label = classify(sample)
        if label.startswith("reject_"):
            rejects[label] += 1
            continue
        buckets[label].append(sample)

    outputs = {
        "good_tool_workflow": pick_best_with_meta(buckets["good_tool_workflow"]),
        "good_tool_patch_single_python": pick_best_with_meta(buckets["good_tool_patch_single_python"]),
        "good_inline_algorithm": pick_best_with_meta(buckets["good_inline_algorithm"]),
        "good_inline_file_task": pick_best_with_meta(buckets["good_inline_file_task"]),
    }

    outputs["good_tool_workflow_recovered"] = pick_best_with_meta(
        [s for s in buckets["good_tool_workflow"] if s.get("recovered_from_failed")]
    )
    outputs["recommended_tool_training"] = build_recommended_pack(outputs["good_tool_workflow"])

    file_map = {
        "recommended_tool_training.jsonl": outputs["recommended_tool_training"],
        "good_tool_workflow.jsonl": outputs["good_tool_workflow"],
        "good_tool_patch_single_python.jsonl": outputs["good_tool_patch_single_python"],
        "good_inline_algorithm.jsonl": outputs["good_inline_algorithm"],
        "good_inline_file_task.jsonl": outputs["good_inline_file_task"],
    }
    if outputs["good_tool_workflow_recovered"]:
        file_map["good_tool_workflow_recovered.jsonl"] = outputs["good_tool_workflow_recovered"]
    written: Dict[str, int] = {}
    for name, rows_out in file_map.items():
        path = out_dir / prefixed(prefix, name)
        write_jsonl(path, rows_out)
        written[prefixed(prefix, name)] = len(rows_out)

    # Per-run breakdown (before dedupe strip removes _source_file)
    run_counts: Dict[str, int] = defaultdict(int)
    for sample in buckets["good_tool_workflow"]:
        run_counts[sample.get("_source_file", "unknown")] += 1

    rec_by_source = Counter(s.get("source_run", "?") for s in outputs["recommended_tool_training"])
    manifest = {
        "prefix": prefix,
        "input_dir": str(input_dir),
        "total_rows_scanned": len(rows),
        "reject_counts": dict(rejects),
        "good_counts": {key: len(value) for key, value in outputs.items()},
        "written_files": written,
        "recommended_by_source": dict(rec_by_source),
        "tool_workflow_by_source": dict(sorted(run_counts.items())),
        "rows_by_source_file": dict(by_source),
        "source_files": sorted({row["_source_file"] for row in rows}),
    }
    with (out_dir / prefixed(prefix, "manifest.json")).open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--out-dir", type=Path, default=CURATED_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--include-failed-recovery",
        action="store_true",
        help="Promote recoverable trajectories from *_failed_attempts.jsonl",
    )
    args = parser.parse_args()

    manifest = curate(args.input_dir, args.out_dir, args.include_failed_recovery, args.prefix)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote curated outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
