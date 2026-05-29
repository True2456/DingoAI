#!/usr/bin/env python3
"""Promote conservatively recoverable trajectories from *_failed_attempts.jsonl into success JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_import_samples import is_json_abort  # noqa: E402


def task_id(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]


def action_types(traj: Dict[str, Any]) -> List[str]:
    return [(t.get("action") or {}).get("type") for t in traj.get("turns") or []]


def last_python_observation(traj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last = None
    for turn in traj.get("turns") or []:
        if (turn.get("action") or {}).get("type") == "python":
            last = turn.get("observation") or {}
    return last


def last_python_passed(traj: Dict[str, Any]) -> bool:
    obs = last_python_observation(traj) or {}
    combined = (obs.get("stdout") or "") + "\n" + (obs.get("stderr") or "")
    if not combined.strip():
        return False
    if any(
        marker in combined
        for marker in ("FAILED", "AssertionError", "Traceback (most recent call last):")
    ):
        return False
    return (
        obs.get("success") is True
        or "OK" in combined
        or "passed" in combined.lower()
        or "Verified" in combined
    )


def recoverable_reason(traj: Dict[str, Any]) -> Optional[str]:
    if not traj or not traj.get("turns"):
        return None
    if traj.get("sandbox_success"):
        return None
    if is_json_abort(traj):
        return None

    types = action_types(traj)
    if "write_file" not in types or "python" not in types or types[-1] != "none":
        return None
    if not last_python_passed(traj):
        return None

    obs_blob = traj.get("observation") or ""
    if "file workflow did not include" in obs_blob:
        return "workflow_reject_false_positive"
    return "stderr_or_patch_false_reject"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def promote_pair(success_path: Path, failed_path: Path, dry_run: bool) -> Dict[str, Any]:
    successes = load_jsonl(success_path)
    existing_instructions = {row.get("instruction") for row in successes}

    candidates: List[Tuple[int, str, Dict[str, Any], Dict[str, Any]]] = []
    if failed_path.exists():
        for row in load_jsonl(failed_path):
            traj = row.get("trajectory") or {}
            reason = recoverable_reason(traj)
            if not reason:
                continue
            instruction = row.get("instruction") or traj.get("instruction")
            if not instruction or instruction in existing_instructions:
                continue
            candidates.append(
                (int(row.get("attempt") or 99), reason, instruction, row)
            )

    # Best attempt per instruction (lowest attempt number)
    by_instruction: Dict[str, Tuple[int, str, Dict[str, Any], Dict[str, Any]]] = {}
    for item in sorted(candidates, key=lambda x: x[0]):
        instruction = item[2]
        if instruction not in by_instruction:
            by_instruction[instruction] = item

    promoted: List[Dict[str, Any]] = []
    for instruction, (_, reason, _, fail_row) in by_instruction.items():
        traj = dict(fail_row.get("trajectory") or {})
        traj["sandbox_success"] = True
        tid = fail_row.get("task_id") or task_id(instruction)
        sample = {
            **traj,
            "task_id": tid,
            "failed_attempt_count": max(0, int(fail_row.get("attempt") or 1) - 1),
            "failed_attempts": [],
            "recovered_from": str(failed_path.name),
            "recovery_reason": reason,
        }
        promoted.append(sample)
        existing_instructions.add(instruction)

    merged = successes + promoted
    if not dry_run and promoted:
        write_jsonl(success_path, merged)

    return {
        "success_path": str(success_path),
        "failed_path": str(failed_path),
        "before": len(successes),
        "promoted": len(promoted),
        "after": len(merged),
        "instructions": [p.get("instruction", "")[:80] for p in promoted],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "generated",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("pairs", nargs="*", help="Optional success.jsonl paths (failed derived)")
    args = parser.parse_args()

    default_pairs = [
        args.root / "New Prompt1.jsonl",
        args.root / "New Prompt2.jsonl",
        args.root / "qwencoder9.jsonl",
    ]
    targets = [Path(p) for p in args.pairs] if args.pairs else default_pairs

    total_promoted = 0
    for success_path in targets:
        failed_path = success_path.with_name(
            success_path.stem + "_failed_attempts" + success_path.suffix
        )
        if not failed_path.exists() and not success_path.exists():
            continue
        report = promote_pair(success_path, failed_path, args.dry_run)
        if report["promoted"]:
            print(json.dumps(report, indent=2))
            total_promoted += report["promoted"]

    print(f"\nTotal promoted: {total_promoted}" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
