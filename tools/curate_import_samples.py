#!/usr/bin/env python3
"""Curate JSONL files from Import Sample Sets into tiered training-ready outputs."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMPORT_DIR = Path(__file__).resolve().parents[1] / "Import Sample Sets"
CURATED_DIR = Path(__file__).resolve().parents[1] / "data" / "curated"
DEFAULT_OUT = CURATED_DIR
DEFAULT_PREFIX = "import"


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


def python_codes(sample: Dict[str, Any]) -> List[str]:
    turns = sample.get("turns")
    codes: List[str] = []
    if isinstance(turns, list) and turns:
        for turn in turns:
            if (turn.get("action") or {}).get("type") == "python":
                codes.append((turn.get("action") or {}).get("input", ""))
        return codes
    for action_type, action_input in parse_legacy_actions(sample.get("actions", "")):
        if action_type == "python":
            codes.append(action_input)
    return codes


_JSON_ABORT_THOUGHTS = frozenset(
    {
        "i failed to generate valid json.",
        "i failed to generate valid json",
    }
)


def _turn_thought(turn: Dict[str, Any]) -> str:
    return (turn.get("thought") or "").strip().lower()


def _completed_file_workflow(sample: Dict[str, Any]) -> bool:
    types = action_types(sample)
    return bool(
        types
        and types[-1] == "none"
        and "write_file" in types
        and "python" in types
        and sample.get("sandbox_success") is True
    )


def is_json_abort(sample: Dict[str, Any]) -> bool:
    """
    True only when the trajectory actually ended on a JSON-parse fallback turn.

    Ignores stale abort phrases left in joined legacy `thought` fields when structured
    `turns` show a completed sandbox workflow (common after premature-none retries).
    """
    turns = sample.get("turns")
    if isinstance(turns, list) and turns:
        last = turns[-1]
        last_type = (last.get("action") or {}).get("type")
        if last_type == "none" and _turn_thought(last) in _JSON_ABORT_THOUGHTS:
            return True
        if _completed_file_workflow(sample):
            return False
        if sample.get("sandbox_success") is True:
            return False
        if any(_turn_thought(t) in _JSON_ABORT_THOUGHTS for t in turns):
            return True
        return False

    if sample.get("sandbox_success") is True:
        return False
    blob = json.dumps(sample).lower()
    return "failed to generate valid json" in blob


def is_fallback_stub(sample: Dict[str, Any]) -> bool:
    blob = json.dumps(sample).lower()
    return "src/fallback.py" in blob and "def solve" in blob


def python_verified(sample: Dict[str, Any]) -> bool:
    turns = sample.get("turns")
    if isinstance(turns, list) and turns:
        for turn in turns:
            if (turn.get("action") or {}).get("type") != "python":
                continue
            obs = turn.get("observation") or {}
            combined = (obs.get("stdout") or "") + "\n" + (obs.get("stderr") or "")
            if any(marker in combined for marker in ("AssertionError", "Traceback (most recent call last):", "FAILED (")):
                continue
            if obs.get("success") or obs.get("verification_passed"):
                return True
            if ("OK" in combined or "passed" in combined.lower() or "Verified" in combined) and "FAILED" not in combined:
                return True
    observation = str(sample.get("observation", "")).lower()
    if any(marker in observation for marker in ("assertionerror", "traceback", "failed (")):
        return False
    return any(token in observation for token in ("verified", "all tests passed", "environment complete", "successfully"))


def classify(sample: Dict[str, Any]) -> str:
    if not (sample.get("instruction") or "").strip():
        return "reject_no_instruction"
    if not (sample.get("thought") or "").strip() and not sample.get("turns"):
        return "reject_empty_thought"
    if is_json_abort(sample):
        return "reject_json_abort"
    if sample.get("sandbox_success") is False:
        return "reject_sandbox_failed"
    if is_fallback_stub(sample):
        return "reject_fallback_stub"

    types = action_types(sample)
    if not types:
        return "reject_no_actions"
    if types[-1] != "none":
        return "reject_incomplete"
    if "python" not in types:
        return "reject_no_python"

    for code in python_codes(sample):
        if not code.strip():
            return "reject_empty_python"
        try:
            ast.parse(code)
        except SyntaxError:
            return "reject_bad_python_syntax"

    if not python_verified(sample):
        return "reject_python_unverified"

    instruction = (sample.get("instruction") or "").lower()
    file_task = any(
        keyword in instruction
        for keyword in (
            "src/",
            "tests/",
            "write ",
            "create src",
            "patch",
            "buggy",
            "refactor",
            "index.html",
            "list the workspace",
            "list_dir",
            "read_file",
            "verify",
        )
    )

    if "write_file" in types and "python" in types:
        if any(keyword in instruction for keyword in ("buggy", "patch", "expose", "rerun", "failing")):
            if types.count("python") < 2:
                return "good_tool_patch_single_python"
        return "good_tool_workflow"

    if file_task:
        return "good_inline_file_task"

    return "good_inline_algorithm"


def score(sample: Dict[str, Any]) -> int:
    types = action_types(sample)
    value = len(types)
    value += types.count("write_file") * 4
    value += types.count("python") * 5
    value += types.count("read_file") * 2
    value += types.count("list_dir")
    if sample.get("turns"):
        value += 8
    return value


def load_rows(import_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(import_dir.glob("*.jsonl")):
        if path.name.endswith(".bak"):
            continue
        # Subfolders (e.g. _archived_fallback_stubs/) are excluded from import curation.
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


def strip_internal(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in sample.items() if not key.startswith("_")}


def pick_best_per_instruction(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for sample in rows:
        instruction = sample.get("instruction", "")
        if instruction not in best or score(sample) > score(best[instruction]):
            best[instruction] = sample
    return [strip_internal(best[instruction]) for instruction in sorted(best)]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prefixed(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def curate(import_dir: Path, out_dir: Path, prefix: str = DEFAULT_PREFIX) -> Dict[str, Any]:
    rows = load_rows(import_dir)
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "good_tool_workflow": [],
        "good_tool_patch_single_python": [],
        "good_inline_algorithm": [],
        "good_inline_file_task": [],
    }
    rejects: Counter[str] = Counter()

    for sample in rows:
        label = classify(sample)
        if label.startswith("reject_"):
            rejects[label] += 1
            continue
        buckets[label].append(sample)

    outputs = {
        "good_tool_workflow": pick_best_per_instruction(buckets["good_tool_workflow"]),
        "good_tool_patch_single_python": pick_best_per_instruction(buckets["good_tool_patch_single_python"]),
        "good_inline_algorithm": pick_best_per_instruction(buckets["good_inline_algorithm"]),
        "good_inline_file_task": pick_best_per_instruction(buckets["good_inline_file_task"]),
    }

    files = {
        "good_tool_workflow.jsonl": outputs["good_tool_workflow"],
        "good_tool_patch_single_python.jsonl": outputs["good_tool_patch_single_python"],
        "good_inline_algorithm.jsonl": outputs["good_inline_algorithm"],
        "good_inline_file_task.jsonl": outputs["good_inline_file_task"],
    }
    written = {}
    for name, rows_out in files.items():
        path = out_dir / prefixed(prefix, name)
        write_jsonl(path, rows_out)
        written[prefixed(prefix, name)] = len(rows_out)

    manifest = {
        "prefix": prefix,
        "import_dir": str(import_dir),
        "total_rows": len(rows),
        "reject_counts": dict(rejects),
        "good_counts": {key: len(value) for key, value in outputs.items()},
        "written_files": written,
        "source_files": sorted({row["_source_file"] for row in rows}),
    }
    manifest_path = out_dir / prefixed(prefix, "manifest.json")
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-dir", type=Path, default=IMPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    args = parser.parse_args()

    manifest = curate(args.import_dir, args.out_dir, args.prefix)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote curated outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
