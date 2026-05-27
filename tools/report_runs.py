#!/usr/bin/env python3
"""Generate a self-contained HTML report for trajectory generation runs."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ACTION_ORDER = ["write_file", "read_file", "list_dir", "python", "none"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return value or "trajectory-report"


def short_teacher(path: str | None) -> str:
    if not path:
        return "unknown"
    if path == "fallback":
        return "fallback"
    return Path(path).name


def action_sequence(sample: dict[str, Any]) -> list[str]:
    return [
        (turn.get("action") or {}).get("type", "unknown")
        for turn in sample.get("turns", [])
    ]


def classify_task(instruction: str) -> str:
    text = instruction.lower()
    if any(k in text for k in ["index.html", "style.css", "frontend", "app.js", "main.js"]):
        return "frontend"
    if any(k in text for k in ["security", "sql injection", "path traversal", "secret", "sanitize", "validator"]):
        return "security"
    if any(k in text for k in ["patch", "buggy", "failing", "fails", "repair"]):
        return "repair"
    if "refactor" in text:
        return "refactor"
    if any(k in text for k in ["src/", "tests/", "directory structure", "package", "module"]):
        return "multi-file"
    return "algorithm"


def workflow_quality(sample: dict[str, Any]) -> dict[str, bool]:
    seq = action_sequence(sample)
    has_write = "write_file" in seq
    has_python = "python" in seq
    has_inspect = "read_file" in seq or "list_dir" in seq
    has_none = "none" in seq
    verified = any(
        (turn.get("action") or {}).get("type") == "python"
        and (turn.get("observation") or {}).get("success", False)
        for turn in sample.get("turns", [])
    )
    return {
        "tool_workflow": has_write and has_python,
        "inspect_before_or_after": has_inspect,
        "verified": verified,
        "completed": has_none,
    }


def build_summary(paths: list[Path]) -> dict[str, Any]:
    rows_by_file = {str(path): load_jsonl(path) for path in paths}
    rows = [row for file_rows in rows_by_file.values() for row in file_rows]

    action_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    curriculum_counts: Counter[str] = Counter()
    turn_counts = []
    failed_attempt_total = 0
    quality_counts: Counter[str] = Counter()

    for sample in rows:
        teacher_counts[short_teacher(sample.get("teacher_model"))] += 1
        curriculum_counts[classify_task(sample.get("instruction", ""))] += 1
        turns = sample.get("turns", [])
        turn_counts.append(len(turns))
        failed_attempt_total += int(sample.get("failed_attempt_count", 0) or 0)
        action_counts.update(action_sequence(sample))
        for key, ok in workflow_quality(sample).items():
            if ok:
                quality_counts[key] += 1

    total = len(rows)
    success_count = sum(1 for row in rows if row.get("sandbox_success", True))
    fallback_count = sum(1 for row in rows if row.get("teacher_model") == "fallback")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": list(rows_by_file.keys()),
        "total_samples": total,
        "sandbox_success_count": success_count,
        "sandbox_success_rate": success_count / total if total else 0,
        "fallback_count": fallback_count,
        "failed_attempt_total": failed_attempt_total,
        "avg_turns": mean(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
        "teacher_counts": dict(teacher_counts),
        "action_counts": dict(action_counts),
        "curriculum_counts": dict(curriculum_counts),
        "quality_counts": dict(quality_counts),
        "samples": rows,
    }


def pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "0%"
    return f"{(numerator / denominator) * 100:.0f}%"


def bars(title: str, data: dict[str, int], color: str = "#00f3ff") -> str:
    if not data:
        return chart_card(title, "<p class='muted'>No data.</p>")
    max_value = max(data.values()) or 1
    rows = []
    for key, value in sorted(data.items(), key=lambda item: (-item[1], item[0])):
        width = max(2, value / max_value * 100)
        rows.append(
            f"<div class='bar-row'><span>{html.escape(str(key))}</span>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%;background:{color}'></div></div>"
            f"<strong>{value}</strong></div>"
        )
    return chart_card(title, "\n".join(rows))


def chart_card(title: str, body: str) -> str:
    return f"<section class='card'><h2>{html.escape(title)}</h2>{body}</section>"


def metric(label: str, value: str, sub: str = "") -> str:
    return (
        "<div class='metric'>"
        f"<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>"
        f"<small>{html.escape(sub)}</small>"
        "</div>"
    )


def sample_table(samples: list[dict[str, Any]]) -> str:
    rows = []
    for idx, sample in enumerate(samples, 1):
        instruction = html.escape(sample.get("instruction", ""))
        teacher = html.escape(short_teacher(sample.get("teacher_model")))
        seq = " -> ".join(action_sequence(sample))
        quality = workflow_quality(sample)
        flags = []
        if sample.get("teacher_model") == "fallback":
            flags.append("fallback")
        if not quality["tool_workflow"]:
            flags.append("no tool workflow")
        if not quality["verified"]:
            flags.append("no verification")
        if int(sample.get("failed_attempt_count", 0) or 0):
            flags.append(f"{sample.get('failed_attempt_count')} recovered failures")
        flag_text = ", ".join(flags) if flags else "good"
        rows.append(
            "<tr>"
            f"<td>{idx}</td><td>{instruction}</td><td>{teacher}</td>"
            f"<td>{len(sample.get('turns', []))}</td><td><code>{html.escape(seq)}</code></td>"
            f"<td>{html.escape(flag_text)}</td>"
            "</tr>"
        )
    return (
        "<section class='card wide'><h2>Sample Review</h2>"
        "<table><thead><tr><th>#</th><th>Instruction</th><th>Teacher</th><th>Turns</th><th>Actions</th><th>Flags</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def render_html(summary: dict[str, Any]) -> str:
    total = summary["total_samples"]
    quality = summary["quality_counts"]
    metrics = "".join(
        [
            metric("Samples", str(total), "accepted rows in JSONL"),
            metric("Sandbox Success", pct(summary["sandbox_success_count"], total), f"{summary['sandbox_success_count']} / {total}"),
            metric("Tool Workflow", pct(quality.get("tool_workflow", 0), total), "write_file + python"),
            metric("Verified", pct(quality.get("verified", 0), total), "python action succeeded"),
            metric("Avg Turns", f"{summary['avg_turns']:.1f}", f"max {summary['max_turns']}"),
            metric("Recovered Failures", str(summary["failed_attempt_total"]), "linked failed attempts"),
        ]
    )

    action_counts = {key: summary["action_counts"].get(key, 0) for key in ACTION_ORDER if summary["action_counts"].get(key, 0)}
    other_actions = {
        key: value
        for key, value in summary["action_counts"].items()
        if key not in ACTION_ORDER
    }
    action_counts.update(other_actions)

    files = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in summary["files"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trajectory Run Report</title>
  <style>
    :root {{ --bg:#070810; --card:#101522; --text:#e8f7ff; --muted:#8ea4b8; --cyan:#00f3ff; --pink:#ff2b7a; --violet:#8b5cf6; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:radial-gradient(circle at top left, #16213a, var(--bg) 48%); color:var(--text); }}
    main {{ max-width:1200px; margin:0 auto; padding:32px; }}
    header {{ margin-bottom:24px; }}
    h1 {{ margin:0 0 8px; letter-spacing:.08em; text-transform:uppercase; text-shadow:0 0 24px rgba(0,243,255,.45); }}
    h2 {{ margin:0 0 16px; font-size:18px; }}
    code {{ color:#b9f7ff; }}
    .muted, small {{ color:var(--muted); }}
    .metrics {{ display:grid; grid-template-columns:repeat(6, 1fr); gap:14px; margin:24px 0; }}
    .metric, .card {{ background:rgba(16,21,34,.82); border:1px solid rgba(0,243,255,.18); border-radius:18px; box-shadow:0 0 30px rgba(0,0,0,.22); }}
    .metric {{ padding:16px; min-height:116px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .metric strong {{ display:block; margin:8px 0 4px; font-size:30px; color:var(--cyan); }}
    .grid {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:16px; }}
    .card {{ padding:18px; overflow:hidden; }}
    .wide {{ grid-column:1 / -1; }}
    .bar-row {{ display:grid; grid-template-columns:160px 1fr 48px; gap:12px; align-items:center; margin:10px 0; font-size:14px; }}
    .bar-track {{ height:12px; background:#1d2636; border-radius:999px; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; box-shadow:0 0 14px currentColor; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:10px; border-bottom:1px solid rgba(255,255,255,.08); vertical-align:top; }}
    th {{ text-align:left; color:var(--muted); text-transform:uppercase; font-size:11px; letter-spacing:.08em; }}
    ul {{ margin:0; padding-left:18px; }}
    @media (max-width:900px) {{ .metrics {{ grid-template-columns:repeat(2, 1fr); }} .grid {{ grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:120px 1fr 40px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Trajectory Run Report</h1>
    <p class="muted">Generated {html.escape(summary["generated_at"])}</p>
    <ul>{files}</ul>
  </header>
  <section class="metrics">{metrics}</section>
  <section class="grid">
    {bars("Action Mix", action_counts, "var(--cyan)")}
    {bars("Teacher Contribution", summary["teacher_counts"], "var(--pink)")}
    {bars("Curriculum Breakdown", summary["curriculum_counts"], "var(--violet)")}
    {bars("Quality Signals", dict(summary["quality_counts"]), "#22c55e")}
    {sample_table(summary["samples"])}
  </section>
</main>
</body>
</html>
"""


def write_report(paths: list[Path], output_dir: Path | None) -> Path:
    summary = build_summary(paths)
    if output_dir is None:
        stem = slugify(Path(summary["files"][0]).stem if summary["files"] else "trajectory-report")
        output_dir = Path("reports") / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_for_json = dict(summary)
    summary_for_json["samples"] = [
        {
            "task_id": sample.get("task_id"),
            "instruction": sample.get("instruction"),
            "teacher_model": short_teacher(sample.get("teacher_model")),
            "sandbox_success": sample.get("sandbox_success", True),
            "failed_attempt_count": sample.get("failed_attempt_count", 0),
            "turn_count": len(sample.get("turns", [])),
            "actions": action_sequence(sample),
            "quality": workflow_quality(sample),
        }
        for sample in summary["samples"]
    ]

    (output_dir / "summary.json").write_text(json.dumps(summary_for_json, indent=2), encoding="utf-8")
    report_path = output_dir / "report.html"
    report_path.write_text(render_html(summary), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML dashboard for trajectory JSONL files.")
    parser.add_argument("jsonl", nargs="+", type=Path, help="Trajectory JSONL file(s) to summarize.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write report.html and summary.json.")
    args = parser.parse_args()

    missing = [str(path) for path in args.jsonl if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input file(s): {', '.join(missing)}")

    report_path = write_report(args.jsonl, args.output_dir)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
