#!/usr/bin/env python3
"""Curate all training sources into data/curated/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from curate_data_trajectories import curate as curate_data, merge_tool_training
from curate_import_samples import CURATED_DIR, IMPORT_DIR, curate as curate_import
from curate_samples import GENERATED_DIR, curate as curate_generated


def write_summary(
    out_dir: Path,
    import_manifest: dict,
    generated_manifest: dict,
    data_manifest: dict,
    merged: dict,
) -> None:
    imp = import_manifest["good_counts"]
    gen = generated_manifest["good_counts"]
    dat = data_manifest["good_counts"]
    rec = generated_manifest.get("recommended_by_source", {})

    summary = f"""# Curated training data (`data/curated/`)

All curated JSONL lives in this folder. Prefixes: `import_`, `generated_`, `data_`.

## Tool-use training (start here)

| File | Count | Notes |
|------|------:|-------|
| **`all_tool_training.jsonl`** | **{merged.get('count', 0)}** | **Merged** generated + data + import tool workflows + patch-tier (deduped) |
| `generated_recommended_tool_training.jsonl` | {gen.get('recommended_tool_training', 0)} | MLX `data/generated/` only |
| `data_good_tool_workflow.jsonl` | {dat.get('good_tool_workflow.jsonl', dat.get('good_tool_workflow', 0))} | `iteration_*` + `generated_trajectories_*` |
| `import_good_tool_workflow.jsonl` | {imp.get('good_tool_workflow', 0)} | Import Sample Sets |
| `generated_good_tool_workflow_recovered.jsonl` | {gen.get('good_tool_workflow_recovered', 0)} | Promoted from failed logs |

### `all_tool_training.jsonl` by source pack

| Pack | Count |
|------|------:|
{chr(10).join(f"| `{src}` | {cnt} |" for src, cnt in sorted(merged.get('by_pack', {}).items(), key=lambda x: -x[1]))}

### MLX recommended pack by run

| Source | Count |
|--------|------:|
{chr(10).join(f"| `{src}` | {cnt} |" for src, cnt in sorted(rec.items(), key=lambda x: -x[1]))}

## Inline Python (optional; no `write_file`)

| File | Count |
|------|------:|
| `data_good_inline_algorithm.jsonl` | {dat.get('good_inline_algorithm', 0)} |
| `data_good_inline_file_task.jsonl` | {dat.get('good_inline_file_task', 0)} |
| `import_good_inline_algorithm.jsonl` | {imp.get('good_inline_algorithm', 0)} |
| `import_good_inline_file_task.jsonl` | {imp.get('good_inline_file_task', 0)} |

## Other

| File | Count |
|------|------:|
| `import_good_tool_patch_single_python.jsonl` | {imp.get('good_tool_patch_single_python', 0)} |
| `generated_good_tool_patch_single_python.jsonl` | {gen.get('good_tool_patch_single_python', 0)} |

## Scan stats

- **Import:** {import_manifest['total_rows']} rows → `import_manifest.json`
- **Generated:** {generated_manifest['total_rows_scanned']} rows → `generated_manifest.json`
- **Iterations / trajectories:** {data_manifest['total_rows']} rows → `data_manifest.json`

Re-run: `python3 tools/curate_all.py`
"""
    (out_dir / "SUMMARY.md").write_text(summary)

    combined = {
        "import": import_manifest,
        "generated": generated_manifest,
        "data": data_manifest,
        "all_tool_training": merged,
    }
    with (out_dir / "manifest.json").open("w") as handle:
        json.dump(combined, handle, indent=2)


def main() -> None:
    out_dir = CURATED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    import_manifest = curate_import(IMPORT_DIR, out_dir, prefix="import")
    generated_manifest = curate_generated(
        GENERATED_DIR,
        out_dir,
        include_failed_recovery=True,
        prefix="generated",
    )
    data_manifest = curate_data(out_dir.parent, out_dir, prefix="data")
    merged = merge_tool_training(out_dir)
    data_manifest["all_tool_training"] = merged

    write_summary(out_dir, import_manifest, generated_manifest, data_manifest, merged)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "import": import_manifest,
                "generated": generated_manifest,
                "data": data_manifest,
                "all_tool_training": merged,
            },
            indent=2,
        )
    )
    print(f"\nAll curated files: {out_dir}")


if __name__ == "__main__":
    main()
