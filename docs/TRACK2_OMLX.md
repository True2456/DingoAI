# Track 2: oMLX / Claude Code wire format

Two parallel tracks share the **same sandbox and curation**, but different **training/generation wire formats**:

| Track | Training `actions` after `<channel|>` | Use case |
|-------|----------------------------------------|----------|
| **Dingo** (default) | `write_file: {...} \| python: ...` | Dingo web UI, JSON agent gen |
| **oMLX** | `<|tool_call>call:Read{file_path: <|"|>...<|"|>}<tool_call|>` | Claude Code VS Code + oMLX |

You do **not** need a separate repo or host — only a **convert step** and/or **`wire_format=omlx_claude`** on generate/train.

## In the DingoAI web UI

1. Set **Training track** to **oMLX / Claude Code** (default stays **Dingo** — nothing is overwritten).
2. Optional: **Build oMLX pack from Dingo curated** — writes only `all_omlx_tool_training.jsonl`.
3. **Train** uses the oMLX pack path and passes `--wire-format omlx_claude` to the job.

## Fast path (CLI)

```bash
python3 tools/curate_all.py
python3 tools/build_omlx_training_pack.py
./run_train_only.sh data/curated/all_omlx_tool_training.jsonl 520 models/mlx_self_training/pilot_v4_omlx "" omlx_claude
```

Fuse adapter → point `~/.claude/settings.json` Opus at the new fused folder.

## New trajectories (native oMLX format)

1. In Dingo UI or `config/default_config.json`, set:
   ```json
   "generation": { "wire_format": "omlx_claude", ... }
   ```
2. Use preset **`omlx-claude-prompt`** (tasks say Read/Write/Edit/Bash) or **`omlx-claude-combined`**.
3. Generate as usual (`run_generate.sh` / web UI).
4. Curate + merge into your pack; optional `build_omlx_training_pack.py` if rows are still Dingo-shaped from old runs.
5. Train with `--wire-format omlx_claude`.

## What was added

- `mlx_foundation/src/agent_wire_formats.py` — convert + parse Gemma `call:Tool{args}`
- `tools/build_omlx_training_pack.py` — Dingo pack → `all_omlx_tool_training.jsonl`
- Sandbox: `Read`/`Write`/`Edit`/`Bash` aliases (`edit`, `bash`)
- Generator + trainer: `wire_format` flag

## Preset only?

A **preset alone is not enough**. Presets only change **task wording**. Track 2 also needs:

- **Training** on `call:Read{...}` completions (`--wire-format omlx_claude`), and/or
- **Generation** with `wire_format: omlx_claude` so new trajectories are native.

## Claude Code after training

- Keep using **oMLX** (no custom VS Code host).
- Lower temperature on fused model in oMLX model settings (~0.3).
- Consider disabling thinking budget for coding on that model.
