# Generation & Teacher Findings

Living notes from generation runs, Antigravity comparisons, and prompt tuning.  
**This file is tracked in git** (unlike `scratch/`, which is gitignored).

Update this doc when you complete an Antigravity benchmark task or learn something from a failed run.

---

## Antigravity log locations (this install)

| What | Path |
|------|------|
| Conversation artifacts (plans, task lists, walkthroughs) | `~/.gemini/antigravity/brain/<conversation-uuid>/` |
| Transcript & tool history | `~/.gemini/antigravity/brain/<uuid>/.system_generated/logs/transcript.jsonl` |
| Benchmark workspaces (optional) | `~/Documents/Antigravity Chat data/Test N/` |

Older exports may reference `~/.gemini/antigravity-ide/brain/` — **this machine uses `antigravity`, not `antigravity-ide`.**

---

## Benchmark tasks (failed MLX → Antigravity)

Source list for copy-paste: `scratch/failed_tasks_for_antigravity.md` (local only).

| Task | Status | Conversation UUID | Notes |
|------|--------|-------------------|--------|
| 1 — Path traversal / `file_server.py` | Done | `01c7a552-e12a-443d-bee5-8274fd1427f7` | See [Test 1](#test-1-path-traversal-security) below |
| 2 — `text_processor` patch | Done | `01404600-eea7-4d42-8e3b-64f57d3a8d9c` | See [Test 2](#test-2-text_processor-patch) below |
| 3 — `validators` refactor | Done | `fa15c93f-a779-4f03-9adb-afc7c2bcf573` | See [Test 3](#test-3-validators-refactor) below |
| 4 — `config_parser` JSON | Done | `caa90896-8c5b-4a4c-a910-a4a7d56bfb0e` | See [Test 4](#test-4-config_parser-json-validation) below |
| 5 — Buggy CSV `parser` | Done | `a1fe1089-6337-42b2-8d2a-7315f388d5a8` | See [Test 5](#test-5-buggy-csv-parser-patch) below |

After each Antigravity run: add the UUID to this table and append a short section below.

---

## Test 1: Path traversal security

**Task:** Create naive `src/file_server.py`, tests proving traversal, patch with `realpath`, rerun tests.

**MLX run:** `data/generated/qwencoder7.jsonl` — **0/5 task success** for this instruction (all 3 teachers failed on this task).

### Antigravity winning sequence (~85s, Gemini 3.5 Flash High)

1. `list_dir` workspace  
2. Plan + user approval (`implementation_plan.md`)  
3. `write_to_file` naive `file_server.py` (`os.path.join` only)  
4. `write_to_file` tests with **`test_naive_*`** — proves `../secret.txt` reads outside base  
5. `run_command` `python3 -m unittest` → **OK** (vulnerability demonstrated)  
6. `replace_file_content` patch `file_server.py` (`realpath` + prefix guard + `ValueError`)  
7. `replace_file_content` rewrite tests → `assertRaises(ValueError)` on traversal  
8. `run_command` rerun → **FAIL** (macOS `/private/var` vs `/var`)  
9. `replace_file_content` fix asserts to `os.path.realpath`  
10. `run_command` rerun → **OK**  
11. `walkthrough.md` + task checklist complete  

### MLX teachers on same task (qwencoder7)

| Teacher | Sequence | Failure |
|---------|----------|---------|
| Qwen3.5-9B | `write_file` → JSON fail → `none` | No tests, no `python` |
| Qwen3.6-35B | 2× `write_file` → `none` | Never ran tests |
| Gemma 31B | vuln → tests → `python` (exploit OK) → patch → `none` | **No `python` after patch** → workflow reject |

**Lesson:** Patch/security tasks need **two verification passes** — prove bug, then prove fix. Gemma was one step short; Antigravity reran tests three times.

### Tool mapping (Antigravity → sandbox JSON)

| Antigravity | Sandbox `action_type` |
|-------------|------------------------|
| `write_to_file` | `write_file` |
| `replace_file_content` | `read_file` + `write_file` |
| `run_command` (unittest) | `python` (must run again after patch) |
| `list_dir` | `list_dir` |

---

## Test 2: `text_processor` patch

**Task:** Buggy `reverse_words` (punctuation bug) → read/inspect → patch → verify with asserts.

**Workspace:** `~/Documents/Antigravity Chat data/Test 2/`  
**Logs:** `~/.gemini/antigravity/brain/01404600-eea7-4d42-8e3b-64f57d3a8d9c/.system_generated/logs/transcript.jsonl`  
**MLX run:** `qwencoder7_failed_attempts.jsonl` — all 3 teachers failed on this task.

### Antigravity winning sequence (~55s, Gemini 3.5 Flash High)

1. `write_to_file` buggy `src/text_processor.py` — reverses each word with `[::-1]` (punctuation moves: `"hello,"` → `",olleh"`)
2. `view_file` inspect the buggy implementation (satisfies “read to inspect”)
3. `write_to_file` `tests/test_text_processor.py` with asserts (`hello, world!` → `olleh, dlrow!`)
4. `run_command` `python3 tests/test_text_processor.py` → **FAIL** (`Expected 'olleh, dlrow!', got ',olleh !dlrow'`)
5. `replace_file_content` patch — two-pointer swap of **alphanumeric only** within each word
6. `run_command` rerun → **OK** (`All tests passed!`)
7. `view_file` final check + user-facing summary

No implementation plan or walkthrough artifact this time — faster, direct execution.

### Task interpretation note

Antigravity interpreted `reverse_words` as **reverse letters inside each word** while keeping punctuation positions (e.g. `hello, world!` → `olleh, dlrow!`). The prompt did not pin “reverse word order” vs “reverse characters per word.” MLX Gemma often used **word-order reversal** (`split` + reverse list), which is a different bug.

**For task generation:** Be explicit, e.g. “reverse the **order of words**” vs “reverse **characters within each word**.”

### MLX teachers on same task (qwencoder7)

| Teacher | Sequence | Failure |
|---------|----------|---------|
| Qwen3.5-9B | JSON fail → `none` | No files |
| Qwen3.6-35B | `write_file` → `read_file` → `none` | Never ran verification |
| Gemma 31B | `write_file` (word-order bug) → `write_file` tests → `python` → patch → `none` | Wrong semantics + **no post-patch `python`** |

### Lessons (adds to Test 1)

- **`read_file` / `view_file` before patch** when the task says “inspect” — Antigravity did this; encode as step 4 in patch flow.
- **First `python` should expect failure** on patch tasks — stderr/assert message is the observation teachers should learn from.
- **Simpler patch tasks** may not need plan approval; still need **two `python` turns** when the task says “patch then verify.”

---

## Test 3: Validators refactor

**Task:** Duplicated email/phone validation in `src/validators.py` → extract shared helper to `src/utils.py` → verify identical outputs.

**Workspace:** `~/Documents/Antigravity Chat data/Test 3/`  
**Logs:** `~/.gemini/antigravity/brain/fa15c93f-a779-4f03-9adb-afc7c2bcf573/.system_generated/logs/transcript.jsonl`  
**MLX run:** all 3 teachers failed — mostly JSON parse → `none` (no successful trajectory).

### Antigravity winning sequence (~1m 40s, Gemini 3.5 Flash High)

1. `write_to_file` `src/validators.py` — duplicated `validate_email` / `validate_phone` logic (type check, strip, length, regex, logging)
2. `write_to_file` **`src/validators_original.py`** — backup copy of pre-refactor code (clever: enables parity testing)
3. `write_to_file` `src/utils.py` — shared `validate_format(value, pattern, max_length)`
4. `write_to_file` overwrite `src/validators.py` — thin wrappers calling `validate_format`
5. `write_to_file` `src/__init__.py` — package marker for imports
6. `write_to_file` `verify.py` — imports **both** `validators_original` and refactored `validators`, asserts same bool per input
7. `run_command` `python3 verify.py` → **OK** (`Verification successful: Both modules produced identical outputs`)

No separate “failing then passing” run — refactor parity is the verification story.

### MLX teachers on same task (qwencoder7)

| Teacher | Failure |
|---------|---------|
| Qwen3.5-9B | `write_file` started → JSON fail → `none` |
| Qwen3.6-35B | JSON fail → `none` |
| Gemma 31B | JSON fail → `none` |

This task never got far enough in MLX to test refactor workflow — **JSON reliability** was the blocker, not refactor difficulty.

### Lessons (refactor-specific)

- **Behavior-preserving refactors:** save `*_original.py` (or read + copy) before rewriting, then compare old vs new in one verification script.
- **Multi-file refactor pattern:** `legacy.py` → `utils.py` (helper) → `module.py` (thin API) → `__init__.py` → `verify.py` → `python`.
- **Verification script** should import both implementations and loop test cases with `assert orig == refactored` — stronger than “run pytest once.”
- **Task generator hint:** For refactor tasks, explicitly ask to “keep a copy of the pre-refactor module and assert identical outputs on a shared test matrix.”

---

## Test 4: `config_parser` JSON validation

**Task:** `src/config_parser.py` reads/validates JSON config → `tests/test_config_parser.py` for valid/invalid inputs → **list workspace** → run tests proving malformed JSON raises correctly.

**Workspace:** `~/Documents/Antigravity Chat data/Test 4/`  
**Logs:** `~/.gemini/antigravity/brain/caa90896-8c5b-4a4c-a910-a4a7d56bfb0e/.system_generated/logs/transcript.jsonl`  
**MLX run:** all 3 teachers failed on this task in `qwencoder7` (per scratch list; dominant mode likely JSON parse → `none`).

### Antigravity winning sequence (~34s, Gemini 3.5 Flash High)

1. `list_dir` workspace → empty  
2. `write_to_file` `src/config_parser.py` — `load_config()`, custom `ConfigValidationError`, schema: `host`, `port` (1–65535, rejects `bool` as int), optional `debug` (default `False`), nested `database.username` / `database.password`  
3. `write_to_file` `tests/test_config_parser.py` — **19** `unittest` cases with `tempfile` JSON fixtures; `assertRaises` for `json.JSONDecodeError`, `FileNotFoundError`, `ConfigValidationError`  
4. `list_dir` root → `src/`, `tests/`  
5. `list_dir` `src/` → `config_parser.py`  
6. `list_dir` `tests/` → `test_config_parser.py`  
7. `run_command` `python3 -m unittest discover -s tests` → **OK** (`Ran 19 tests in 0.006s`)

No patch cycle — greenfield implementation. No plan/walkthrough artifacts.

### Task compliance

The prompt explicitly asked to **list the workspace**; Antigravity satisfied that with three listing passes after creating files (not just one empty check at the start).

### MLX teachers on same task (qwencoder7)

| Teacher | Failure (expected) |
|---------|-------------------|
| Qwen3.5-9B | JSON parse → `none` before full trajectory |
| Qwen3.6-35B | Same |
| Gemma 31B | Same |

Large combined `write_file` payloads (parser + 19-test suite) are especially hostile to single-turn JSON extraction at 1024 `max_tokens`.

### Lessons (validation / parser tasks)

- **Greenfield parsers:** one correct implementation + broad `unittest` matrix; single passing `python` is enough (no fail-then-patch flow).
- **Error typing:** distinguish `json.JSONDecodeError` (malformed JSON) vs custom `ConfigValidationError` (schema) vs `FileNotFoundError` — tests should `assertRaises` each explicitly.
- **Fixtures:** `tempfile.TemporaryDirectory` + helper `create_temp_config()` avoids checking in sample JSON files.
- **`list_dir` when asked:** map “list the workspace” to sandbox `list_dir` after files exist, not only at turn 0.
- **Task generator:** If the sandbox task says “list workspace,” keep that phrase so trajectories learn `list_dir` as a required step.
- **MLX backlog:** This task type benefits most from **JSON retry** and **higher `max_tokens`** — not from patch-workflow prompts.

---

## Test 5: Buggy CSV `parser` patch

**Task:** Buggy `src/parser.py` → `tests/test_parser.py` exposes bug → **read** `parser.py` → patch → rerun tests successfully.

**Workspace:** `~/Documents/Antigravity Chat data/Test 5/`  
**Logs:** `~/.gemini/antigravity/brain/a1fe1089-6337-42b2-8d2a-7315f388d5a8/.system_generated/logs/transcript.jsonl`  
**MLX run:** DeepSeek-V4-Flash-2bit-DQ, Qwen3.6-27B, Qwen3.6-35B — all failed in `qwencoder7` (per scratch list).

### Antigravity winning sequence (~2m, Gemini 3.5 Flash High)

1. `list_dir` → empty  
2. `write_to_file` `src/parser.py` — naive `line.split(",")` (ignores quotes)  
3. `write_to_file` `tests/test_parser.py` — plain `assert` tests (`test_simple_csv`, `test_quoted_comma_csv`)  
4. `run_command` `python3 -m pytest …` → **FAIL** (`No module named pytest`)  
5. `write_to_file` rewrite tests as **`unittest`** (no external deps)  
6. `run_command` `python3 tests/test_parser.py` → **FAIL** (`ModuleNotFoundError: No module named 'src'`)  
7. `run_command` `PYTHONPATH=. python3 tests/test_parser.py` → **FAIL** (`test_quoted_comma_csv` — splits `"Doe, John"` into extra fields)  
8. `view_file` `src/parser.py` (satisfies “read before patch”)  
9. `replace_file_content` patch — `import csv` + `csv.reader(lines)`  
10. `run_command` `PYTHONPATH=. python3 tests/test_parser.py` → **OK** (`Ran 2 tests … OK`)

### Environment recovery (worth teaching)

| Problem | Fix Antigravity used |
|---------|----------------------|
| `pytest` not installed | Switch to stdlib `unittest` |
| `from src.parser import …` fails | `PYTHONPATH=.` on the command line |

MLX sandbox should assume **no pytest** and either set `PYTHONPATH` for `src/` imports or use `sys.path.insert` in tests (Test 4 used the latter).

### Bug & fix

- **Bug:** comma-split without respecting double-quoted fields (`"Doe, John",Engineer` → three columns).  
- **Fix:** delegate to stdlib `csv.reader` — minimal, correct patch (not a hand-rolled state machine).

### MLX teachers on same task (qwencoder7)

| Teacher | Notes |
|---------|--------|
| DeepSeek-V4-Flash-2bit-DQ | Failed (task-specific teacher set differs from Tasks 1–4) |
| Qwen3.6-27B-oQ8-mtp | Failed |
| Qwen3.6-35B | Failed |

Same patch-workflow failure mode as Tests 1–2 when MLX *did* produce trajectories: missing post-patch `python`, or JSON parse abort.

### Lessons (canonical patch task)

- **Ideal patch arc:** buggy code → tests → **`python` must fail** on the exposing case → `read_file` → patch → **`python` must pass**.  
- Antigravity hit both required `python` outcomes (steps 7 and 10); Gemma on Test 1 skipped the second.  
- **Prefer `unittest`** in generated tasks/sandbox — avoids dependency failures.  
- **Import path:** document `PYTHONPATH=.` or `sys.path` in teacher prompt when layout is `src/` + `tests/`.  
- **Quoted-field bugs** are a good generator template: simple naive impl, one failing integration test, stdlib fix.

**All five Antigravity benchmarks complete** — use this table + sections as the reference for the next MLX generation prompt/config pass.

---

## qwencoder7 run summary (2026-05-28)

| Metric | Value |
|--------|--------|
| Successful trajectories | 5 / ~10 tasks |
| Failed teacher attempts | 23 |
| Dominant failure mode | **21/23** — `extract_first_json` failed → injected `none` → discard |
| Winners | Gemma 31B (3), Qwen3.6 (2) |
| Weak teacher | Qwen3.5-9B first in order — burns attempts on JSON parse |

### Recommended config (until JSON retry is implemented)

```json
"teacher_attempt_order": [3, 2, 1]
```

(Gemma first, Qwen3.6 second, Qwen3.5 last — or drop 9B from generation entirely.)

### Backlog (code / config)

- [ ] Retry same turn on JSON parse failure (do not inject `none` immediately)  
- [ ] Raise trajectory `max_tokens` from 1024 → 2048 for large `write_file` payloads  
- [x] Patch/security prompt guidance in `generator.py` (see that file)  
- [ ] DeepSeek V4 — blocked until `mlx-lm` supports `model_type: deepseek_v4` on PyPI  

---

## Prompt rules encoded in code

The teacher system prompt in `mlx_foundation/src/generator/generator.py` includes guidance derived from Tests 1–5 and qwencoder7. When changing prompts, update **both** the code and the “Prompt rules” bullet list in this section.

**Patch / security / buggy-module tasks (Tests 1, 2, 5):**

1. `write_file` initial (buggy if required) implementation  
2. `write_file` tests that demonstrate or fail on the bug  
3. `python` — run tests; **must observe failure** on the exposing case when task says “expose the bug”  
4. `read_file` before patch when the task requires it  
5. `write_file` fixed implementation (or patch via read+write)  
6. `write_file` updated tests only if assertions must change (Test 1 security)  
7. `python` — **rerun** tests; required after any patch  
8. `none` only after step 7 succeeds  

Use **stdlib `unittest`** (not `pytest`). For `src/` package imports from `tests/`, use `sys.path.insert` in the test file or run with `PYTHONPATH=.`  

**Refactor / behavior-preserving tasks (from Test 3):**

1. `write_file` duplicated or legacy implementation  
2. `write_file` backup copy (e.g. `module_original.py`) if parity verification is required  
3. `write_file` shared helper module (`utils.py`)  
4. `write_file` refactored thin wrappers  
5. `write_file` `__init__.py` when using package imports  
6. `write_file` verification script that imports both versions and asserts identical outputs  
7. `python` run verification — `none` only after success  

**Validation / parser tasks (from Test 4):**

1. `write_file` implementation with explicit exception types (`JSONDecodeError`, custom validation error, `FileNotFoundError` as appropriate)  
2. `write_file` `unittest` module using tempfile fixtures and `assertRaises` per error class  
3. `list_dir` when the task requires listing the workspace (after files exist)  
4. `python` via `unittest discover` or module run — `none` only after all tests pass  

**macOS:** Compare paths with `os.path.realpath()`, not `os.path.abspath()`, when asserting equality in tests.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-28 | Initial doc: Test 1 Antigravity analysis, qwencoder7 summary, prompt rules |
| 2026-05-28 | Test 2: text_processor patch analysis; task wording ambiguity note |
| 2026-05-28 | Test 3: validators refactor; backup+parity verify pattern |
| 2026-05-28 | Test 4: config_parser; greenfield unittest + list_dir compliance |
| 2026-05-28 | Test 5: CSV parser patch; unittest + PYTHONPATH; all benchmarks done |
| 2026-05-28 | Pipeline: unittest stderr OK, patch expected-fail turns, JSON retry, gated force_failure |
