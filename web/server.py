#!/usr/bin/env python3
"""Local web GUI server for the MLX self-training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
CONFIG_PATH = ROOT / "config" / "default_config.json"
PRESETS_PATH = ROOT / "config" / "dingo_presets.json"
PYTHON = ROOT / "mlx_foundation" / "venv" / "bin" / "python"
MAIN = ROOT / "mlx_foundation" / "src" / "main.py"

JOBS: dict[str, dict] = {}

SESSION_STATS: dict = {
    "train_tokens_total": 0,
    "gen_tokens_total": 0,
    "last_train_tokens_by_job": {},
    "train_tokens_per_sec": None,
    "gen_tokens_per_sec": None,
}

TRAIN_STATS_RE = re.compile(
    r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.eE+-]+), "
    r"It/sec ([\d.]+), Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB"
)
VAL_LOSS_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
BOOTSTRAP_RE = re.compile(r"-> Generated (\d+)/(\d+) unique tasks")
TRAJECTORY_RE = re.compile(
    r"Generating agentic trajectory \(Attempt (\d+)/(\d+)\) using teacher .+ for task: '(.{0,80})"
)
RESUME_CKPT_RE = re.compile(r"Resumed from checkpoint: (\d+) samples already completed")
GENERATE_TARGET_RE = re.compile(r"Generating (\d+) agentic trajectories")
DONE_TRAJ_RE = re.compile(r"Done\. (\d+) trajectories saved")
GEN_STATS_RE = re.compile(
    r"\[gen-stats\] label=(\S+) tokens=(\d+) elapsed_s=([\d.]+) tok_s=([\d.]+)"
)


def reset_session_stats() -> dict:
    SESSION_STATS.clear()
    SESSION_STATS.update(
        {
            "train_tokens_total": 0,
            "gen_tokens_total": 0,
            "last_train_tokens_by_job": {},
            "train_tokens_per_sec": None,
            "gen_tokens_per_sec": None,
        }
    )
    return session_stats_payload()


def session_stats_payload() -> dict:
    train_live = None
    gen_live = None
    for job in JOBS.values():
        if job.get("status") not in {"starting", "running", "stopping"}:
            continue
        job_stats = job.get("stats") or {}
        if job_stats.get("kind") == "train" and job_stats.get("tokens_per_sec") is not None:
            train_live = job_stats["tokens_per_sec"]
        if job_stats.get("kind") == "generate" and job_stats.get("tokens_per_sec") is not None:
            gen_live = job_stats["tokens_per_sec"]
    return {
        "train_tokens_total": int(SESSION_STATS.get("train_tokens_total", 0)),
        "gen_tokens_total": int(SESSION_STATS.get("gen_tokens_total", 0)),
        "train_tokens_per_sec": train_live if train_live is not None else SESSION_STATS.get("train_tokens_per_sec"),
        "gen_tokens_per_sec": gen_live if gen_live is not None else SESSION_STATS.get("gen_tokens_per_sec"),
    }


def accumulate_session_stats(job_id: str, stats: dict, line: str) -> None:
    if match := GEN_STATS_RE.search(line):
        tokens = int(match.group(2))
        tok_s = float(match.group(4))
        SESSION_STATS["gen_tokens_total"] = int(SESSION_STATS.get("gen_tokens_total", 0)) + tokens
        SESSION_STATS["gen_tokens_per_sec"] = tok_s
        stats.setdefault("kind", "generate")
        stats["tokens_per_sec"] = tok_s
        stats["gen_tokens_last"] = tokens
        stats["gen_label"] = match.group(1)
        stats["gen_elapsed_s"] = float(match.group(3))
        stats["updated_at"] = time.time()
        return

    if match := TRAIN_STATS_RE.search(line):
        trained_tokens = int(match.group(6))
        last_by_job = SESSION_STATS.setdefault("last_train_tokens_by_job", {})
        previous = int(last_by_job.get(job_id, 0))
        delta = max(0, trained_tokens - previous)
        if delta:
            SESSION_STATS["train_tokens_total"] = int(SESSION_STATS.get("train_tokens_total", 0)) + delta
        last_by_job[job_id] = trained_tokens
        SESSION_STATS["train_tokens_per_sec"] = float(match.group(5))


def output_path_from_command(command: list[str]) -> str | None:
    for index, part in enumerate(command):
        if part == "--output" and index + 1 < len(command):
            return command[index + 1]
    return None


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def parse_log_line_stats(job_id: str, stats: dict, line: str) -> None:
    accumulate_session_stats(job_id, stats, line)

    if match := TRAIN_STATS_RE.search(line):
        stats.update(
            {
                "kind": "train",
                "iter": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "learning_rate": match.group(3),
                "it_per_sec": float(match.group(4)),
                "tokens_per_sec": float(match.group(5)),
                "trained_tokens": int(match.group(6)),
                "peak_mem_gb": float(match.group(7)),
                "updated_at": time.time(),
            }
        )
        return

    if match := VAL_LOSS_RE.search(line):
        stats.setdefault("kind", "train")
        stats["val_iter"] = int(match.group(1))
        stats["val_loss"] = float(match.group(2))
        stats["updated_at"] = time.time()
        return

    if match := GENERATE_TARGET_RE.search(line):
        stats["kind"] = "generate"
        stats["target_trajectories"] = int(match.group(1))
        return

    if match := BOOTSTRAP_RE.search(line):
        stats["kind"] = "generate"
        stats["bootstrap_done"] = int(match.group(1))
        stats["bootstrap_total"] = int(match.group(2))
        return

    if match := RESUME_CKPT_RE.search(line):
        stats["kind"] = "generate"
        stats["saved_trajectories"] = int(match.group(1))
        return

    if match := TRAJECTORY_RE.search(line):
        stats["kind"] = "generate"
        stats["attempt"] = int(match.group(1))
        stats["max_attempts"] = int(match.group(2))
        stats["current_task"] = match.group(3)
        stats["updated_at"] = time.time()
        return

    if match := DONE_TRAJ_RE.search(line):
        stats["kind"] = "generate"
        stats["saved_trajectories"] = int(match.group(1))
        return

    if "--> Discarded trajectory" in line:
        stats["discards"] = int(stats.get("discards", 0)) + 1
    elif "Failed to parse teacher response" in line:
        stats["parse_failures"] = int(stats.get("parse_failures", 0)) + 1


def enrich_job_stats(job: dict) -> dict:
    stats = dict(job.get("stats") or {})
    if job.get("status") in {"starting", "running", "stopping"}:
        output_value = output_path_from_command(job.get("command") or [])
        if output_value:
            rows = count_jsonl_rows(resolve_repo_path(output_value))
            if rows:
                stats.setdefault("kind", "generate")
                stats["saved_trajectories"] = rows
                if stats.get("target_trajectories"):
                    stats["progress_pct"] = round(
                        100 * rows / max(stats["target_trajectories"], 1),
                        1,
                    )
    return stats


def job_for_api(job: dict) -> dict:
    payload = dict(job)
    payload["stats"] = enrich_job_stats(job)
    return payload


def resolve_browse_path(value: str | None) -> Path:
    if not value:
        return ROOT
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def browse_roots() -> list[dict]:
    roots = [
        ("Workspace", ROOT),
        ("Home", Path.home()),
        ("LM Studio Models", Path.home() / ".lmstudio" / "models"),
        ("Data", ROOT / "data"),
        ("Generated", ROOT / "data" / "generated"),
        ("Models", ROOT / "models"),
    ]
    return [
        {"label": label, "path": str(path)}
        for label, path in roots
        if path.exists()
    ]


def browse_path(value: str | None) -> dict:
    path = resolve_browse_path(value)
    if path.is_file():
        path = path.parent
    if not path.exists():
        existing = path
        while not existing.exists() and existing.parent != existing:
            existing = existing.parent
        path = existing if existing.exists() else ROOT

    entries = []
    try:
        children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        for child in children[:500]:
            try:
                stat = child.stat()
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                    "size": stat.st_size,
                })
            except OSError:
                continue
    except OSError as exc:
        return {
            "current": str(path),
            "parent": str(path.parent) if path.parent != path else "",
            "roots": browse_roots(),
            "entries": [],
            "error": str(exc),
        }

    return {
        "current": str(path),
        "parent": str(path.parent) if path.parent != path else "",
        "roots": browse_roots(),
        "entries": entries,
    }


def read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def send_json(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def load_presets() -> dict:
    if not PRESETS_PATH.exists():
        return {"presets": {}}
    with PRESETS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_presets(store: dict) -> None:
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRESETS_PATH.open("w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
        f.write("\n")


def preset_slug(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip().lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "preset"


def sanitize_preset_name(name: str) -> str:
    slug = preset_slug(name)
    if not slug:
        raise ValueError("Preset name is required.")
    return slug


def compact_name(path: str) -> str:
    return Path(path).name if path else ""


def latest_adapter() -> str:
    base = ROOT / "models" / "mlx_self_training"
    if not base.exists():
        return ""
    candidates = sorted(base.glob("iteration_*"), key=lambda p: p.name, reverse=True)
    for path in candidates:
        if (path / "adapters.safetensors").exists():
            return str(path.relative_to(ROOT))
    return ""


GEN_OUTPUT_DIR = "data/generated"


def normalize_generate_output(path_value: str | None) -> str:
    """Bare run names become data/generated/<name>.jsonl; full paths get .jsonl if missing."""
    v = (path_value or "").strip().replace("\\", "/")
    if not v:
        return f"{GEN_OUTPUT_DIR}/dingo_run.jsonl"
    bare = "/" not in v and not v.startswith("~")
    if bare:
        if not v.lower().endswith(".jsonl"):
            v = f"{v}.jsonl"
        return f"{GEN_OUTPUT_DIR}/{v}"
    if not v.lower().endswith(".jsonl"):
        base = v.rsplit("/", 1)[-1]
        if "." not in base:
            v = f"{v}.jsonl"
    return v


def path_exists_for_payload(path_value: str | None) -> bool:
    if not path_value:
        return False
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def preflight_job(payload: dict) -> None:
    if payload.get("overwrite"):
        return
    job_type = payload.get("type")
    if job_type == "generate":
        output = normalize_generate_output(payload.get("output"))
        if path_exists_for_payload(output):
            raise ValueError(
                f"Generate output already exists: {output}. Choose a new name or enable overwrite."
            )
    if job_type == "train" and path_exists_for_payload(payload.get("train_output")):
        raise ValueError(f"Train output directory already exists: {payload.get('train_output')}. Choose a new path or enable overwrite.")


def build_command(payload: dict) -> list[str]:
    job_type = payload.get("type")
    config_arg = ["--config", str(CONFIG_PATH.relative_to(ROOT))]

    wire_format = payload.get("wire_format")

    if job_type == "generate":
        output = normalize_generate_output(payload.get("output"))
        cmd = [
            str(PYTHON), "-u", str(MAIN.relative_to(ROOT)),
            "--mode", "generate-only",
            "--samples", str(payload.get("samples", 20)),
            "--output", output,
            *config_arg,
        ]
        if wire_format and wire_format != "dingo":
            cmd.extend(["--wire-format", wire_format])
        if payload.get("overwrite"):
            cmd.append("--overwrite")
        return cmd

    if job_type == "train":
        cmd = [
            str(PYTHON), "-u", str(MAIN.relative_to(ROOT)),
            "--mode", "train-only",
            "--data", payload["data"],
            "--train-iters", str(payload.get("train_iters", 200)),
            "--train-output", payload.get("train_output", "models/mlx_self_training/gui_train"),
            *config_arg,
        ]
        if wire_format and wire_format != "dingo":
            cmd.extend(["--wire-format", wire_format])
        if payload.get("resume"):
            cmd.extend(["--resume", payload["resume"]])
        if payload.get("overwrite"):
            cmd.append("--overwrite")
        return cmd

    if job_type == "build_omlx_pack":
        script = ROOT / "tools" / "build_omlx_training_pack.py"
        return [str(PYTHON), "-u", str(script)]

    if job_type == "resume":
        return ["./run_resume.sh", *config_arg]

    if job_type == "full":
        return ["./run_full.sh", *config_arg]

    if job_type == "smoke":
        return ["./run_smoke.sh", *config_arg]

    if job_type == "report":
        files = payload.get("files") or []
        if not files:
            raise ValueError("report job requires at least one file")
        return ["./run_report.sh", *files]

    raise ValueError(f"Unknown job type: {job_type}")


def stream_process(job_id: str, command: list[str]) -> None:
    job = JOBS[job_id]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        job["pid"] = process.pid
        job["status"] = "running"
        assert process.stdout is not None
        stats = job.setdefault("stats", {})
        for line in process.stdout:
            clean = line.rstrip("\n")
            job["log"].append(clean)
            job["log"] = job["log"][-2000:]
            parse_log_line_stats(job_id, stats, clean)
        code = process.wait()
        job["exit_code"] = code
        job["status"] = "completed" if code == 0 else "failed"
    except Exception as exc:
        job["status"] = "failed"
        job["exit_code"] = -1
        job["log"].append(f"[server error] {exc}")
    finally:
        job["ended_at"] = time.time()


def start_job(payload: dict) -> dict:
    preflight_job(payload)
    command = build_command(payload)
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "type": payload.get("type"),
        "status": "starting",
        "command": command,
        "log": [],
        "stats": {},
        "started_at": time.time(),
        "ended_at": None,
        "exit_code": None,
        "pid": None,
    }
    JOBS[job_id] = job
    thread = threading.Thread(target=stream_process, args=(job_id, command), daemon=True)
    thread.start()
    return job


def stop_job(job_id: str) -> bool:
    job = JOBS.get(job_id)
    if not job or not job.get("pid") or job.get("status") != "running":
        return False
    try:
        os.kill(job["pid"], signal.SIGTERM)
        job["status"] = "stopping"
        job["log"].append("[server] Stop requested.")
        return True
    except OSError:
        return False


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            config = load_config()
            send_json(self, {
                "config": config,
                "teacher_names": [compact_name(p) for p in config["models"].get("teachers", [])],
                "latest_adapter": latest_adapter(),
            })
            return
        if parsed.path == "/api/jobs":
            send_json(
                self,
                {
                    "jobs": [job_for_api(job) for job in JOBS.values()],
                    "session_stats": session_stats_payload(),
                },
            )
            return
        if parsed.path == "/api/session-stats":
            send_json(self, session_stats_payload())
            return
        if parsed.path == "/api/browse":
            query = parse_qs(parsed.query)
            send_json(self, browse_path((query.get("path") or [""])[0]))
            return
        if parsed.path == "/api/presets":
            send_json(self, load_presets())
            return
        if parsed.path == "/api/file-stats":
            query = parse_qs(parsed.query)
            rel = (query.get("path") or [""])[0]
            path = Path(rel).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                send_json(self, {"error": "file not found"}, 404)
                return
            lines = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
            send_json(self, {"path": str(path.relative_to(ROOT)), "lines": lines})
            return
        if parsed.path.startswith("/api/jobs/"):
            segments = [part for part in parsed.path.split("/") if part]
            if len(segments) >= 3 and segments[0] == "api" and segments[1] == "jobs":
                job_id = segments[2]
                job = JOBS.get(job_id)
                if len(segments) == 4 and segments[3] == "log":
                    if not job:
                        send_json(self, {"error": "not found"}, 404)
                        return
                    body = "\n".join(job.get("log") or []) + "\n"
                    raw = body.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="dingoai-job-{job_id}.log"',
                    )
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                payload = {"job": job_for_api(job)} if job else {"error": "not found"}
                send_json(self, payload, 200 if job else 404)
                return
        return super().do_GET()

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/presets/"):
                name = sanitize_preset_name(unquote(parsed.path.rsplit("/", 1)[-1]))
                store = load_presets()
                presets = store.setdefault("presets", {})
                if name not in presets:
                    send_json(self, {"error": "preset not found"}, 404)
                    return
                del presets[name]
                save_presets(store)
                send_json(self, {"ok": True, "deleted": name})
                return
            send_json(self, {"error": "not found"}, 404)
        except Exception as exc:
            send_json(self, {"error": str(exc)}, 400)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/config":
                payload = read_json_body(self)
                save_config(payload["config"])
                send_json(self, {"ok": True, "config": payload["config"]})
                return
            if parsed.path == "/api/session-stats/reset":
                send_json(self, {"ok": True, "session_stats": reset_session_stats()})
                return
            if parsed.path == "/api/jobs":
                payload = read_json_body(self)
                job = start_job(payload)
                send_json(
                    self,
                    {"job": job_for_api(job), "session_stats": session_stats_payload()},
                )
                return
            if parsed.path == "/api/presets":
                payload = read_json_body(self)
                name = sanitize_preset_name(payload.get("name") or "")
                preset_type = payload.get("type")
                if preset_type not in ("models", "prompt", "combined"):
                    raise ValueError("Preset type must be models, prompt, or combined.")
                store = load_presets()
                presets = store.setdefault("presets", {})
                entry = {
                    "type": preset_type,
                    "label": payload.get("label") or name,
                    "description": payload.get("description") or "",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if preset_type in ("models", "combined"):
                    entry["models"] = payload.get("models")
                    entry["generation"] = payload.get("generation")
                    entry["hardware"] = payload.get("hardware")
                if preset_type in ("prompt", "combined"):
                    entry["task_system_prompt"] = payload.get("task_system_prompt") or ""
                presets[name] = entry
                save_presets(store)
                send_json(self, {"ok": True, "name": name, "preset": entry})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/stop"):
                job_id = parsed.path.split("/")[-2]
                send_json(self, {"ok": stop_job(job_id)})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/clear-log"):
                job_id = parsed.path.split("/")[-2]
                job = JOBS.get(job_id)
                if not job:
                    send_json(self, {"error": "not found"}, 404)
                    return
                job["log"] = []
                job["stats"] = {}
                send_json(self, {"ok": True, "job": job_for_api(job)})
                return
            send_json(self, {"error": "not found"}, 404)
        except Exception as exc:
            send_json(self, {"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local MLX self-training web GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DingoAI console running at http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
