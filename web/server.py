#!/usr/bin/env python3
"""Local web GUI server for the MLX self-training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
CONFIG_PATH = ROOT / "config" / "default_config.json"
PYTHON = ROOT / "mlx_foundation" / "venv" / "bin" / "python"
MAIN = ROOT / "mlx_foundation" / "src" / "main.py"

JOBS: dict[str, dict] = {}


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
    if job_type == "generate" and path_exists_for_payload(payload.get("output")):
        raise ValueError(f"Generate output already exists: {payload.get('output')}. Choose a new path or enable overwrite.")
    if job_type == "train" and path_exists_for_payload(payload.get("train_output")):
        raise ValueError(f"Train output directory already exists: {payload.get('train_output')}. Choose a new path or enable overwrite.")


def build_command(payload: dict) -> list[str]:
    job_type = payload.get("type")
    config_arg = ["--config", str(CONFIG_PATH.relative_to(ROOT))]

    if job_type == "generate":
        cmd = [
            str(PYTHON), "-u", str(MAIN.relative_to(ROOT)),
            "--mode", "generate-only",
            "--samples", str(payload.get("samples", 20)),
            "--output", payload.get("output", "data/generated/gui_generation.jsonl"),
            *config_arg,
        ]
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
        if payload.get("resume"):
            cmd.extend(["--resume", payload["resume"]])
        if payload.get("overwrite"):
            cmd.append("--overwrite")
        return cmd

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
        for line in process.stdout:
            job["log"].append(line.rstrip("\n"))
            job["log"] = job["log"][-2000:]
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
            send_json(self, {"jobs": list(JOBS.values())})
            return
        if parsed.path == "/api/browse":
            query = parse_qs(parsed.query)
            send_json(self, browse_path((query.get("path") or [""])[0]))
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            send_json(self, {"job": job} if job else {"error": "not found"}, 200 if job else 404)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/config":
                payload = read_json_body(self)
                save_config(payload["config"])
                send_json(self, {"ok": True, "config": payload["config"]})
                return
            if parsed.path == "/api/jobs":
                payload = read_json_body(self)
                job = start_job(payload)
                send_json(self, {"job": job})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/stop"):
                job_id = parsed.path.split("/")[-2]
                send_json(self, {"ok": stop_job(job_id)})
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
    print(f"Web GUI running at http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
