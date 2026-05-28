#!/usr/bin/env python3
"""Capture DingoAI web console screenshots for the README."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
HOST = "127.0.0.1"
PORT = 8777
BASE = f"http://{HOST}:{PORT}"


def wait_for_server(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    urls = (f"{BASE}/", f"{BASE}/api/config", f"{BASE}/api/presets")
    while time.time() < deadline:
        if all(_ok(url) for url in urls):
            return
        time.sleep(0.25)
    raise RuntimeError(f"Server did not become ready at {BASE}")


def _ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install playwright: pip install playwright && playwright install chromium", file=sys.stderr)
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    python = ROOT / "mlx_foundation" / "venv" / "bin" / "python"
    server_py = ROOT / "web" / "server.py"
    proc = subprocess.Popen(
        [str(python), str(server_py), "--host", HOST, "--port", str(PORT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 1400})
            page.goto(f"{BASE}/", wait_until="load", timeout=30000)

            # Wait until React-like init finishes (config + presets loaded)
            page.wait_for_selector("h1", timeout=15000)
            page.wait_for_function(
                """() => {
                    const h1 = document.querySelector('h1');
                    if (!h1 || h1.textContent.trim() !== 'DingoAI') return false;
                    const err = document.body.innerText || '';
                    if (err.includes('Request failed') || err.includes('404')) return false;
                    return !!document.querySelector('#save-config');
                }""",
                timeout=15000,
            )
            page.wait_for_timeout(500)

            page.screenshot(path=str(OUT / "dingoai-console-overview.png"), full_page=False)

            browser.close()

        text = (OUT / "dingoai-console-overview.png").read_bytes()
        if len(text) < 5000:
            raise RuntimeError("Screenshot file looks too small — capture may have failed")
        print(f"Wrote {OUT / 'dingoai-console-overview.png'} ({len(text) // 1024} KB)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
