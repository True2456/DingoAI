#!/usr/bin/env python3
"""Capture DingoAI web console screenshots for the README."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
URL = "http://127.0.0.1:8766/"
PORT = 8766


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
        [str(python), str(server_py), "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.5)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 2000})
            page.goto(URL, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(800)

            page.screenshot(path=str(OUT / "dingoai-console-overview.png"), full_page=True)

            # Jobs section (scroll down)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(400)
            jobs = page.locator("#jobs")
            if jobs.count():
                jobs.screenshot(path=str(OUT / "dingoai-jobs-panel.png"))

            browser.close()
        print(f"Wrote screenshots to {OUT}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
