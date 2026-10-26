from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST")
    req.add_header("Content-Type", "application/json")
    if os.environ.get("SS_TAKER_TOKEN"):
        req.add_header("X-API-Key", os.environ["SS_TAKER_TOKEN"])
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_health(timeout_s: int = 30) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = request_json(f"{BASE_URL}/health")
            if health.get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover - smoke script
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def main() -> int:
    server = subprocess.Popen(
        ["ss-taker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_health()
        result = request_json(
            f"{BASE_URL}/pick-ss",
            {"lang": "eng", "psm": 6, "preserve_layout": False, "mode": "balanced"},
        )
        print(json.dumps(result, indent=2))
        text = result.get("text", "")
        if not isinstance(text, str):
            raise RuntimeError("OCR response did not contain text")
        print(f"text_length={len(text)}")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
