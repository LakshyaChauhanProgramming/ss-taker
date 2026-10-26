from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = Path("output")


def request_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, method="GET" if data is None else "POST")
    req.add_header("Content-Type", "application/json")
    if os.environ.get("SS_TAKER_TOKEN"):
        req.add_header("X-API-Key", os.environ["SS_TAKER_TOKEN"])
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def image_stats(image: Image.Image) -> dict[str, Any]:
    gray = image.convert("L")
    arr = np.asarray(gray)
    return {
        "width": image.width,
        "height": image.height,
        "mean": round(float(arr.mean()), 3),
        "stddev": round(float(arr.std()), 3),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "probably_blank": float(arr.std()) < 2.0 or int(arr.max()) - int(arr.min()) < 8,
    }


def decode_preview(result: dict[str, Any]) -> Path | None:
    encoded = result.get("full_ss_base64")
    if not isinstance(encoded, str) or not encoded:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "debug-preview.png"
    path.write_bytes(base64.b64decode(encoded))
    return path


def main() -> int:
    print("== health ==")
    print(json.dumps(request_json("/health"), indent=2))
    print(json.dumps(request_json("/debug"), indent=2))
    print(json.dumps(request_json("/health/backend"), indent=2))
    print(json.dumps(request_json("/health/ocr"), indent=2))

    print("== pick-ss ==")
    result = request_json("/pick-ss", {"lang": "eng", "psm": 11, "preserve_layout": False, "mode": "balanced"})
    preview_path = decode_preview(result)
    print(json.dumps({
        "text_length": len(str(result.get("chosen_text", ""))),
        "json_path": result.get("json_path"),
        "image_hash": result.get("image_hash"),
        "capture_stats": result.get("capture_stats"),
        "candidate_summary": [
            {
                "variant": candidate.get("variant"),
                "text_length": candidate.get("text_length", len(str(candidate.get("cleaned_text", "")))),
                "mean_confidence": candidate.get("mean_confidence"),
                "score": candidate.get("score"),
                "width": candidate.get("width"),
                "height": candidate.get("height"),
            }
            for candidate in result.get("candidate_texts", [])
        ],
        "preview_path": str(preview_path) if preview_path else None,
    }, indent=2))

    if preview_path:
        with Image.open(preview_path) as image:
            print("== decoded preview stats ==")
            print(json.dumps(image_stats(image), indent=2))

    capture_stats = result.get("capture_stats") if isinstance(result.get("capture_stats"), dict) else {}
    all_empty = all(int(candidate.get("text_length", len(str(candidate.get("cleaned_text", ""))))) == 0 for candidate in result.get("candidate_texts", []))
    if capture_stats.get("probably_blank"):
        print("RCA: screenshot capture is blank or nearly uniform; OCR has no readable pixels.")
        return 2
    if all_empty:
        print("RCA: screenshot has pixel variation, but Tesseract found no words; inspect output/debug-preview.png and try a smaller region.")
        return 3
    print("RCA: capture and OCR produced text.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"debug_capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
