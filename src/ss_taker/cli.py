from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("SS_TAKER_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("ss_taker.app:app", host=host, port=port, reload=False)

