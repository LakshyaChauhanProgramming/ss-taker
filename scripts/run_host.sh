#!/usr/bin/env sh
set -eu
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn ss_taker.app:app --host "${SS_TAKER_HOST:-127.0.0.1}" --port "${PORT:-8000}"
