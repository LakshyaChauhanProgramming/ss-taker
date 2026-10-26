#!/usr/bin/env sh
set -eu
.venv/bin/uvicorn ss_taker.app:app --host "${SS_TAKER_HOST:-127.0.0.1}" --port "${PORT:-8000}"
