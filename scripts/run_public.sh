#!/usr/bin/env sh
# Start ss-taker and an ngrok tunnel, then print the public pick-ss endpoint.
# All service output is written to output/*.log; stdout contains only the URL.
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PORT=${PORT:-8000}
NGROK_API_PORT=${NGROK_API_PORT:-4040}
OUTPUT_DIR="$ROOT_DIR/output"
SERVER_LOG="$OUTPUT_DIR/server.log"
NGROK_LOG="$OUTPUT_DIR/ngrok.log"

fail() {
    printf '%s\n' "$1" >&2
    exit 1
}

stop_listeners() {
    pids=$(lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true)
    [ -z "$pids" ] || kill $pids 2>/dev/null || true

    attempts=0
    while [ "$attempts" -lt 20 ]; do
        lsof -tiTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 || return 0
        sleep 0.25
        attempts=$((attempts + 1))
    done
    fail "Port $1 is still in use"
}

wait_for_server() {
    attempts=0
    while [ "$attempts" -lt 60 ]; do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
        attempts=$((attempts + 1))
    done
    fail "Server did not start; see $SERVER_LOG"
}

wait_for_tunnel_url() {
    attempts=0
    while [ "$attempts" -lt 60 ]; do
        tunnels=$(curl -fsS "http://127.0.0.1:$NGROK_API_PORT/api/tunnels" 2>/dev/null || true)
        if [ -n "$tunnels" ]; then
            url=$(printf '%s' "$tunnels" | python3 -c '
import json
import sys

try:
    tunnels = json.load(sys.stdin).get("tunnels", [])
except (json.JSONDecodeError, OSError):
    tunnels = []
public_urls = [t.get("public_url", "") for t in tunnels]
print(next((url for url in public_urls if url.startswith("https://")), ""), end="")
')
            if [ -n "$url" ]; then
                printf '%s/pick-ss\n' "$url"
                return 0
            fi
        fi
        sleep 0.5
        attempts=$((attempts + 1))
    done
    fail "ngrok did not create a tunnel; see $NGROK_LOG"
}

# ngrok forwards public traffic to localhost, so binding to 127.0.0.1 does not
# protect the API here; a token is the only thing keeping strangers out.
[ -n "${SS_TAKER_TOKEN:-}" ] || fail "Set SS_TAKER_TOKEN before exposing ss-taker publicly (e.g. export SS_TAKER_TOKEN=\$(openssl rand -hex 32))"
export SS_TAKER_TOKEN
command -v ngrok >/dev/null 2>&1 || fail "ngrok is not installed"
command -v lsof >/dev/null 2>&1 || fail "lsof is not installed"

mkdir -p "$OUTPUT_DIR"
stop_listeners "$PORT"
stop_listeners "$NGROK_API_PORT"

(
    cd "$ROOT_DIR"
    exec scripts/run_host.sh
) >"$SERVER_LOG" 2>&1 &

wait_for_server

(
    cd "$ROOT_DIR"
    exec ngrok http "$PORT" --log=stdout
) >"$NGROK_LOG" 2>&1 &

wait_for_tunnel_url
