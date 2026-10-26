# ss-taker

Captures the screen, runs Tesseract OCR on it and returns the text over a small FastAPI service.

## Run

```sh
scripts/run_api.sh          # creates .venv, installs, serves on 127.0.0.1:8000
```

```sh
curl -X POST http://127.0.0.1:8000/pick-ss \
  -H "X-API-Key: $SS_TAKER_TOKEN" -H "Content-Type: application/json" \
  -d '{"mode": "fast"}'
```

## Faster OCR with tesserocr (recommended)

```sh
.venv/bin/pip install -e ".[fast]"
```

With `tesserocr` installed, Tesseract runs inside the server process and each language model loads once per
OCR thread, instead of starting a `tesseract` process (and reloading the model) for every call. Without it the
service falls back to `pytesseract` automatically. `GET /health/ocr` reports which `engine` is active.

Measured on full-HD screenshots, 4-core laptop (median):

| mode | pytesseract | tesserocr |
|---|---|---|
| `fast` | 1149 ms | 500 ms |
| `balanced` | 2117 ms | 953 ms |
| `quality_hd_ultra` | 64466 ms | 4963 ms |
| `fast`, 900x300 region | 344 ms | 122 ms |

The tesserocr wheel bundles its own Tesseract (5.5.x), so its text can differ slightly from the system
`tesseract` CLI on noisy areas such as browser tabs and icons. It uses the system language data
(`/usr/share/tesseract-ocr/*/tessdata`, or `TESSDATA_PREFIX`). Memory grows by roughly 50 MB per OCR thread.

## OCR modes

| mode | what it does |
|---|---|
| `fast` | OCR on the raw capture only (~1s on a full-HD screen) |
| `balanced` (default) | raw first; adds a preprocessed pass only if the raw read is weak |
| `quality_hd_ultra` | raw, then 8 upscaled/thresholded variants in parallel; slowest |

Use `region: [x, y, width, height]` to OCR only part of the screen; it is faster and usually cleaner.

## Configuration

| variable | default | purpose |
|---|---|---|
| `SS_TAKER_TOKEN` | unset | when set, every endpoint except `/health` needs `X-API-Key: <token>` or `Authorization: Bearer <token>` |
| `SS_TAKER_HOST` | `127.0.0.1` | bind address; set `0.0.0.0` only on a trusted network |
| `PORT` | `8000` | listen port |
| `SS_TAKER_SAVE_SCREENSHOTS` | off | also store a preview PNG (base64) in `output/*.json` |
| `SS_TAKER_OCR_TIMEOUT` | `30` | seconds per Tesseract call before giving up |
| `SS_TAKER_OCR_ENGINE` | `auto` | `auto` uses tesserocr when installed; `pytesseract` forces the CLI |
| `SS_TAKER_EARLY_EXIT_CONFIDENCE` | `70` | in `balanced`, skip extra passes when raw confidence is at least this |

## Exposing it publicly

`scripts/run_public.sh` opens an ngrok tunnel and refuses to start without `SS_TAKER_TOKEN`:

```sh
export SS_TAKER_TOKEN=$(openssl rand -hex 32)
scripts/run_public.sh
```

Anyone with the URL and token can read whatever is on your screen, so share both carefully and stop the tunnel when you are done.

## Tests

```sh
.venv/bin/python -m pytest -q
```
