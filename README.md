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

## Meeting summaries

A meeting recording combines two sources into one timestamped transcript:

- **screen**: OCR of a screen region (point `region` at the captions or chat pane) every few seconds. Lines that
  were already seen are dropped, and a live caption that grows between captures replaces its earlier version.
- **audio**: speech recognition with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on audio
  recorded with `parec`. `others` is what the computer plays (the remote participants), `me` is the microphone.
  Audio is processed in memory and never written to disk.

The summary (overview, key topics, decisions) comes from [OpenRouter](https://openrouter.ai). The default model
`openrouter/free` picks any free model per request; the response reports which one was used.

```sh
.venv/bin/pip install -e ".[audio]"     # plus: sudo apt install pulseaudio-utils (for parec)
export OPENROUTER_API_KEY=sk-or-...
```

```sh
# start; returns the meeting id
curl -X POST http://127.0.0.1:8000/meetings -H "Content-Type: application/json" \
  -d '{"title": "Weekly sync", "region": [0, 700, 1920, 300], "audio_sources": ["others", "me"]}'

curl -X POST http://127.0.0.1:8000/meetings/<id>/stop
curl http://127.0.0.1:8000/meetings/<id>/transcript
curl -X POST http://127.0.0.1:8000/meetings/<id>/summarize -H "Content-Type: application/json" \
  -d '{"language": "English"}'
```

| endpoint | purpose |
|---|---|
| `POST /meetings` | start recording (only one meeting records at a time; `409` otherwise) |
| `POST /meetings/{id}/stop` | stop; waits for the last audio chunk to be transcribed |
| `GET /meetings`, `GET /meetings/{id}` | status, segment count, errors, last summary |
| `GET /meetings/{id}/transcript` | plain-text transcript, one `[mm:ss] (source) text` line per segment |
| `POST /meetings/{id}/summarize` | summarize; also works while recording for a summary so far |
| `GET /health/meeting` | whether audio capture and the LLM key are available |

Each meeting is saved to `output/meetings/<id>.json` when it stops and after each summary.

| variable | default | purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | unset | required for `/summarize` |
| `SS_TAKER_LLM_MODEL` | `openrouter/free` | any OpenRouter model id |
| `SS_TAKER_LLM_URL` | OpenRouter chat completions | any OpenAI-compatible endpoint |
| `SS_TAKER_LLM_TIMEOUT` | `120` | seconds per LLM call |
| `SS_TAKER_LLM_CHUNK_CHARS` | `24000` | longer transcripts are summarized in parts, then merged |
| `SS_TAKER_WHISPER_MODEL` | `base` | `small` is noticeably better for Hindi/Hinglish but ~3x slower on CPU |
| `SS_TAKER_AUDIO_CHUNK_SECONDS` | `30` | audio is transcribed in chunks of this length |
| `SS_TAKER_AUDIO_SILENCE_RMS` | `0.003` | quieter chunks are skipped (Whisper invents text on silence) |

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
