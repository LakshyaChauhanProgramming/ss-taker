from __future__ import annotations

import hashlib
import base64
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import select
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

try:
    import tesserocr
except ImportError:  # optional: pip install -e ".[fast]"
    tesserocr = None

from .cache import CacheEntry, ResultCache
from .preprocess import preprocess_image

LOGGER = logging.getLogger("ss_taker")

BACKEND_CANDIDATES = ("spectacle", "portal", "scrot", "gnome-screenshot", "import", "grim")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# A full-HD screen can take well over 8s to OCR while a meeting app is busy
# using the CPU; a timeout that short silently returned empty text.
OCR_TIMEOUT_SECONDS = _env_float("SS_TAKER_OCR_TIMEOUT", 30.0)
# If the untouched screenshot already reads this well, skip the other variants.
EARLY_EXIT_CONFIDENCE = _env_float("SS_TAKER_EARLY_EXIT_CONFIDENCE", 70.0)
EARLY_EXIT_MIN_WORDS = 5
OCR_MODES = ("fast", "balanced", "quality_hd_ultra")


@dataclass(slots=True)
class OCRResult:
    mode: str
    text: str
    processed_text: str
    raw_text: str
    candidate_texts: list[dict[str, object]]
    capture_stats: dict[str, object]
    cached: bool
    image_hash: str
    timings_ms: dict[str, float]
    output_path: Path
    screenshot_b64: str


@dataclass(slots=True)
class CaptureDiagnostics:
    display: str | None
    wayland_display: str | None
    xdg_runtime_dir: str | None
    x11_socket: bool
    x11_connected: bool
    backend: str | None
    tesseract: bool
    languages: list[str]
    ocr_ready: bool
    ocr_error: str | None
    startup_error: str | None

    @property
    def ready(self) -> bool:
        return self.startup_error is None and self.ocr_ready and self.x11_connected and self.backend is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "display": self.display,
            "wayland_display": self.wayland_display,
            "xdg_runtime_dir": self.xdg_runtime_dir,
            "x11_socket": self.x11_socket,
            "x11_connected": self.x11_connected,
            "backend": self.backend,
            "tesseract": self.tesseract,
            "languages": self.languages,
            "ocr_ready": self.ocr_ready,
            "ocr_error": self.ocr_error,
            "startup_error": self.startup_error,
        }


class TesserocrEngine:
    """Runs Tesseract in-process via tesserocr.

    pytesseract starts a new tesseract process per call, which writes the image
    to a temp file and reloads the language model every time. Here each model is
    loaded once and reused. A PyTessBaseAPI is not thread-safe, so every thread
    keeps its own instance per language.
    """

    def __init__(self, tessdata: str) -> None:
        self.tessdata = tessdata
        self._local = threading.local()

    def _api(self, lang: str):
        apis = self._local.__dict__.setdefault("apis", {})
        api = apis.get(lang)
        if api is None:
            api = tesserocr.PyTessBaseAPI(path=self.tessdata, lang=lang, oem=tesserocr.OEM.LSTM_ONLY)
            apis[lang] = api
        return api

    def image_to_data(
        self,
        image: np.ndarray,
        *,
        lang: str,
        psm: int,
        whitelist: Optional[str],
        preserve_layout: bool,
    ) -> dict[str, list[str]] | None:
        """Same shape as pytesseract's image_to_data(Output.DICT); None on timeout."""
        api = self._api(lang)
        # Variables stick to the API object between calls, so set every one explicitly.
        api.SetPageSegMode(psm)
        api.SetVariable("user_defined_dpi", "300")
        api.SetVariable("tessedit_char_whitelist", whitelist or "")
        api.SetVariable("preserve_interword_spaces", "1" if preserve_layout else "0")
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        api.SetImageBytes(image.tobytes(), width, height, channels, width * channels)
        if not api.Recognize(timeout=int(OCR_TIMEOUT_SECONDS * 1000)):
            return None
        return _parse_tsv(api.GetTSVText(0))


def _parse_tsv(tsv: str) -> dict[str, list[str]]:
    columns = ("level", "page_num", "block_num", "par_num", "line_num", "word_num",
               "left", "top", "width", "height", "conf", "text")
    data: dict[str, list[str]] = {name: [] for name in columns}
    for row in tsv.splitlines():
        fields = row.split("\t")
        if len(fields) < len(columns) - 1 or fields[0] == "level":
            continue
        fields += [""] * (len(columns) - len(fields))
        for name, value in zip(columns, fields):
            data[name].append(value)
    return data


def _find_tessdata() -> str | None:
    candidates = [
        os.environ.get("TESSDATA_PREFIX"),
        *sorted(glob.glob("/usr/share/tesseract-ocr/*/tessdata"), reverse=True),
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
    ]
    for candidate in candidates:
        if candidate and glob.glob(os.path.join(candidate, "*.traineddata")):
            return candidate
    return None


_tesserocr_engine: TesserocrEngine | None = None
_tesserocr_checked = False
_tesserocr_lock = threading.Lock()


def get_tesserocr_engine() -> TesserocrEngine | None:
    """The in-process engine, or None to use pytesseract.

    SS_TAKER_OCR_ENGINE=pytesseract forces the subprocess path; the default
    (auto) uses tesserocr whenever it is installed and can find tessdata.
    """
    global _tesserocr_engine, _tesserocr_checked
    with _tesserocr_lock:
        if _tesserocr_checked:
            return _tesserocr_engine
        _tesserocr_checked = True
        choice = os.environ.get("SS_TAKER_OCR_ENGINE", "auto").strip().lower()
        if choice == "pytesseract":
            return None
        if tesserocr is None:
            if choice == "tesserocr":
                LOGGER.warning("SS_TAKER_OCR_ENGINE=tesserocr but tesserocr is not installed; using pytesseract")
            return None
        tessdata = _find_tessdata()
        if tessdata is None:
            LOGGER.warning("tesserocr installed but no tessdata found (set TESSDATA_PREFIX); using pytesseract")
            return None
        _tesserocr_engine = TesserocrEngine(tessdata)
        return _tesserocr_engine


def ocr_engine_name() -> str:
    return "tesserocr" if get_tesserocr_engine() is not None else "pytesseract"


def _ocr_worker(
    image_bytes: bytes,
    shape: tuple[int, int, int],
    lang: str,
    psm: int,
    whitelist: Optional[str],
    preserve_layout: bool,
) -> str:
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    config = f"--oem 1 --dpi 300 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    if preserve_layout:
        config += " -c preserve_interword_spaces=1"
    try:
        return pytesseract.image_to_string(
            Image.fromarray(image), lang=lang, config=config, timeout=OCR_TIMEOUT_SECONDS
        )
    except RuntimeError:
        return ""


def _ocr_worker_scored(
    image_bytes: bytes,
    shape: tuple[int, int, int],
    lang: str,
    psm: int,
    whitelist: Optional[str],
    preserve_layout: bool,
) -> tuple[str, float, list[dict[str, object]]]:
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    data = None
    engine = get_tesserocr_engine()
    if engine is not None:
        try:
            data = engine.image_to_data(image, lang=lang, psm=psm, whitelist=whitelist, preserve_layout=preserve_layout)
        except RuntimeError as exc:
            # Typically a language tesserocr cannot load; the CLI may still have it.
            LOGGER.warning("tesserocr failed (%s); falling back to pytesseract", exc)
        else:
            if data is None:
                return "", -1.0, []
    if data is None:
        config = f"--oem 1 --dpi 300 --psm {psm}"
        if whitelist:
            config += f' -c tessedit_char_whitelist="{whitelist}"'
        if preserve_layout:
            config += " -c preserve_interword_spaces=1"
        try:
            data = pytesseract.image_to_data(
                Image.fromarray(image),
                lang=lang,
                config=config,
                output_type=Output.DICT,
                timeout=OCR_TIMEOUT_SECONDS,
            )
        except RuntimeError:
            return "", -1.0, []
    confidences: list[float] = []
    words: list[dict[str, object]] = []
    for raw_conf, word in zip(data.get("conf", []), data.get("text", []), strict=False):
        if not word or not word.strip():
            continue
        try:
            conf = float(raw_conf)
        except Exception:
            continue
        if conf >= 0:
            confidences.append(conf)
            words.append({"text": word.strip(), "confidence": conf})
    mean_conf = sum(confidences) / len(confidences) if confidences else -1.0
    text = _join_confident_words(data)
    if not text:
        # Re-running OCR with image_to_string doubled the cost; the words are
        # already here, just without the confidence filter.
        text = _join_confident_words(data, min_confidence=0.0)
    return text, mean_conf, words


def _join_confident_words(data: dict[str, list[object]], min_confidence: float = 35.0) -> str:
    lines: dict[tuple[int, int, int], list[str]] = {}
    for idx, raw_word in enumerate(data.get("text", [])):
        word = str(raw_word).strip()
        if not word:
            continue
        try:
            conf = float(data.get("conf", [])[idx])
        except Exception:
            continue
        if conf < min_confidence:
            continue
        key = (
            int(data.get("block_num", [0])[idx]),
            int(data.get("par_num", [0])[idx]),
            int(data.get("line_num", [0])[idx]),
        )
        lines.setdefault(key, []).append(word)
    return "\n".join(" ".join(words) for _, words in sorted(lines.items()) if words)


def _probe_x11_socket(display: str | None) -> bool:
    if not display or not display.startswith(":"):
        return False
    sock_num = display.split(".", 1)[0].lstrip(":")
    return Path(f"/tmp/.X11-unix/X{sock_num}").exists()


def _probe_x11_connection(display: str | None) -> bool:
    if not display:
        return False
    try:
        result = subprocess.run(
            ["xdpyinfo"],
            env={**os.environ, "DISPLAY": display},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _pick_backend() -> str | None:
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("XDG_CURRENT_DESKTOP", "").upper() == "KDE":
        if shutil.which("spectacle"):
            return "spectacle"
    if os.environ.get("DISPLAY"):
        return "mss"
    if shutil.which("gdbus") and os.environ.get("WAYLAND_DISPLAY"):
        return "portal"
    for name in BACKEND_CANDIDATES:
        if name == "portal":
            continue
        if shutil.which(name):
            return name
    return None


def _probe_tesseract() -> tuple[bool, list[str], str | None]:
    try:
        langs = pytesseract.get_languages(config="")
        return True, langs, None
    except Exception as exc:
        return False, [], str(exc)


class OCRService:
    def __init__(self, output_dir: str | Path = "output", *, start_pool: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = ResultCache()
        self.display = os.environ.get("DISPLAY")
        self.wayland_display = os.environ.get("WAYLAND_DISPLAY")
        self.xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        self.x11_socket = _probe_x11_socket(self.display)
        self.x11_connected = _probe_x11_connection(self.display)
        self.backend = _pick_backend()
        self.tesseract, self.languages, self.tesseract_error = _probe_tesseract()
        self.startup_error = self._startup_error()
        self.stats = {
            "screenshots": 0,
            "ocr_requests": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
        }
        # Threads, not processes: a process pool made small OCR requests stall
        # during worker startup on some desktops. The pool is kept for the
        # service's lifetime so each thread's tesserocr models load only once.
        self.pool: ThreadPoolExecutor | None = None
        # Screenshots of meetings can hold sensitive content, so only the OCR
        # text is written to disk unless the image is explicitly requested.
        self.save_screenshots = _env_flag("SS_TAKER_SAVE_SCREENSHOTS")

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 1, thread_name_prefix="ss-taker-ocr")
        return self.pool

    def _startup_error(self) -> str | None:
        if self.display:
            if not self.x11_socket:
                return "X11 socket not mounted"
            if not self.x11_connected:
                return "Cannot connect to display"
        elif self.wayland_display:
            if not self.xdg_runtime_dir:
                return "XDG_RUNTIME_DIR is not set for Wayland"
            if not self.backend:
                return "Screenshot backend unavailable"
        else:
            return "DISPLAY is not set. Run with -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
        if not self.tesseract:
            return f"OCR unavailable: {self.tesseract_error or 'unknown error'}"
        return None

    def diagnostics(self) -> CaptureDiagnostics:
        return CaptureDiagnostics(
            display=self.display,
            wayland_display=self.wayland_display,
            xdg_runtime_dir=self.xdg_runtime_dir,
            x11_socket=self.x11_socket,
            x11_connected=self.x11_connected,
            backend=self.backend,
            tesseract=self.tesseract,
            languages=self.languages,
            ocr_ready=self.tesseract and self.backend is not None,
            ocr_error=self.tesseract_error,
            startup_error=self.startup_error,
        )

    def capture_and_ocr(
        self,
        *,
        region: Optional[tuple[int, int, int, int]],
        lang: str,
        psm: int,
        whitelist: Optional[str],
        preserve_layout: bool,
        mode: str = "balanced",
        persist: bool = True,
    ) -> OCRResult:
        """With persist=False nothing is written to output/; the meeting recorder
        polls every few seconds and keeps its own transcript instead."""
        if self.startup_error:
            raise RuntimeError(self.startup_error)
        if mode not in OCR_MODES:
            LOGGER.warning("unsupported OCR mode %s; using balanced", mode)
            mode = "balanced"
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        raw = self._capture(region)
        capture_stats = self._image_stats(raw)
        timings["capture_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        image_hash = hashlib.md5(raw.tobytes()).hexdigest()
        cached = self.cache.get(image_hash)
        timings["hash_ms"] = (time.perf_counter() - t1) * 1000
        if cached:
            screenshot_b64 = self._encode_image_b64(self._resample_preview(raw)) if self.save_screenshots else ""
            return OCRResult(
                mode=mode,
                text=cached.text,
                processed_text=cached.text,
                raw_text=cached.text,
                candidate_texts=[],
                capture_stats=capture_stats,
                cached=True,
                image_hash=image_hash,
                timings_ms=timings,
                output_path=Path(cached.output_path),
                screenshot_b64=screenshot_b64,
            )

        t3 = time.perf_counter()
        candidate_texts = self._ocr_candidates(
            raw,
            lang=lang,
            psm=psm,
            whitelist=whitelist,
            preserve_layout=preserve_layout,
            mode=mode,
            timings=timings,
        )
        timings["ocr_ms"] = (time.perf_counter() - t3) * 1000 - timings.get("preprocess_ms", 0.0)

        t4 = time.perf_counter()
        cleaned_candidates = []
        for candidate in candidate_texts:
            cleaned = self._cleanup_text(str(candidate["text"]), preserve_layout=preserve_layout)
            candidate = {
                **candidate,
                "cleaned_text": cleaned,
                "score": self._score_candidate(cleaned, float(candidate.get("mean_confidence", -1.0))),
            }
            cleaned_candidates.append(candidate)
        winner = max(cleaned_candidates, key=lambda item: item["score"], default={"cleaned_text": "", "text": ""})
        cleaned_processed = self._candidate_lookup(cleaned_candidates, "processed") or ""
        cleaned_raw = self._candidate_lookup(cleaned_candidates, "raw") or ""
        cleaned = str(winner.get("cleaned_text", ""))
        timings["cleanup_ms"] = (time.perf_counter() - t4) * 1000

        timings["total_ms"] = sum(timings.values())
        screenshot_b64 = self._encode_image_b64(self._resample_preview(raw)) if self.save_screenshots else ""

        output_path = Path()
        if persist:
            output_path = self._persist(
                cleaned,
                cached=False,
                image_hash=image_hash,
                timings_ms=timings,
                mode=mode,
                processed_text=cleaned_processed,
                raw_text=cleaned_raw,
                candidate_texts=cleaned_candidates,
                capture_stats=capture_stats,
                screenshot_b64=screenshot_b64,
            )
        self.cache.set(CacheEntry(image_hash=image_hash, text=cleaned, output_path=str(output_path)))
        return OCRResult(
            mode=mode,
            text=cleaned,
            processed_text=cleaned_processed,
            raw_text=cleaned_raw,
            candidate_texts=cleaned_candidates,
            capture_stats=capture_stats,
            cached=False,
            image_hash=image_hash,
            timings_ms=timings,
            output_path=output_path,
            screenshot_b64=screenshot_b64,
        )

    def _capture(self, region: Optional[tuple[int, int, int, int]]) -> np.ndarray:
        if self.backend == "mss":
            raw = self._capture_with_mss(region)
            if not self._is_probably_blank(raw) or region is not None:
                return raw
            LOGGER.warning("mss capture looks blank; trying external screenshot backends")
            for backend in self._fallback_backends():
                try:
                    fallback = self._capture_with_backend(backend)
                    if not self._is_probably_blank(fallback):
                        LOGGER.info("using fallback screenshot backend %s after blank mss capture", backend)
                        return fallback
                    LOGGER.warning("fallback backend %s also looks blank", backend)
                except Exception as exc:
                    LOGGER.warning("fallback backend %s capture failed: %s", backend, exc)
            return raw

        if region is None and self.backend is not None and self.backend != "mss":
            try:
                return self._capture_with_backend(self.backend)
            except Exception as exc:
                LOGGER.warning("backend %s capture failed: %s", self.backend, exc)
                if self.wayland_display and self.display is None:
                    raise RuntimeError(f"Screenshot backend unavailable: {exc}") from exc
                LOGGER.warning("falling back to mss capture")
        return self._capture_with_mss(region)

    def _capture_with_mss(self, region: Optional[tuple[int, int, int, int]]) -> np.ndarray:
        with mss.mss() as sct:
            if region is None:
                monitor = sct.monitors[1]
            else:
                x, y, w, h = region
                monitor = {"left": x, "top": y, "width": w, "height": h}
            shot = sct.grab(monitor)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _fallback_backends(self) -> list[str]:
        return [name for name in ("spectacle", "scrot", "gnome-screenshot", "import", "grim", "portal") if shutil.which(name) or name == "portal"]

    def _capture_with_backend(self, backend: str) -> np.ndarray:
        if backend == "portal":
            return self._capture_with_portal()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            if backend == "scrot":
                cmd = ["scrot", tmp.name]
            elif backend == "spectacle":
                cmd = ["spectacle", "--background", "--nonotify", "--fullscreen", "--output", tmp.name]
            elif backend == "gnome-screenshot":
                cmd = ["gnome-screenshot", "-f", tmp.name]
            elif backend == "import":
                cmd = ["import", tmp.name]
            elif backend == "grim":
                cmd = ["grim", tmp.name]
            else:
                raise RuntimeError(f"Unsupported backend {backend}")
            env = {**os.environ}
            if self.display:
                env["DISPLAY"] = self.display
            if self.wayland_display:
                env["WAYLAND_DISPLAY"] = self.wayland_display
            if self.xdg_runtime_dir:
                env["XDG_RUNTIME_DIR"] = self.xdg_runtime_dir
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            image = cv2.imread(tmp.name, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Backend {backend} produced unreadable output")
            return image

    def _capture_with_portal(self) -> np.ndarray:
        if not self.wayland_display:
            raise RuntimeError("Wayland display not available for portal capture")
        token = f"ss_taker_{int(time.time() * 1000)}"
        call = subprocess.run(
            [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.freedesktop.portal.Desktop",
                "--object-path",
                "/org/freedesktop/portal/desktop",
                "--method",
                "org.freedesktop.portal.Screenshot.Screenshot",
                "",
                "{'interactive': <false>, 'handle_token': <'%s'>}" % token,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if call.returncode != 0:
            raise RuntimeError(f"Portal call failed: {call.stderr.strip() or call.stdout.strip()}")
        request_path = self._extract_gdbus_handle(call.stdout)
        if not request_path:
            raise RuntimeError(f"Portal did not return a request path: {call.stdout.strip()}")
        monitor = subprocess.Popen(
            [
                "gdbus",
                "monitor",
                "--session",
                "--dest",
                "org.freedesktop.portal.Desktop",
                "--object-path",
                request_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.time() + 30
            uri = None
            code = None
            while time.time() < deadline:
                if not monitor.stdout:
                    break
                ready, _, _ = select.select([monitor.stdout], [], [], 0.5)
                if not ready:
                    if monitor.poll() is not None:
                        break
                    continue
                line = monitor.stdout.readline()
                if not line:
                    if monitor.poll() is not None:
                        break
                    continue
                if "Response" in line and "uri" in line:
                    if "uint32 0" in line:
                        code = 0
                    elif "uint32" in line:
                        try:
                            code = int(line.split("uint32", 1)[1].split(",", 1)[0].strip())
                        except Exception:
                            code = None
                    if "uri" in line:
                        uri = line.split("uri", 1)[1].strip().strip("')")
                        uri = uri.split("'", 1)[1] if "'" in uri else uri
                    if code == 0 and uri:
                        break
            if code != 0 or not uri:
                raise RuntimeError("Portal screenshot failed or timed out")
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except Exception:
                monitor.kill()
        with urllib.request.urlopen(uri, timeout=20) as resp:
            data = resp.read()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Portal screenshot returned unreadable image")
        return image

    @staticmethod
    def _extract_gdbus_handle(output: str) -> str | None:
        for token in output.replace("(", " ").replace(")", " ").split():
            cleaned = token.strip("'\",")
            if cleaned.startswith("objectpath"):
                cleaned = cleaned.removeprefix("objectpath").strip("'\", ")
            if cleaned.startswith("/org/freedesktop/portal/desktop/request/"):
                return cleaned
        return None

    def capture_backend_name(self) -> str | None:
        return self.backend or "mss"

    def _ocr(self, image: np.ndarray, *, lang: str, psm: int, whitelist: Optional[str], preserve_layout: bool) -> str:
        return _ocr_worker(
            image.tobytes(),
            image.shape,
            lang,
            psm,
            whitelist,
            preserve_layout,
        )

    def _ocr_candidates(
        self,
        raw: np.ndarray,
        *,
        lang: str,
        psm: int,
        whitelist: Optional[str],
        preserve_layout: bool,
        mode: str,
        timings: dict[str, float],
    ) -> list[dict[str, object]]:
        def run(label: str, image: np.ndarray) -> dict[str, object]:
            text, mean_conf, words = self._ocr_scored(
                image,
                lang=lang,
                psm=psm,
                whitelist=whitelist,
                preserve_layout=preserve_layout,
            )
            return {
                "variant": label,
                "text": text,
                "mean_confidence": mean_conf,
                "word_count": len(words),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }

        # Screen text is usually crisp, so the raw capture alone is often good
        # enough. Only pay for preprocessing and extra passes when it is not.
        pool = self._ensure_pool()
        candidates = [pool.submit(run, "raw", raw).result()]
        if mode == "fast" or (mode == "balanced" and self._reads_well(candidates[0])):
            return candidates

        t0 = time.perf_counter()
        processed = preprocess_image(raw)
        rest = [variant for variant in self._quality_variants(raw, processed, mode) if variant[0] != "raw"]
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Tesseract releases the GIL (tesserocr) or runs as its own process
        # (pytesseract), so these variants really do run in parallel.
        candidates.extend(pool.map(lambda variant: run(*variant), rest))
        return candidates

    @staticmethod
    def _reads_well(candidate: dict[str, object]) -> bool:
        return (
            float(candidate["mean_confidence"]) >= EARLY_EXIT_CONFIDENCE
            and int(candidate["word_count"]) >= EARLY_EXIT_MIN_WORDS
        )

    def _quality_variants(self, raw: np.ndarray, processed: np.ndarray, mode: str) -> list[tuple[str, np.ndarray]]:
        if mode == "fast":
            return [("raw", raw)]
        if mode == "balanced":
            return [
                ("raw", raw),
                ("processed", processed),
            ]
        if mode != "quality_hd_ultra":
            LOGGER.warning("unsupported OCR mode %s; using balanced", mode)
            return self._quality_variants(raw, processed, "balanced")
        return [
            ("raw", raw),
            ("light", preprocess_image(raw, blur=False, upscale=False, threshold=False, sharpen=False)),
            ("hd_gray_2x", preprocess_image(raw, blur=False, upscale=True, scale=2.0, threshold=False, sharpen=True)),
            ("hd_gray_3x", preprocess_image(raw, blur=False, upscale=True, scale=3.0, threshold=False, sharpen=True)),
            ("hd_gray_4x", preprocess_image(raw, blur=False, upscale=True, scale=4.0, threshold=False, sharpen=True)),
            ("hd_binary_2x", preprocess_image(raw, blur=True, upscale=True, scale=2.0, threshold=True, sharpen=False)),
            ("hd_binary_3x", preprocess_image(raw, blur=True, upscale=True, scale=3.0, threshold=True, sharpen=False)),
            ("hd_soft_2x", preprocess_image(raw, blur=True, upscale=True, scale=2.0, threshold=False, sharpen=True)),
            ("processed", processed),
        ]

    def _ocr_scored(
        self,
        image: np.ndarray,
        *,
        lang: str,
        psm: int,
        whitelist: Optional[str],
        preserve_layout: bool,
    ) -> tuple[str, float, list[dict[str, object]]]:
        return _ocr_worker_scored(
            image.tobytes(),
            image.shape,
            lang,
            psm,
            whitelist,
            preserve_layout,
        )

    def _cleanup_text(self, text: str, *, preserve_layout: bool) -> str:
        # Note: no blanket "rn" -> "m" fix here; it turned "return" into "retum".
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for line in cleaned.splitlines():
            line = line.strip()
            if not line:
                continue
            if self._is_garbage_line(line):
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        if not preserve_layout:
            cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
            cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    def _persist(
        self,
        text: str,
        *,
        cached: bool,
        image_hash: str,
        timings_ms: dict[str, float],
        mode: str,
        processed_text: str,
        raw_text: str,
        candidate_texts: list[dict[str, object]],
        capture_stats: dict[str, object],
        screenshot_b64: str,
    ) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
        path = self.output_dir / f"{stamp}.json"
        payload = {
            "text": text,
            "mode": mode,
            "cached": cached,
            "image_hash": image_hash,
            "processed_text": processed_text,
            "raw_text": raw_text,
            "chosen_text": text,
            "candidate_texts": candidate_texts,
            "capture_stats": capture_stats,
            "timings_ms": timings_ms,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if screenshot_b64:
            payload["full_ss_base64"] = screenshot_b64
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        LOGGER.info("persisted OCR output to %s", path)
        return path

    @staticmethod
    def _score_candidate(text: str, mean_confidence: float) -> float:
        if not text.strip():
            return -1000.0
        word_count = len(text.split())
        char_count = len(text.strip())
        confidence = mean_confidence if mean_confidence >= 0 else 0.0
        junk_penalty = OCRService._junk_ratio(text) * 80.0
        return confidence + word_count * 2.0 + min(char_count / 20.0, 20.0) - junk_penalty

    @staticmethod
    def _is_garbage_line(line: str) -> bool:
        if len(line) <= 2 and not any(ch.isalnum() for ch in line):
            return True
        return OCRService._junk_ratio(line) > 0.55

    @staticmethod
    def _junk_ratio(text: str) -> float:
        if not text:
            return 1.0
        meaningful = sum(ch.isalnum() or ch.isspace() or ch in ".,:/_+-@#()[]{}" for ch in text)
        return 1.0 - (meaningful / len(text))

    @staticmethod
    def _candidate_lookup(candidates: list[dict[str, object]], variant: str) -> str | None:
        for candidate in candidates:
            if candidate.get("variant") == variant:
                return str(candidate.get("cleaned_text", ""))
        return None

    @staticmethod
    def _encode_image_b64(image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _resample_preview(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
        height, width = image.shape[:2]
        if width <= max_width and height <= max_height:
            return image
        scale = min(max_width / float(width), max_height / float(height))
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _image_stats(image: np.ndarray) -> dict[str, object]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "mean": round(float(np.mean(gray)), 3),
            "stddev": round(float(np.std(gray)), 3),
            "min": int(np.min(gray)),
            "max": int(np.max(gray)),
            "probably_blank": OCRService._is_probably_blank(image),
        }

    @staticmethod
    def _is_probably_blank(image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.std(gray)) < 2.0 or int(np.max(gray)) - int(np.min(gray)) < 8
