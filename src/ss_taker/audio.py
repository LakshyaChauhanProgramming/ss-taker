from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:  # optional: pip install -e ".[audio]"
    WhisperModel = None

LOGGER = logging.getLogger("ss_taker")

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le
CHUNK_SECONDS = float(os.environ.get("SS_TAKER_AUDIO_CHUNK_SECONDS", "30"))
WHISPER_MODEL = os.environ.get("SS_TAKER_WHISPER_MODEL", "base")
# Chunks quieter than this (RMS of [-1, 1] samples) are skipped: Whisper tends
# to invent "Thank you." on silence.
SILENCE_RMS = float(os.environ.get("SS_TAKER_AUDIO_SILENCE_RMS", "0.003"))


def audio_available() -> str | None:
    """None when audio capture can run, otherwise the reason it cannot."""
    if WhisperModel is None:
        return "faster-whisper is not installed (pip install -e '.[audio]')"
    if shutil.which("parec") is None:
        return "parec not found (install pulseaudio-utils)"
    return None


def resolve_source(name: str) -> str | None:
    """Map "others"/"me" to a PulseAudio/PipeWire source.

    "others" is the monitor of the default output, i.e. what the meeting app plays;
    "me" is the default microphone. Any other value is used as a source name as-is.
    """
    if name == "me":
        return "@DEFAULT_SOURCE@"
    if name != "others":
        return name
    try:
        sink = subprocess.run(
            ["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sink = ""
    return f"{sink}.monitor" if sink else "@DEFAULT_MONITOR@"


@dataclass(slots=True)
class SpeechSegment:
    start: float
    """Seconds since the recording started."""
    text: str


class Transcriber:
    """Loads one Whisper model lazily and shares it between all recordings."""

    _instance: "Transcriber | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, model_size: str = WHISPER_MODEL) -> None:
        if WhisperModel is None:
            raise RuntimeError(audio_available() or "faster-whisper unavailable")
        self.model_size = model_size
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._lock = threading.Lock()

    @classmethod
    def shared(cls) -> "Transcriber":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def transcribe(self, samples: np.ndarray, offset: float, language: str | None = None) -> list[SpeechSegment]:
        if samples.size == 0 or float(np.sqrt(np.mean(samples**2))) < SILENCE_RMS:
            return []
        with self._lock:
            segments, _info = self._model.transcribe(
                samples, language=language, vad_filter=True, beam_size=1, condition_on_previous_text=False
            )
            return [
                SpeechSegment(start=offset + seg.start, text=seg.text.strip())
                for seg in segments
                if seg.text.strip()
            ]


class AudioRecorder:
    """Records one source with parec and transcribes it in fixed-size chunks.

    Audio stays in memory; only the recognized text reaches on_segment.
    """

    def __init__(
        self,
        name: str,
        on_segment: Callable[[str, SpeechSegment], None],
        *,
        language: str | None = None,
        chunk_seconds: float = CHUNK_SECONDS,
        transcriber_factory: Callable[[], Transcriber] = Transcriber.shared,
    ) -> None:
        self.name = name
        self.on_segment = on_segment
        self.language = language
        self.chunk_bytes = int(chunk_seconds * SAMPLE_RATE) * BYTES_PER_SAMPLE
        self.transcriber_factory = transcriber_factory
        self.error: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._chunks: queue.Queue[tuple[float, bytes] | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        cmd = ["parec", "--format=s16le", f"--rate={SAMPLE_RATE}", "--channels=1", "--raw"]
        source = resolve_source(self.name)
        if source:
            cmd.append(f"--device={source}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._reader = threading.Thread(target=self._read_loop, name=f"audio-read-{self.name}", daemon=True)
        self._worker = threading.Thread(target=self._transcribe_loop, name=f"audio-asr-{self.name}", daemon=True)
        self._reader.start()
        self._worker.start()

    def stop(self) -> None:
        """Stop recording, then wait for the queued audio to be transcribed."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._reader is not None:
            self._reader.join()
        if self._worker is not None:
            self._worker.join()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        offset = 0.0
        buffer = b""
        while True:
            data = self._proc.stdout.read(self.chunk_bytes - len(buffer))
            if not data:
                break
            buffer += data
            if len(buffer) >= self.chunk_bytes:
                self._chunks.put((offset, buffer))
                offset += len(buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                buffer = b""
        if buffer:
            self._chunks.put((offset, buffer))
        if self._proc.wait() not in (0, -15) and self._proc.stderr is not None:
            self.error = self._proc.stderr.read().decode(errors="replace").strip() or "parec exited"
            LOGGER.warning("audio source %s failed: %s", self.name, self.error)
        self._chunks.put(None)

    def _transcribe_loop(self) -> None:
        transcriber: Transcriber | None = None
        while (item := self._chunks.get()) is not None:
            offset, raw = item
            try:
                transcriber = transcriber or self.transcriber_factory()
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                for segment in transcriber.transcribe(samples, offset, self.language):
                    self.on_segment(self.name, segment)
            except Exception as exc:  # keep recording even if one chunk fails
                self.error = f"transcription failed: {exc}"
                LOGGER.exception("audio transcription failed for %s", self.name)
