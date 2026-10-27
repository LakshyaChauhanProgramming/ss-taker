from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from . import summarizer
from .audio import AudioRecorder, SpeechSegment, audio_available
from .transcript import Segment, Transcript, render

LOGGER = logging.getLogger("ss_taker")


class _OCR(Protocol):
    def capture_and_ocr(self, **kwargs) -> object: ...


@dataclass(slots=True)
class MeetingConfig:
    title: str = ""
    ocr: bool = True
    region: Optional[tuple[int, int, int, int]] = None
    """Screen area to read, e.g. the captions or chat pane. None reads the whole screen."""
    ocr_interval_seconds: float = 5.0
    lang: str = "eng"
    audio: bool = True
    audio_sources: list[str] = field(default_factory=lambda: ["others"])
    """"others" = what the computer plays (the other participants), "me" = microphone."""
    audio_language: Optional[str] = None
    """Whisper language code such as "hi" or "en"; None lets Whisper detect it."""


class MeetingSession:
    def __init__(
        self,
        config: MeetingConfig,
        ocr_service: _OCR | None,
        output_dir: Path,
        recorder_factory: Callable[..., AudioRecorder] = AudioRecorder,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.ocr_service = ocr_service
        self.output_dir = output_dir
        self.recorder_factory = recorder_factory
        self.transcript = Transcript()
        self.status = "created"
        self.started_at = 0.0
        self.stopped_at: float | None = None
        self.summary: summarizer.MeetingSummary | None = None
        self.errors: list[str] = []
        self.ocr_captures = 0
        self._stop = threading.Event()
        self._ocr_thread: threading.Thread | None = None
        self._recorders: list[tuple[AudioRecorder, float]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self.started_at = time.time()
        self.status = "recording"
        if self.config.ocr and self.ocr_service is not None:
            self._ocr_thread = threading.Thread(target=self._ocr_loop, name=f"meeting-ocr-{self.id}", daemon=True)
            self._ocr_thread.start()
        if self.config.audio:
            reason = audio_available()
            if reason:
                self._error(f"audio disabled: {reason}")
            else:
                for name in self.config.audio_sources:
                    recorder = self.recorder_factory(
                        name, self._on_speech, language=self.config.audio_language
                    )
                    try:
                        recorder.start()
                    except OSError as exc:
                        self._error(f"audio source {name} failed to start: {exc}")
                        continue
                    self._recorders.append((recorder, time.time() - self.started_at))
        LOGGER.info("meeting %s recording (ocr=%s, audio=%d sources)", self.id, self.config.ocr, len(self._recorders))

    def stop(self) -> None:
        with self._lock:
            if self.status != "recording":
                return
            self.status = "stopping"
        self._stop.set()
        if self._ocr_thread is not None:
            self._ocr_thread.join()
        for recorder, _offset in self._recorders:
            # Waits for the last audio chunk to be transcribed.
            recorder.stop()
            if recorder.error:
                self._error(f"audio:{recorder.name}: {recorder.error}")
        self.stopped_at = time.time()
        self.status = "stopped"
        self.save()
        LOGGER.info("meeting %s stopped with %d segments", self.id, len(self.transcript))

    def summarize(self, *, language: str = "English") -> summarizer.MeetingSummary:
        self.summary = summarizer.summarize(self.transcript_text(), title=self.config.title, language=language)
        self.save()
        return self.summary

    def transcript_text(self) -> str:
        return render(self.transcript.segments())

    def info(self) -> dict[str, object]:
        end = self.stopped_at or time.time()
        return {
            "id": self.id,
            "title": self.config.title,
            "status": self.status,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "duration_seconds": round(end - self.started_at, 1) if self.started_at else 0.0,
            "ocr": self.config.ocr,
            "audio_sources": [r.name for r, _ in self._recorders],
            "ocr_captures": self.ocr_captures,
            "segments": len(self.transcript),
            "errors": list(self.errors),
            "summary": self.summary.to_dict() if self.summary else None,
        }

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.id}.json"
        payload = {**self.info(), "segments": [s.to_dict() for s in self.transcript.segments()]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _ocr_loop(self) -> None:
        assert self.ocr_service is not None
        last_error = ""
        while not self._stop.is_set():
            t = time.time() - self.started_at
            try:
                result = self.ocr_service.capture_and_ocr(
                    region=self.config.region,
                    lang=self.config.lang,
                    psm=6,
                    whitelist=None,
                    preserve_layout=True,  # keep line breaks so captions dedupe per line
                    mode="fast",
                    persist=False,
                )
                self.ocr_captures += 1
                self.transcript.add_screen_text(t, result.text)
                last_error = ""
            except Exception as exc:
                # Log a repeating failure once, not every few seconds.
                if str(exc) != last_error:
                    self._error(f"screen capture failed: {exc}")
                    last_error = str(exc)
            self._stop.wait(self.config.ocr_interval_seconds)

    def _on_speech(self, name: str, segment: SpeechSegment) -> None:
        offset = next((o for r, o in self._recorders if r.name == name), 0.0)
        self.transcript.add(Segment(t=offset + segment.start, source=f"audio:{name}", text=segment.text))

    def _error(self, message: str) -> None:
        LOGGER.warning("meeting %s: %s", self.id, message)
        self.errors.append(message)


class MeetingManager:
    """Keeps meetings in memory. Only one can record at a time since they share the screen and audio."""

    def __init__(self, output_dir: str | Path = "output/meetings") -> None:
        self.output_dir = Path(output_dir)
        self._meetings: dict[str, MeetingSession] = {}
        self._lock = threading.Lock()

    def start(self, config: MeetingConfig, ocr_service: _OCR | None, **kwargs) -> MeetingSession:
        with self._lock:
            active = self.active()
            if active is not None:
                raise MeetingBusyError(active.id)
            session = MeetingSession(config, ocr_service, self.output_dir, **kwargs)
            self._meetings[session.id] = session
            session.start()
            return session

    def get(self, meeting_id: str) -> MeetingSession | None:
        return self._meetings.get(meeting_id)

    def active(self) -> MeetingSession | None:
        return next((m for m in self._meetings.values() if m.status in ("recording", "stopping")), None)

    def all(self) -> list[MeetingSession]:
        return sorted(self._meetings.values(), key=lambda m: m.started_at, reverse=True)

    def stop_all(self) -> None:
        for meeting in list(self._meetings.values()):
            meeting.stop()


class MeetingBusyError(RuntimeError):
    def __init__(self, meeting_id: str) -> None:
        super().__init__(f"Meeting {meeting_id} is already recording")
        self.meeting_id = meeting_id
