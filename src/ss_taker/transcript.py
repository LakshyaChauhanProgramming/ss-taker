from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from threading import Lock

# How many recent screen lines a new OCR line is compared against. Caption and
# chat panes keep showing the same lines for a while, so each capture repeats
# most of the previous one.
RECENT_LINES = 60
SIMILAR_RATIO = 0.85
MIN_LINE_CHARS = 4


@dataclass(slots=True)
class Segment:
    t: float
    """Seconds since the meeting started."""
    source: str
    """"screen" for OCR text, "audio:<name>" for speech, e.g. audio:others."""
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize(line: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", line.lower()).split())


class Transcript:
    """Thread-safe, time-ordered list of segments from the OCR and audio workers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._segments: list[Segment] = []
        # (normalized text, index into _segments) of the latest screen lines.
        self._recent_screen: list[tuple[str, int]] = []

    def add_screen_text(self, t: float, text: str) -> int:
        """Add the lines of one OCR capture that were not seen recently.

        A live caption usually grows between captures ("we should" -> "we should
        ship friday"); the longer version replaces the earlier segment instead of
        being added twice. Returns the number of new segments.
        """
        added = 0
        with self._lock:
            for line in text.splitlines():
                line = line.strip()
                norm = _normalize(line)
                if len(norm) < MIN_LINE_CHARS:
                    continue
                match = self._match_recent(norm)
                if match is None:
                    self._segments.append(Segment(t=t, source="screen", text=line))
                    self._remember(norm, len(self._segments) - 1)
                    added += 1
                    continue
                seen_norm, index = match
                if len(norm) > len(seen_norm) and norm.startswith(seen_norm):
                    self._segments[index].text = line
                    self._recent_screen = [
                        (norm if i == index else n, i) for n, i in self._recent_screen
                    ]
        return added

    def add(self, segment: Segment) -> None:
        with self._lock:
            self._segments.append(segment)

    def segments(self) -> list[Segment]:
        with self._lock:
            return sorted(self._segments, key=lambda s: s.t)

    def __len__(self) -> int:
        with self._lock:
            return len(self._segments)

    def _match_recent(self, norm: str) -> tuple[str, int] | None:
        for seen, index in reversed(self._recent_screen):
            if norm == seen or seen.startswith(norm):
                return seen, index
            if norm.startswith(seen):
                return seen, index
            if SequenceMatcher(None, norm, seen).ratio() >= SIMILAR_RATIO:
                return seen, index
        return None

    def _remember(self, norm: str, index: int) -> None:
        self._recent_screen.append((norm, index))
        if len(self._recent_screen) > RECENT_LINES:
            del self._recent_screen[0]


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def render(segments: list[Segment]) -> str:
    """One line per segment, e.g. "[03:15] (audio:others) let's ship friday"."""
    return "\n".join(f"[{format_timestamp(s.t)}] ({s.source}) {s.text}" for s in segments)
