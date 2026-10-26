from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional


@dataclass(slots=True)
class CacheEntry:
    image_hash: str
    text: str
    output_path: str


class ResultCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self._entry: Optional[CacheEntry] = None

    def get(self, image_hash: str) -> Optional[CacheEntry]:
        with self._lock:
            if self._entry and self._entry.image_hash == image_hash:
                return self._entry
            return None

    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entry = entry
