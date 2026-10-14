"""Decoder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from digimode.ingest import AudioBuffer


@dataclass
class DecodeResult:
    mode: str
    lines: list[str] = field(default_factory=list)
    raw_output: str = ""
    backend: str = ""
    ok: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "backend": self.backend,
            "lines": list(self.lines),
            "error": self.error,
        }


class Decoder(ABC):
    """Adapter interface for offline file-based decoders."""

    mode: str

    @abstractmethod
    def decode(self, audio: AudioBuffer, work_dir: Path | None = None) -> DecodeResult:
        ...
