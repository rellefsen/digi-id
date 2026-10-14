"""Stub/mock decoder when a real backend is unavailable."""

from __future__ import annotations

from pathlib import Path

from digimode.decode.base import DecodeResult, Decoder
from digimode.ingest import AudioBuffer


class StubDecoder(Decoder):
    def __init__(self, mode: str = "FT8", reason: str = "no backend"):
        self.mode = mode
        self.reason = reason

    def decode(self, audio: AudioBuffer, work_dir: Path | None = None) -> DecodeResult:
        return DecodeResult(
            mode=self.mode,
            lines=[],
            raw_output="",
            backend="stub",
            ok=False,
            error=(
                f"Stub decoder for {self.mode}: {self.reason}. "
                "Install wsjtx (provides /usr/bin/jt9) for real FT8 decode. "
                "JS8Call adapter is planned for M2."
            ),
        )
