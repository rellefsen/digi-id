"""Decoder adapters for digimodes."""

from digimode.decode.base import DecodeResult, Decoder
from digimode.decode.ft8_jt9 import Ft8Jt9Decoder, find_jt9
from digimode.decode.stub import StubDecoder

__all__ = [
    "DecodeResult",
    "Decoder",
    "Ft8Jt9Decoder",
    "StubDecoder",
    "find_jt9",
    "get_ft8_decoder",
]


def get_ft8_decoder() -> Decoder:
    """Prefer real jt9; fall back to stub documenting the gap."""
    jt9 = find_jt9()
    if jt9 is not None:
        return Ft8Jt9Decoder(jt9_path=jt9)
    return StubDecoder(mode="FT8", reason="jt9/WSJT-X not found on PATH")
