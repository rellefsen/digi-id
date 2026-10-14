#!/usr/bin/env python3
"""Offline smoke test for M1."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from digimode.classify import classify
from digimode.decode import get_ft8_decoder
from digimode.ingest import load_wav
from digimode.live import LiveUnavailableError, live_listen_record


def main() -> int:
    wav = ROOT / "samples" / "ft8_cq_k1abc.wav"
    audio = load_wav(wav)
    c = classify(audio)
    assert c.mode == "FT8", c
    dec = get_ft8_decoder().decode(audio)
    assert dec.ok and any("K1ABC" in line for line in dec.lines), dec
    session = live_listen_record()
    try:
        session.start()
        session.stop()
        print("FAIL: live start without a device should raise")
        return 1
    except LiveUnavailableError:
        pass
    import subprocess
    demo = subprocess.run([sys.executable, str(ROOT / "tests" / "live_from_file_demo.py")], cwd=ROOT)
    if demo.returncode != 0:
        return demo.returncode
    slots = subprocess.run([sys.executable, str(ROOT / "tests" / "test_ft8_slots.py")], cwd=ROOT)
    if slots.returncode != 0:
        return slots.returncode
    print("smoke OK:", c.mode, dec.lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
