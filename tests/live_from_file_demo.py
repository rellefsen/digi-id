#!/usr/bin/env python3
"""Scripted live demo: ring-buffer tee without a sound card.

1. Squelch unit checks (open / hang / close) and ring wrap.
2. Fast file feed of a loud tone: monitor gate vs pre-squelch WAV.
3. Weak FT8 file with gate-off skips jt9; always-on CLI prints CQ K1ABC FN42.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from digimode.ingest import AudioBuffer, load_wav
from digimode.live import (  # noqa: E402
    LiveUnavailableError,
    NoiseSquelch,
    RingBuffer,
    SquelchConfig,
    decode_window_allowed,
    live_listen_record,
)
from digimode.live import LiveMode, SquelchPolicy


def _const(db: float, n: int) -> np.ndarray:
    amp = 10 ** (db / 20.0)
    return np.full(n, amp, dtype=np.float32)


def check_ring() -> None:
    ring = RingBuffer(8)
    ring.write(np.arange(5, dtype=np.float32))
    data, pos, over = ring.read(0, 3)
    assert not over and pos == 3 and list(data) == [0, 1, 2]
    ring.write(np.arange(5, 12, dtype=np.float32))  # write_abs = 12, cap 8, reader at 3
    data, pos, over = ring.read(pos, 8)
    assert over, "reader must skip when more than capacity behind"
    assert data.size > 0


def check_squelch() -> None:
    sr = 48000
    n = 480  # 10 ms
    sq = NoiseSquelch(SquelchConfig(open_db=6, close_db=3, hang_ms=500), sr)
    assert sq.process(_const(-60, n)) is False
    # +4 dB is below the 6 dB open threshold.
    assert sq.process(_const(-56, n)) is False
    assert sq.process(_const(-50, n)) is True  # +10 dB
    assert sq.ever_open
    # +4 dB relative to the original floor stays above the 3 dB close line.
    assert sq.process(_const(-56, n)) is True
    assert sq.process(_const(-80, n)) is True  # hang started
    # 40 more 10 ms blocks = 400 ms; hang is 500 ms, still open.
    for _ in range(40):
        assert sq.process(_const(-80, n)) is True
    # Past 500 ms of hang.
    for _ in range(20):
        sq.process(_const(-80, n))
    assert sq.is_open() is False
    assert sq.any_open(0, n) is False
    assert sq.any_open(2 * n, 3 * n) is True
    assert decode_window_allowed(LiveMode.BOTH, SquelchPolicy.GATE_OFF, False) is False
    assert decode_window_allowed(LiveMode.BOTH, SquelchPolicy.ALWAYS_ON, False) is True
    assert decode_window_allowed(LiveMode.LISTEN, SquelchPolicy.ALWAYS_ON, True) is False


def _tone_wav(path: Path) -> int:
    sr = 48000
    noise = _const(-60, int(0.30 * sr))
    t = np.arange(int(0.40 * sr)) / sr
    tone = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    tail = _const(-60, int(1.00 * sr))
    samples = np.concatenate([noise, tone, tail])
    AudioBuffer(samples=samples, sample_rate=sr).write_wav(path)
    return int(samples.size)


def check_tone_tee(tmp: Path) -> None:
    src = tmp / "tone.wav"
    rec = tmp / "tone-rec.wav"
    n = _tone_wav(src)
    session = live_listen_record(
        mode="listen-only",
        from_file=src,
        record_path=rec,
        squelch_hang_ms=200,
        pace_realtime=False,
        sample_rate=48000,
    )
    session.start()
    assert session.wait(30), "tone session did not finish"
    session.stop()
    st = session.status()
    assert st.capture_opens == 1
    assert st.input_backend == "file"
    assert st.captured_frames == n
    assert st.wav_frames == n, (st.wav_frames, n)
    assert st.monitor_frames == n
    assert st.squelch_ever_open
    assert st.squelch_open is False
    assert 0 < st.monitor_audible_frames < st.monitor_frames
    assert st.decode_attempts == 0
    assert st.monitor_latency_ms < 80
    got = load_wav(rec)
    assert abs(len(got.samples) - n) <= 1
    assert float(np.max(np.abs(got.samples))) > 0.1  # pre-squelch tone survived
    # Panic mute: same signal, monitor path runs but stays silent.
    session = live_listen_record(
        mode="listen+decode",
        from_file=src,
        mute=True,
        squelch_policy="always-on",
        force_mode="RTTY",  # stub decoder, do not spend a jt9 run on a tone
        pace_realtime=False,
    )
    # Tone file is ~1.7 s, shorter than the 15 s FT8 window.
    session.start()
    assert session.wait(30)
    session.stop()
    st = session.status()
    assert st.monitor_frames > 0
    assert st.monitor_audible_frames == 0
    assert st.squelch_ever_open


def check_gate_off(tmp: Path) -> None:
    rec = tmp / "gate-off.wav"
    session = live_listen_record(
        mode="listen+decode",
        from_file=ROOT / "samples" / "ft8_cq_k1abc.wav",
        record_path=rec,
        squelch_policy="gate-off",
        pace_realtime=False,
    )
    session.start()
    assert session.wait(60)
    session.stop()
    st = session.status()
    assert st.decode_attempts == 0, st
    assert st.decode_skipped_gate >= 1, st
    assert st.squelch_ever_open is False
    assert st.wav_frames == st.captured_frames > 0
    assert st.analysis_frames == st.captured_frames
    assert st.monitor_frames == st.captured_frames
    assert st.monitor_audible_frames == 0


def check_cli_always_on(tmp: Path) -> None:
    rec = tmp / "live-ft8.wav"
    cmd = [
        sys.executable,
        "-m",
        "digimode",
        "--live",
        "--live-from-file",
        str(ROOT / "samples" / "ft8_cq_k1abc.wav"),
        "--live-mode",
        "listen+decode",
        "--squelch-policy",
        "always-on",
        "--squelch-open-db",
        "6",
        "--squelch-close-db",
        "3",
        "--squelch-hang-ms",
        "500",
        "--record",
        str(rec),
        "--live-fast",
    ]
    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"live CLI failed: {proc.returncode}")
    if "CQ K1ABC FN42" not in proc.stdout:
        raise SystemExit("FT8 text missing from live stream decode")
    if "class=FT8" not in proc.stdout:
        raise SystemExit("classifier did not report FT8 on the live window")
    if "monitor=noop" not in proc.stdout and "monitor=sounddevice" not in proc.stdout:
        raise SystemExit("monitor status missing")
    audio = load_wav(rec)
    if float(np.max(np.abs(audio.samples))) < 1e-4:
        raise SystemExit("recording looks like silence (squelch leaked into WAV)")
    print("live demo OK", rec, "frames", len(audio.samples))


def check_no_device() -> None:
    session = live_listen_record()
    try:
        session.start()
    except LiveUnavailableError:
        return
    session.stop()
    raise SystemExit("expected LiveUnavailableError with no capture device")


def main() -> int:
    check_ring()
    check_squelch()
    check_no_device()
    with tempfile.TemporaryDirectory(prefix="digimode-live-") as tmp:
        tmp_path = Path(tmp)
        check_tone_tee(tmp_path)
        check_gate_off(tmp_path)
        check_cli_always_on(tmp_path)
    print("live_from_file_demo OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
