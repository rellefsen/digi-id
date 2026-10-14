#!/usr/bin/env python3
"""FT8 windows lock to UTC slots for device capture, not for file playback."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from digimode.cli import build_parser  # noqa: E402
from digimode.ft8_slots import (  # noqa: E402
    Ft8WindowCutter,
    capture_epoch_utc,
    frames_until_ft8_slot,
)
from digimode.ingest import AudioBuffer, load_wav  # noqa: E402
from digimode.live import format_status, live_listen_record, parse_ft8_align  # noqa: E402


def check_frames_until() -> None:
    sr = 48000
    assert frames_until_ft8_slot(0.0, sr) == 0
    assert frames_until_ft8_slot(15.0, sr) == 0
    assert frames_until_ft8_slot(30.0, sr) == 0
    # Just after a boundary: keep the slot. Just before: still wait.
    assert frames_until_ft8_slot(0.04, sr) == 0
    assert frames_until_ft8_slot(14.96, sr) == int(round(0.04 * sr))
    assert frames_until_ft8_slot(5.0, sr) == 10 * sr
    assert frames_until_ft8_slot(7.5, sr) == int(round(7.5 * sr))
    # Python modulo keeps a negative timestamp on the same 15 s grid.
    assert frames_until_ft8_slot(-1.0, sr) == 1 * sr
    try:
        frames_until_ft8_slot(0.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("sample_rate 0 should fail")


def check_capture_epoch() -> None:
    # Callback at t=1000, one second already captured, ADC 10 ms before now.
    epoch = capture_epoch_utc(1000.0, 48000, 48000, 0.01)
    assert abs(epoch - 998.99) < 1e-9
    # Unusable delays fall back to "now is the first sample of this block".
    assert capture_epoch_utc(1000.0, 0, 48000, -1.0) == 1000.0
    assert capture_epoch_utc(1000.0, 0, 48000, 5.0) == 1000.0


def check_cutter() -> None:
    sr = 100
    period = 15.0
    need = 1500

    aligned = Ft8WindowCutter(sr, period_s=period, align_utc=True, tolerance_s=0.05)
    aligned.set_epoch(5.0)  # 5 s into the slot → drop 10 s
    data = np.arange(4000, dtype=np.float32)
    windows = aligned.push(data, 0)
    assert aligned.lead_in_frames == 1000
    assert len(windows) == 2
    assert windows[0].frame0 == 1000
    assert list(windows[0].samples[:3]) == [1000, 1001, 1002]
    assert windows[0].samples.size == need
    assert windows[1].frame0 == 2500
    assert list(windows[1].samples[:2]) == [2500, 2501]
    assert aligned.slots == 2
    assert aligned.hold_frames == 0

    # 40 ms past a boundary is close enough to start immediately.
    close = Ft8WindowCutter(sr, period_s=period, align_utc=True, tolerance_s=0.05)
    close.set_epoch(0.04)
    got = close.push(np.ones(need, dtype=np.float32), 0)
    assert close.lead_in_frames == 0
    assert len(got) == 1 and got[0].frame0 == 0

    # 100 ms early still waits for the boundary. Small blocks must land
    # on that boundary, not one tolerance early.
    early = Ft8WindowCutter(sr, period_s=period, align_utc=True, tolerance_s=0.05)
    early.set_epoch(14.90)
    early.push(np.arange(20, dtype=np.float32), 0)
    assert early.lead_in_frames == 10
    assert early.hold_frames == 0  # window is open on the remaining 10
    walked = Ft8WindowCutter(48000, align_utc=True, tolerance_s=0.05)
    walked.set_epoch(5.0)
    frame = 0
    step = 480
    total = 25 * 48000
    while frame < total:
        walked.push(np.zeros(step, dtype=np.float32), frame)
        frame += step
    assert walked.lead_in_frames == 10 * 48000
    assert walked.slots == 1

    # A gap abandons the open period instead of splicing across the hole.
    gap = Ft8WindowCutter(sr, period_s=period, align_utc=True, tolerance_s=0.05)
    gap.set_epoch(0.0)
    assert gap.push(np.arange(50, dtype=np.float32), 0) == []
    assert gap.push(np.arange(10, dtype=np.float32), 200) == []
    assert gap.resyncs == 1
    assert gap.lead_in_frames == 10
    assert gap.slots == 0

    # Stream mode ignores the clock and keeps the whole leading period.
    stream = Ft8WindowCutter(sr, period_s=period, align_utc=False)
    stream.set_epoch(5.0)
    got = stream.push(np.arange(need + 100, dtype=np.float32), 0)
    assert stream.lead_in_frames == 0
    assert len(got) == 1 and got[0].frame0 == 0
    assert list(got[0].samples[:3]) == [0, 1, 2]
    tail = stream.flush()
    assert tail is None  # 100 frames is under 90% of a period

    near = Ft8WindowCutter(sr, period_s=period, align_utc=False)
    near.push(np.ones(1350, dtype=np.float32), 0)
    kept = near.flush()
    assert kept is not None and kept.samples.size == 1350
    short = Ft8WindowCutter(sr, period_s=period, align_utc=False)
    short.push(np.ones(1349, dtype=np.float32), 0)
    assert short.flush() is None


def _session(path: Path, **kwargs):
    session = live_listen_record(
        mode=kwargs.pop("mode", "listen-only"),
        from_file=path,
        pace_realtime=False,
        sample_rate=48000,
        **kwargs,
    )
    session.start()
    assert session.wait(90), "session did not finish"
    session.stop()
    return session.status()


def check_session_geometry(tmp: Path) -> None:
    ft8 = load_wav(ROOT / "samples" / "ft8_cq_k1abc.wav").resample(48000)
    assert ft8.samples.size == 15 * 48000
    lead = np.full(10 * 48000, 1e-4, dtype=np.float32)
    padded = tmp / "lead-plus-ft8.wav"
    AudioBuffer(
        samples=np.concatenate([lead, np.asarray(ft8.samples, dtype=np.float32)]),
        sample_rate=48000,
    ).write_wav(padded)

    # Sample 0 is 5 s into a slot, so the 10 s lead is dropped and the
    # FT8 recording occupies the whole next period.
    on_slot = _session(padded, ft8_align="utc", ft8_epoch_utc=5.0)
    assert on_slot.ft8_align_utc is True
    assert on_slot.ft8_lead_in_frames == 10 * 48000
    assert on_slot.ft8_slots == 1
    assert on_slot.decode_attempts == 0
    assert "ft8=utc" in format_status(on_slot)

    # Starting on a boundary keeps the lead and splits the transmission.
    split = _session(padded, ft8_align="utc", ft8_epoch_utc=0.0)
    assert split.ft8_lead_in_frames == 0
    assert split.ft8_slots == 1

    # auto + file stays on the stream clock.
    streamed = _session(ROOT / "samples" / "ft8_cq_k1abc.wav")
    assert streamed.ft8_align_utc is False
    assert streamed.ft8_lead_in_frames == 0
    assert streamed.ft8_slots == 1
    assert "ft8=stream" in format_status(streamed)

    # A fixed clock 5 s into a slot is what the device callback would stamp.
    stamped = _session(
        ROOT / "samples" / "ft8_cq_k1abc.wav",
        ft8_align="utc",
        utc_clock=lambda: 1010.0,  # 1010 % 15 == 5
    )
    assert stamped.ft8_lead_in_frames == 10 * 48000
    assert stamped.ft8_slots == 0

    # An explicit epoch wins over the clock.
    forced = _session(
        ROOT / "samples" / "ft8_cq_k1abc.wav",
        ft8_align="utc",
        ft8_epoch_utc=0.0,
        utc_clock=lambda: 1010.0,
    )
    assert forced.ft8_lead_in_frames == 0
    assert forced.ft8_slots == 1


def check_aligned_decode(tmp: Path) -> None:
    ft8 = load_wav(ROOT / "samples" / "ft8_cq_k1abc.wav").resample(48000)
    lead = np.full(10 * 48000, 1e-4, dtype=np.float32)
    padded = tmp / "lead-plus-ft8.wav"
    AudioBuffer(
        samples=np.concatenate([lead, np.asarray(ft8.samples, dtype=np.float32)]),
        sample_rate=48000,
    ).write_wav(padded)

    hit = _session(
        padded,
        mode="listen+decode",
        squelch_policy="always-on",
        ft8_align="utc",
        ft8_epoch_utc=5.0,
    )
    assert hit.decode_attempts == 1, hit
    assert hit.last_decode_line and "K1ABC" in hit.last_decode_line, hit

    miss = _session(
        padded,
        mode="listen+decode",
        squelch_policy="always-on",
        ft8_align="utc",
        ft8_epoch_utc=0.0,
    )
    text = " ".join(miss.decode_lines)
    assert "K1ABC" not in text, miss


def check_defaults() -> None:
    assert parse_ft8_align("auto", from_file=False) is True
    assert parse_ft8_align("auto", from_file=True) is False
    assert parse_ft8_align("utc", from_file=True) is True
    assert parse_ft8_align("stream", from_file=False) is False
    assert live_listen_record().config.ft8_align_utc is True
    wav = ROOT / "samples" / "ft8_cq_k1abc.wav"
    assert live_listen_record(from_file=wav).config.ft8_align_utc is False
    assert live_listen_record(from_file=wav, ft8_align="utc").config.ft8_align_utc is True
    args = build_parser().parse_args(["--live", "--ft8-align", "utc"])
    assert args.ft8_align == "utc"
    try:
        parse_ft8_align("slots", from_file=False)
    except ValueError:
        pass
    else:
        raise AssertionError("bad alignment mode should fail")


def main() -> int:
    check_frames_until()
    check_capture_epoch()
    check_cutter()
    check_defaults()
    with tempfile.TemporaryDirectory(prefix="digimode-ft8-slot-") as tmp:
        path = Path(tmp)
        check_session_geometry(path)
        check_aligned_decode(path)
    print("ft8 slot alignment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
