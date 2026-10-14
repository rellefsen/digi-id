"""CLI: python -m digimode path/to/file.wav"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from digimode import __version__
from digimode.classify import classify
from digimode.decode import get_ft8_decoder
from digimode.decode.ft8_jt9 import Ft8Jt9Decoder, find_jt9
from digimode.decode.stub import StubDecoder
from digimode.ingest import load_wav


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="digimode",
        description="Offline digital-mode identifier and FT8 decoder, with a local live audio tee.",
    )
    p.add_argument("wav", type=Path, nargs="?", default=None, help="Input WAV file (mono or stereo)")
    p.add_argument(
        "--mode",
        choices=("auto", "FT8", "RTTY", "PSK31"),
        default="auto",
        help="Force a mode (skip classify for decode path); default: auto",
    )
    p.add_argument(
        "--decode/--no-decode",
        default=True,
        dest="do_decode",
        help="Attempt decode when mode is FT8 (default: on)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text",
    )
    p.add_argument(
        "--jt9",
        type=str,
        default=None,
        help="Path to jt9 binary (else PATH / DIGIMODE_JT9)",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Live listen + record + decode (local device or --live-from-file)",
    )
    p.add_argument(
        "--device",
        default="default",
        help="Capture device index or name substring (with --live)",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List local PortAudio capture/playback devices and exit",
    )
    p.add_argument(
        "--live-from-file",
        type=Path,
        default=None,
        help="Feed a WAV through the live ring buffer (no second sound device)",
    )
    p.add_argument(
        "--live-mode",
        default="listen+decode",
        help="listen-only, decode-only, or listen+decode (default)",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=None,
        help="0-based input channel (default: mono / channel 0, or average for files)",
    )
    p.add_argument(
        "--output-device",
        default=None,
        help="Monitor playback device (default output; no-op if none)",
    )
    p.add_argument("--squelch-open-db", type=float, default=6.0, help="Open threshold above noise floor")
    p.add_argument("--squelch-close-db", type=float, default=3.0, help="Close threshold above noise floor")
    p.add_argument("--squelch-hang-ms", type=float, default=500.0, help="Squelch hang time in milliseconds")
    p.add_argument(
        "--squelch-policy",
        default="gate-off",
        choices=("gate-off", "always-on"),
        help="gate-off (default) or always-on (weak FT8). Monitor always follows squelch.",
    )
    p.add_argument("--record", type=Path, default=None, help="Write pre-squelch WAV to this path")
    p.add_argument("--mute", action="store_true", help="Panic-mute monitor at start")
    p.add_argument("--blocksize", type=int, default=480, help="Capture/monitor frames (480 = 10 ms at 48 kHz)")
    p.add_argument("--sample-rate", type=int, default=48000, help="Live session rate (file input is resampled)")
    p.add_argument("--seconds", type=float, default=None, help="Stop live capture after N seconds")
    p.add_argument(
        "--live-fast",
        action="store_true",
        help="With --live-from-file, do not pace the feed in real time",
    )
    p.add_argument(
        "--ft8-align",
        choices=("auto", "utc", "stream"),
        default="auto",
        help="FT8 period alignment. auto: UTC 15 s slots for a sound device, "
        "stream start for --live-from-file",
    )
    p.add_argument("-V", "--version", action="version", version=f"digimode {__version__}")
    return p


def _print_devices() -> int:
    from digimode.live import list_devices, list_output_devices

    inputs = list_devices()
    outputs = list_output_devices()
    if not inputs and not outputs:
        print("No PortAudio devices found (monitor will no-op; use --live-from-file).")
        return 0
    print("Inputs:")
    for dev in inputs:
        print(f"  [{dev.id}] {dev.name}  ch={dev.max_input_channels}  {dev.default_sample_rate} Hz")
    if not inputs:
        print("  (none)")
    print("Outputs:")
    for dev in outputs:
        print(f"  [{dev.id}] {dev.name}  {dev.default_sample_rate} Hz")
    if not outputs:
        print("  (none)")
    return 0


def _run_live(args) -> int:
    from digimode.live import LiveUnavailableError, format_status, live_listen_record

    if args.live_from_file is not None and not Path(args.live_from_file).is_file():
        print(f"error: live input not found: {args.live_from_file}", file=sys.stderr)
        return 2
    try:
        session = live_listen_record(
            device=args.device,
            mode=args.live_mode,
            record_path=args.record,
            force_mode=args.mode,
            squelch_open_db=args.squelch_open_db,
            squelch_close_db=args.squelch_close_db,
            squelch_hang_ms=args.squelch_hang_ms,
            squelch_policy=args.squelch_policy,
            from_file=args.live_from_file,
            channel=args.channel,
            mute=args.mute,
            sample_rate=args.sample_rate,
            blocksize=args.blocksize,
            pace_realtime=not args.live_fast,
            output_device=args.output_device,
            max_seconds=args.seconds,
            jt9_path=args.jt9,
            ft8_align=args.ft8_align,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Live: single capture → ring → squelched monitor + pre-squelch WAV + decode windows")
    try:
        session.start()
    except LiveUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    shown_decode = 0
    try:
        while True:
            st = session.status()
            print(format_status(st), flush=True)
            if len(st.decode_lines) > shown_decode:
                for line in st.decode_lines[shown_decode:]:
                    print(f"decoded: {line}", flush=True)
                shown_decode = len(st.decode_lines)
            if session.wait(0.5):
                break
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        session.stop()
    st = session.status()
    print(format_status(st), flush=True)
    if len(st.decode_lines) > shown_decode:
        for line in st.decode_lines[shown_decode:]:
            print(f"decoded: {line}", flush=True)
    if st.monitor_backend == "noop":
        print(f"monitor: noop ({st.monitor_detail})")
    else:
        print(f"monitor: {st.monitor_backend} ({st.monitor_detail})")
    if st.record_path and st.wav_frames:
        print(f"recorded: {st.record_path} ({st.wav_frames} frames, pre-squelch)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        return _print_devices()

    if args.live or args.live_from_file is not None:
        return _run_live(args)

    wav_path = args.wav
    if wav_path is None:
        print("error: WAV path required (or pass --live / --live-from-file)", file=sys.stderr)
        return 2
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        print(f"error: file not found: {wav_path}", file=sys.stderr)
        return 2

    try:
        audio = load_wav(wav_path)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: failed to load WAV: {exc}", file=sys.stderr)
        return 2

    if args.mode == "auto":
        classification = classify(audio)
        mode = classification.mode
        confidence = classification.confidence
        scores = classification.scores
        notes = classification.notes
    else:
        from digimode.classify import Classification

        classification = Classification(
            mode=args.mode,
            confidence=1.0,
            scores={args.mode: 1.0},
            notes=["forced by --mode"],
        )
        mode = args.mode
        confidence = 1.0
        scores = classification.scores
        notes = classification.notes

    decode_result = None
    if args.do_decode and mode == "FT8":
        if args.jt9:
            decoder = Ft8Jt9Decoder(jt9_path=args.jt9)
        else:
            decoder = get_ft8_decoder()
        decode_result = decoder.decode(audio)
    elif args.do_decode and mode in ("RTTY", "PSK31"):
        decode_result = StubDecoder(
            mode=mode,
            reason=f"{mode} decode not implemented in M1 (classifier only)",
        ).decode(audio)

    payload = {
        "file": str(wav_path),
        "duration_s": round(audio.duration_s, 3),
        "sample_rate": audio.sample_rate,
        "detected_mode": mode,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "notes": notes,
        "decode": None if decode_result is None else decode_result.as_dict(),
        "jt9_available": find_jt9() is not None,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"file:       {wav_path}")
    print(f"duration:   {audio.duration_s:.2f}s @ {audio.sample_rate} Hz")
    print(f"mode:       {mode}")
    print(f"confidence: {confidence:.3f}")
    print(f"scores:     " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    if notes:
        print("notes:")
        for n in notes:
            print(f"  - {n}")
    if decode_result is not None:
        print(f"decoder:    {decode_result.backend or '(none)'}")
        if decode_result.ok:
            print("decoded:")
            for line in decode_result.lines:
                print(f"  {line}")
        else:
            print(f"decode:     (no text) {decode_result.error or ''}")
            if decode_result.lines:
                print("partial lines:")
                for line in decode_result.lines:
                    print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
