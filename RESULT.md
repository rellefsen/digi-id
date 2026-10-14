# Live audio — RESULT

## What works

- **CLI file path (unchanged):** `python3 -m digimode samples/ft8_cq_k1abc.wav` → mode FT8, text `CQ K1ABC FN42`.
- **Live tee:** one producer writes a float32 ring; three consumers read it. The input is opened at most once (`capture_opens == 1`). No network sockets.
  1. Squelched monitor (immediate mute / decode-only mute / panic `--mute`).
  2. Pre-squelch 16-bit WAV when `--record` is set.
  3. Classify about every 0.75 s and FT8 `jt9` on 15 s windows, on a worker thread. Same `load_wav` / `AudioBuffer.resample` normalizer and `Ft8Jt9Decoder` as the file path.
- **Squelch:** noise floor (fast fall, slow rise), open +6 dB, close +3 dB, hang 500 ms, all tunable. `gate-off` (default) skips classify/decode while closed; `always-on` decodes anyway. Hang is part of the open state, so it applies to the monitor and to gate-off.
- **Modes:** `listen-only`, `decode-only`, `listen+decode`. Monitor mute/unmute is immediate. Decode mode, force-mode, and squelch policy latch at the next analysis window.
- **Virtual input:** `--live-from-file` resamples into the session rate (48 kHz) and calls the same ingest function a PortAudio callback uses. `--device` is the real ALSA/PortAudio path (`sounddevice`, block size 480 ≈ 10 ms, software queue ≈ 20 ms, under the 80 ms target).
- **Headless box:** PortAudio reports no devices. Monitor still runs and counts frames, but playback is a no-op (`monitor=noop`). That was verified, not simulated away.

## Demo command

From the repo root:

```bash
python3 -m digimode --live \
  --live-from-file samples/ft8_cq_k1abc.wav \
  --live-mode listen+decode \
  --squelch-policy always-on \
  --squelch-open-db 6 --squelch-close-db 3 --squelch-hang-ms 500 \
  --record /tmp/digimode-live-ft8.wav \
  --live-fast
```

`--live-fast` only skips real-time pacing of the file feeder. Omit it to play the file through the ring at 1× (about 15 s plus `jt9`).

Checked result: status `class=FT8:0.66`, line `decoded: CQ K1ABC FN42`, recording 720000 frames at 48 kHz (pre-squelch, not silence). The sample sits about 1–2 dB above its own floor, so squelch stays **closed** and the monitor is muted; `always-on` is what lets weak FT8 decode. `gate-off` on the same file skips `jt9` (`decode_skipped_gate`).

Scripted check (squelch/hang, tone monitor vs WAV, gate-off, and the CLI command):

```bash
python3 tests/live_from_file_demo.py
python3 tests/smoke_test.py
```

Real device (when one exists):

```bash
python3 -m digimode --list-devices
python3 -m digimode --live --device default --live-mode listen+decode \
  --squelch-policy always-on --record /tmp/digimode-live.wav
```

`--mode` is still the force-mode override (`auto|FT8|RTTY|PSK31`), not the listen/decode operating mode. Use `--live-mode` for that.

## What it took

- `sounddevice` 0.5.6 installed with `pip --break-system-packages` (Debian PEP 668).
- `libportaudio.so.2` from Debian `libportaudio2` 19.7.0-1+b1 extracted to `/usr/local/lib` and `ldconfig`. `apt-get` against the configured mirror returned 500 / unsigned `InRelease`, so the deb was fetched from deb.debian.org directly. `libjack` and `libasound` were already on the box.
- File feeder + ring + squelch history so a fast feed still gates the monitor per block (not with the final squelch state only).

## Blockers

- **No capture or playback device** on this box (`sounddevice.query_devices()` is empty: ALSA and OSS, zero devices). `--device` is implemented and raises `LiveUnavailableError` with a pointer at `--live-from-file` / `--list-devices`. It was not exercised against a radio, virtual cable, or `snd-aloop`.
- **Monitor cannot be heard here.** The playback `OutputStream` path is real and falls back to a no-op that still applies squelch zeros when open would have played.
- **FT8 windows are counted from stream start**, not aligned to UTC 15 s slots. A live on-air FT8 transmission can be split across two windows and missed until the next period (the allowed one-period lag). `--live-from-file` on a single 15 s `ft8sim` WAV matches one window and decodes.
- **Weak-signal squelch:** this FT8 sample will not open a +6 dB squelch. Use `--squelch-policy always-on` (or lower `--squelch-open-db`) to decode it. That matches the spec.
- JS8Call still out of scope. RTTY/PSK31 decode is still the stub. Classifier is unchanged heuristics.

## Left for later

- UTC-aligned FT8 periods for `--device`.
- Hotkeys. Olivia / JS8 / digi-voice. IQ ingest.
