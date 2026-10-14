# digi-mode-id

Offline digital ham-radio **mode identifier** + **FT8 decoder**, with a **local live audio tee**.

- **No network** during classify, decode, or live capture.
- **File path:** WAV → classify FT8/RTTY/PSK31 → FT8 text via WSJT-X `jt9`.
- **Live path:** one capture (or a file fed into that same ingest) → ring buffer → squelched monitor + pre-squelch WAV + the same classify/`jt9` pipeline.

## What works

| Capability | Status |
|------------|--------|
| WAV ingest (mono float, resample, optional channel) | Yes |
| Classify FT8 / RTTY / PSK31 (heuristics) | Yes |
| Force mode with `--mode` | Yes |
| Decode FT8 via WSJT-X `jt9 -8` | Yes (when `wsjtx` installed) |
| Live listen + pre-squelch WAV + decode tee | Yes (`sounddevice` / PortAudio, or `--live-from-file`) |
| Squelch (open/close/hang, `gate-off` / `always-on`) | Yes |
| RTTY / PSK31 decode | Stub only (classify works) |
| JS8 / Olivia / digi-voice | Out of scope |
| Network streams | Never |

## Dependencies

**Apt (recommended):**

```bash
sudo apt-get install -y python3-numpy python3-scipy wsjtx libportaudio2
python3 -m pip install --user --break-system-packages sounddevice
```

- `numpy` / `scipy` — ingest + spectrogram fingerprints
- `wsjtx` — `/usr/bin/jt9` (FT8 decode), `/usr/bin/ft8sim` (sample generation)
- `libportaudio2` + `sounddevice` — live capture and monitor playback

Optional: `DIGIMODE_JT9=/path/to/jt9` or `--jt9 /path/to/jt9`.

If this environment's apt mirror is broken, `libportaudio.so.2` still has to be on the dynamic linker path or live device open fails. `--live-from-file` does not need a sound card; monitor playback no-ops when there is no output device.

## How to run (file)

From the repo root:

```bash
python3 -m digimode samples/ft8_cq_k1abc.wav
python3 -m digimode samples/rtty_like.wav
python3 -m digimode samples/psk31_like.wav
python3 -m digimode samples/ft8_cq_k1abc.wav --json
python3 -m digimode path/to/recording.wav --mode FT8   # force FT8 decode
```

Example (synthetic FT8 from `ft8sim`):

```
mode:       FT8
confidence: 0.65+
decoder:    jt9:/usr/bin/jt9
decoded:
  CQ K1ABC FN42
```

## Live listen + record + decode

```bash
python3 -m digimode --list-devices

# real input (ALSA/PortAudio device index or name substring)
python3 -m digimode --live --device default --live-mode listen+decode \
  --squelch-policy always-on --record /tmp/digimode-live.wav
# device FT8 windows lock to UTC :00/:15/:30/:45 (--ft8-align stream to disable)

# no radio: push a WAV through the same ring buffer
python3 -m digimode --live \
  --live-from-file samples/ft8_cq_k1abc.wav \
  --live-mode listen+decode \
  --squelch-policy always-on \
  --squelch-open-db 6 --squelch-close-db 3 --squelch-hang-ms 500 \
  --record /tmp/digimode-live-ft8.wav \
  --live-fast
```

`--live-fast` disables real-time pacing of `--live-from-file` (useful for a scripted demo). Omit it to feed at 1×. The weak `ft8_cq_k1abc.wav` sample does **not** open a +6 dB squelch; `--squelch-policy always-on` is what decodes it. Expect `decoded: CQ K1ABC FN42`.

```bash
python3 tests/live_from_file_demo.py
```

Design notes: [`docs/live-audio-spec.md`](docs/live-audio-spec.md). What was verified on this box: [`RESULT.md`](RESULT.md).

Summary:

1. **Input:** local device or virtual cable, or `--live-from-file`. Mono PCM, session rate 48 kHz (file is resampled with the WAV ingest helper). Optional `--channel`.
2. **Squelch:** open ~6 dB above a tracked noise floor, close ~3 dB, hang 500 ms. Closed mutes the monitor. Decode policy `gate-off` (default) or `always-on`.
3. **Modes:** `--live-mode listen-only|decode-only|listen+decode`. Monitor mute is immediate (`--mute`, or switching to decode-only). Decode policy/mode/force-mode apply on the next analysis window. `--mode` is the force-mode override (`auto|FT8|RTTY|PSK31`).
4. **Tee:** single capture callback (or one file feeder) fills one ring. Monitor, WAV writer, and decode each read that ring. WAV is pre-squelch. No second input open.
5. **Latency:** default block 480 frames → about 20 ms of software queue (target under ~80 ms). FT8 window is 15 s and may lag one period. Classify about every 0.75 s off the capture thread.
6. **Status line:** operating mode, squelch open/closed, floor and level, record path, last class+confidence, last decode line, device, monitor backend.

```python
from digimode.live import live_listen_record
session = live_listen_record(
    from_file="samples/ft8_cq_k1abc.wav",
    mode="listen+decode",
    squelch_policy="always-on",
    record_path="/tmp/digimode-live.wav",
    pace_realtime=False,
)
session.start()
session.wait()
session.stop()
print(session.status().last_decode_line)
```

## Samples

| File | Purpose |
|------|---------|
| `samples/ft8_cq_k1abc.wav` | Real FT8 from `ft8sim` (**decodable**) |
| `samples/rtty_like.wav` | Synthetic 170 Hz FSK — classifier smoke |
| `samples/psk31_like.wav` | Synthetic narrow PSK — classifier smoke |
| `samples/noise.wav` | Noise → UNKNOWN / low confidence |

```bash
cd samples
ft8sim "CQ K1ABC FN42" 1500.0 0.0 0.0 1.0 1 -8
mv 000000_000001.wav my_ft8.wav
```

## Architecture

```
digimode/
  ingest.py           # WAV → AudioBuffer (mono float32, resample)
  classify.py         # STFT fingerprints → FT8 / RTTY / PSK31 / UNKNOWN
  live.py             # ring → monitor + pre-squelch WAV + decode windows
  decode/
    base.py           # Decoder adapter interface
    ft8_jt9.py        # Real FT8 via jt9 -8
    stub.py           # Clear failure when backend missing
  cli.py              # python -m digimode
docs/
  live-audio-spec.md
```

**Classifier cues:** FT8 ~6.25 Hz spacing + ~50 Hz cluster + 15 s activity; RTTY ~170 Hz FSK pair; PSK31 narrow single carrier. Heuristic — not ML.

**FT8 decode:** `Ft8Jt9Decoder` writes 12 kHz 16-bit WAV, runs `jt9 -8 -d 3`. Live windows call that same adapter.

## Known limits

- Classifier is fingerprint-based; crowded bands need better features later.
- Only FT8 text decode; RTTY/PSK31 = classify + stub.
- Live device FT8 periods follow UTC 15 s slots. File playback follows the stream clock unless `--ft8-align utc`.
- This box has no ALSA capture/playback device, so monitor output no-ops until one exists.
- No SDR IQ. No JS8Call.

## Offline policy

Install apt/pip packages once at setup. Runtime uses local files, local `jt9`, and local sound devices only.
