# Live audio path — specification

Status: **implemented** in `digimode/live.py` and `python3 -m digimode --live`.
File WAV → classify → FT8 text is unchanged. Live capture uses the same normalizer
and `jt9` decoder after a local PCM tee. **No network** at runtime.

Headless machines with no PortAudio device can pass `--live-from-file sample.wav`
through the same ring. If playback cannot be opened, the monitor still applies
squelch and then no-ops.

## 1. Input

| Item | Requirement |
|------|-------------|
| Source | Local sound device or virtual cable only (ALSA/`sounddevice`/PortAudio). No network streams. |
| Format | Mono PCM, **48 kHz**, 16-bit integer **or** float32 |
| Device | Named / selectable device string (e.g. `hw:1,0`, Pulse/`default`, virtual cable name) |
| Normalize | Same path as file ingest: → mono float32 → resample as needed (12 kHz for FT8/`jt9`, etc.) |

IQ file ingest remains a later option; live M1 is audio PCM only.

## 2. Squelch

| Parameter | Default | Notes |
|-----------|---------|-------|
| Open threshold | ~**6 dB** above noise floor | Opens monitor (and gate-off decode) |
| Close threshold | ~**3 dB** above noise floor | Hysteresis vs open |
| Hang time | **500 ms** | Tunable; keeps open after drop below close |
| Closed behavior | **Mute monitor** | No speaker/headphone output while closed |

**Decode policies** (independent of monitor mute):

- **`gate-off` (default for voice-like):** decode/classify windows only while squelch open (hang applies). Saves CPU; may miss weak tails.
- **`always-on` (preferred for weak FT8):** classify/decode keep running on the ring buffer regardless of squelch. Monitor still follows squelch.

Hang applies to **monitor** and to **gate-off** decode. It does not delay `always-on` decode.

Noise floor: rolling estimate on quiet frames (e.g. percentile of recent block RMS/dBFS).

## 3. Operating modes

| Mode | Monitor | WAV record | Classify/decode |
|------|---------|------------|-----------------|
| **listen-only** | Yes (squelched) | Optional | Off |
| **decode-only** | Muted | Optional | On |
| **listen+decode** (default) | Yes (squelched) | Optional | On |

- **Monitor mute** takes effect **immediately**.
- **Decode mode switches** (listen-only ↔ decode, force-mode, gate policy) apply at the **next analysis window** boundary — never tear down the capture thread mid-block.

## 4. Architecture (single capture → tee)

```
  [local sound device]  (opened once)
           │
           ▼
      capture thread
           │
           ▼
      ring buffer (PCM)
           │
     ┌─────┼──────────────┐
     ▼     ▼              ▼
 monitor  WAV writer   decode/classify
 (squelch) (pre-squelch)  windows
```

**Rules:**

1. **Single open** of the input device — no double-open for monitor vs record vs decode.
2. **Ring buffer** fans out to three consumers:
   - **Monitor** — low-latency playback; applies squelch (mute when closed).
   - **WAV writer** — **pre-squelch faithful copy** of captured PCM (what was heard at the device), optional on/off + path.
   - **Decode/classify** — pulls analysis windows from the same ring; feeds the **same** `ingest → classify → decode` pipeline as file WAV.
3. WAV/IQ **file** path remains first-class and unchanged.

## 5. Latency budgets

| Path | Target |
|------|--------|
| Monitor (capture → ear) | **&lt; ~80 ms** |
| FT8 decode | Period-aligned (~**15 s** slots); do not block monitor |
| PSK/RTTY first text | **&lt; ~2 s** after squelch open / signal present (when decoder exists) |
| Classify refresh | ~**0.5–1 s** |
| Priority | **Never block the monitor/capture thread** on classify or `jt9` |

Heavy work runs on worker thread(s) or subprocess (`jt9`); results posted to a status queue.

## 6. Controls & status

**Controls (CLI / `LiveSession` API):**

| Control | Purpose |
|---------|---------|
| Device | Select capture device |
| Mode | `listen` / `decode` / `both` |
| Squelch | Open/close dB, hang ms, policy `gate-off` \| `always-on` |
| Record | On/off + output path |
| Force-mode | `auto` / `FT8` / `RTTY` / `PSK31` (same as CLI `--mode`) |
| Mute | Force monitor mute |

**Future hotkeys (not required in M1):** `M` mute, `S` squelch toggle/cal, `R` record, `Space` mode cycle.

**Status line / struct:**

- operating mode (`listen-only` / `decode-only` / `listen+decode`)
- squelch state (`open` / `closed`) + floor / level
- record on/off + path
- last classification + confidence
- last decode text line
- device name, underrun/overrun counters (debug)

## 7. Implementation

| Piece | Status |
|-------|--------|
| Spec (this doc) | Yes |
| `LiveSession` / `live_listen_record` | PortAudio via `sounddevice`, plus `--live-from-file` |
| CLI `--live` | Device, live mode, squelch, record, mute, force-mode |
| File WAV pipeline | Unchanged, shared |
| JS8 via JS8Call | Out of scope |

```text
digimode.live.list_devices() -> list[DeviceInfo]
digimode.live.live_listen_record(...)
digimode.live.LiveSession.start() / .stop() / .wait() / .status()
digimode.live.LiveSession.set_mode() / .set_mute() / .set_squelch()
```

```bash
python3 -m digimode --list-devices
python3 -m digimode --live --device default --live-mode listen+decode --record captures/live.wav
python3 -m digimode --live --live-from-file samples/ft8_cq_k1abc.wav --squelch-policy always-on --record /tmp/live.wav
```

`--mode` remains the force-mode override (`auto` / `FT8` / `RTTY` / `PSK31`).
Operating mode is `--live-mode` (`listen-only` / `decode-only` / `listen+decode`).

Block size defaults to 480 frames at 48 kHz (10 ms). The monitor software queue
is two blocks (~20 ms), inside the 80 ms budget. Classify/decode never runs
on the capture callback.

FT8 windows are 15 s of captured audio measured from the start of the stream
(not UTC slot aligned). A real on-air decode may lag one period.

## 8. Offline / safety

- Capture and playback are **local only**.
- No network sockets in the live or decode path.
- Optional WAV on disk is local filesystem only.
