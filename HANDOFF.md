# DigiMode ID — Cursor handoff

Snapshot of the DigiMode ID project as of 2026-10-13. Imported into Cursor from the Grok handoff zip. Project lives at the repo root; start with this file.

## Goal

A local-only tool that takes radio audio, identifies the digital ham mode, and decodes it to readable text. It uses no network at runtime. Inputs are WAV/IQ files and live audio. The operator can listen live while the same stream is recorded to WAV and decoded.

Target modes:
- **FT8 / FT4** via WSJT-X decoder (`jt9`)
- **fldigi family** (PSK31, RTTY, Olivia, MFSK and kin)
- **JS8** via JS8Call or its decoder path (later stage)
- **DMR** — identify and decode clear traffic; **flag when encrypted** (privacy/ALG bits). It must not attempt decryption.
- **D-STAR, YSF**, and NXDN later
- **VARA** — detect only (proprietary)

## Where each piece stands

| Piece | State |
|---|---|
| WAV ingest + normalizer (`digimode/ingest.py`) | Done. Mono float, resample; shared by file and live paths |
| Classifier (`digimode/classify.py`) | Done (heuristic fingerprints). FT8 / RTTY / PSK31 / UNKNOWN pass smoke tests |
| FT8 decode (`digimode/decode/ft8_jt9.py`) | Done. Real offline decode via `/usr/bin/jt9 -8` (apt `wsjtx`); sample → `CQ K1ABC FN42` |
| RTTY / PSK31 decode | Stub only (`decode/stub.py`) |
| Live path (`digimode/live.py`) | Done. One capture → float32 ring buffer → 3 readers: squelched monitor, pre-squelch WAV writer, decode worker. Device opened once |
| Squelch | Done. Noise floor tracking; open +6 dB, close +3 dB, hang 500 ms (tunable); policies `gate-off` (default) / `always-on` |
| Operating modes | Done. `listen-only`, `decode-only`, `listen+decode`; monitor changes immediately, decode at next window |
| Recording | Done. `--record path.wav`, 16-bit pre-squelch, verified 720000 frames at 48 kHz |
| Real sound device (`--device`) | Implemented, **untested on hardware**. RTO's computer has no audio devices; the monitor no-ops there |
| FT8 UTC slot alignment | Done for `--device` (and `--ft8-align utc`). Drops the partial leading slot, then cuts on Unix 15 s boundaries. `--live-from-file` stays on the stream clock unless asked |
| JS8, Olivia, MFSK, digital voice, IQ ingest, hotkeys | Not started |
| Communications log | Specified for the Android app, not built |
| Android app | Strategy only, no code yet |

Details: `RESULT.md`, `README.md`, `docs/live-audio-spec.md`.

## How to run (Linux)

```bash
cd /path/to/digi-mode-id  # repo root
pip install -r requirements.txt        # numpy/scipy/sounddevice
sudo apt install wsjtx libportaudio2   # jt9 + PortAudio
python3 -m digimode samples/ft8_cq_k1abc.wav
python3 -m digimode --live --live-from-file samples/ft8_cq_k1abc.wav \
  --live-mode listen+decode --squelch-policy always-on \
  --record /tmp/live.wav --live-fast
python3 -m digimode --list-devices
python3 tests/smoke_test.py && python3 tests/live_from_file_demo.py
```

Note: `--mode` is the force-mode override (`auto|FT8|RTTY|PSK31`); `--live-mode` is listen/decode.

## Live audio spec (summary)

1. Input: local device or virtual cable, mono PCM 48 kHz, 16-bit or float32, operator picks device/channel.
2. Squelch: open ~6 dB / close ~3 dB above floor, 500 ms hang; closed mutes monitor. `gate-off` gates decode; `always-on` for weak FT8.
3. Modes: listen-only / decode-only / listen+decode (default).
4. Single ring buffer, three consumers, no second device open; WAV is pre-squelch.
5. Latency: monitor < 80 ms; FT8 may lag one 15 s period; PSK/RTTY < 2 s to first text; classify every 0.5–1 s off the audio thread.
6. Controls: device, mode, squelch, record on/off + path, force-mode, panic mute. Status line: mode, squelch, record path, last class + confidence, last decode.

## Android strategy

- **Primary deployment:** a disconnected GrapheneOS phone or tablet listening to an HT or mobile rig's audio (speaker/data jack via TRRS cable or USB-C sound card). No network, no Google Play Services.
- **Architecture:** Android is a thin capture/UI shell. AudioRecord → ring buffer (same tee as Linux: monitor + WAV log + decode) → classifier → mode-specific native engines.
- **andfldigi** as the decoder engine for the fldigi family (PSK, RTTY, Olivia, MFSK). It does **not** cover FT8, JS8, or digital voice.
- **FT8** via a separate FOSS WSJT-style engine (JNI port of the `jt9`/ft8 decoder). Don't build FT8 into andfldigi.
- **DMR / D-STAR / YSF** via FOSS digital-voice libraries (digiham / DSD-family / OP25-class). DMR shows an "encrypted" notice from header bits.
- **Communications log** (optional, off by default): timestamp, frequency (operator-entered; CAT later), mode, decoded text, encrypted flag. Stored on-device as SQLite with CSV export.
- **Suggested build order:** AudioRecord + WAV log → classifier → FT8 → andfldigi family → DMR (clear + encrypted flag) → D-STAR/YSF → JS8.
- **License note:** fldigi/andfldigi and WSJT-X are GPL. Check obligations before distribution.

## Hardware and firmware references

- **Decoders:** WSJT-X `jt9` (FT8/FT4), fldigi/andfldigi, JS8Call, digiham/DSD/OP25 for digital voice, multimon-ng/direwolf where useful. inspectrum and Universal Radio Hacker are for lab/fingerprinting only.
- **Linux deps used:** Python 3, numpy/scipy, `sounddevice` 0.5.6, PortAudio (`libportaudio2` 19.7.0), apt `wsjtx`.
- **Operator radios (audio sources):** Xiegu G106, Xiegu G90, Elecraft KX2, Icom IC-7100, plus 2m/70cm HTs and mobiles.
- **Audio interface:** a radio speaker or data/discriminator output into a USB sound card or TRRS adapter. Discriminator/flat audio is better for 4FSK digital voice.
- **Android hardware:** GrapheneOS-supported Pixel (sealed battery), or a Fairphone with LineageOS/CalyxOS if a removable battery matters.

## Suggested next tasks for Cursor

1. Test `--device` with real hardware or `snd-aloop`.
2. Replace RTTY/PSK31 stubs (fldigi engine or native demod).
3. Add the DMR branch with encrypted-flag detection.
4. Start the Android shell: AudioRecord tee + JNI FT8.
