## Cursor Cloud specific instructions

DigiMode ID is one local Python CLI (`python3 -m digimode`). It has no dev server, database, port, or network calls. Commands for file decode, live audio, and the smoke tests are in `README.md`.

- There is no lint, format, or typecheck config. `python3 tests/smoke_test.py` is the check that covers classify, FT8 `jt9` decode, and the live-from-file path.
- This VM has no ALSA capture or playback device. `python3 -m digimode --list-devices` is empty, `--device` raises `LiveUnavailableError`, and the monitor backend is `noop`. Exercise live audio with `--live-from-file`. The checked-in `samples/ft8_cq_k1abc.wav` sits about 1–2 dB above its own noise floor, so the default +6 dB squelch stays closed; pass `--squelch-policy always-on` (and `--live-fast` to skip 1× pacing).
- FT8 text decode shells out to `/usr/bin/jt9` from the `wsjtx` apt package (`DIGIMODE_JT9` or `--jt9` overrides the path). Classify-only still runs if `jt9` is missing; `tests/smoke_test.py` does not.
- `sounddevice` needs `libportaudio.so.2` (`libportaudio2`) on the loader path. `--live-from-file` does not need a sound card; monitor playback no-ops when PortAudio reports no output device.
