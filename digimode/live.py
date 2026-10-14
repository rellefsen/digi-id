"""Live listen + record + decode tee.

Single capture producer (PortAudio input callback, or a file feeder that
calls the same ingest function) writes mono float32 into one ring buffer.
Three consumers read that ring and never open the input device again:

  * squelched monitor playback (immediate mute)
  * optional pre-squelch WAV writer
  * classify / FT8 windows on a worker thread (same ingest + jt9 path as files)

No network sockets. See docs/live-audio-spec.md.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import wave
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from digimode.classify import classify
from digimode.decode import get_ft8_decoder
from digimode.decode.ft8_jt9 import Ft8Jt9Decoder
from digimode.decode.stub import StubDecoder
from digimode.ingest import AudioBuffer, load_wav


class LiveMode(str, Enum):
    LISTEN = "listen"  # monitor ± WAV, no classify/decode
    DECODE = "decode"  # pipeline on, monitor muted
    BOTH = "both"  # default: listen + decode


class SquelchPolicy(str, Enum):
    GATE_OFF = "gate-off"  # default: classify/decode only while squelch is open
    ALWAYS_ON = "always-on"  # weak FT8: decode regardless of squelch


MODE_LABEL = {
    LiveMode.LISTEN: "listen-only",
    LiveMode.DECODE: "decode-only",
    LiveMode.BOTH: "listen+decode",
}


def parse_live_mode(mode: str) -> LiveMode:
    key = mode.strip().lower().replace("_", "-")
    mapping = {
        "listen": LiveMode.LISTEN,
        "listen-only": LiveMode.LISTEN,
        "decode": LiveMode.DECODE,
        "decode-only": LiveMode.DECODE,
        "both": LiveMode.BOTH,
        "listen+decode": LiveMode.BOTH,
        "listen-decode": LiveMode.BOTH,
    }
    if key not in mapping:
        raise ValueError(
            f"unknown live mode {mode!r}; expected listen-only, decode-only, or listen+decode"
        )
    return mapping[key]


def parse_squelch_policy(policy: str) -> SquelchPolicy:
    key = policy.strip().lower().replace("_", "-")
    mapping = {
        "gate-off": SquelchPolicy.GATE_OFF,
        "gateoff": SquelchPolicy.GATE_OFF,
        "always-on": SquelchPolicy.ALWAYS_ON,
        "alwayson": SquelchPolicy.ALWAYS_ON,
    }
    if key not in mapping:
        raise ValueError(f"unknown squelch policy {policy!r}; expected gate-off or always-on")
    return mapping[key]


def decode_window_allowed(mode: LiveMode, policy: SquelchPolicy, saw_open: bool) -> bool:
    """Whether a finished analysis window should be classified/decoded.

    Hang is already folded into ``saw_open`` by the squelch state machine.
    """
    if mode == LiveMode.LISTEN:
        return False
    if policy == SquelchPolicy.GATE_OFF and not saw_open:
        return False
    return True


@dataclass
class DeviceInfo:
    id: str
    name: str
    max_input_channels: int = 1
    default_sample_rate: int = 48000


@dataclass
class SquelchConfig:
    open_db: float = 6.0  # above noise floor
    close_db: float = 3.0
    hang_ms: float = 500.0
    policy: SquelchPolicy = SquelchPolicy.GATE_OFF


@dataclass
class LiveStatus:
    mode: LiveMode = LiveMode.BOTH
    squelch_open: bool = False
    squelch_ever_open: bool = False
    recording: bool = False
    record_path: str | None = None
    device: str | None = None
    last_mode_label: str | None = None
    last_confidence: float | None = None
    last_decode_line: str | None = None
    decode_lines: tuple[str, ...] = ()
    message: str = "live capture not started"
    noise_floor_db: float | None = None
    level_db: float | None = None
    monitor_backend: str = "none"
    monitor_detail: str = ""
    monitor_latency_ms: float = 0.0
    monitor_frames: int = 0
    monitor_audible_frames: int = 0
    wav_frames: int = 0
    analysis_frames: int = 0
    captured_frames: int = 0
    overruns: int = 0
    capture_opens: int = 0
    decode_attempts: int = 0
    decode_skipped_gate: int = 0
    input_backend: str = ""


@dataclass
class LiveConfig:
    device: str = "default"
    mode: LiveMode = LiveMode.BOTH
    squelch: SquelchConfig = field(default_factory=SquelchConfig)
    record_path: Path | None = None
    force_mode: str = "auto"  # auto|FT8|RTTY|PSK31
    sample_rate: int = 48000
    mute: bool = False
    from_file: Path | None = None
    channel: int | None = None
    output_device: str | None = None
    blocksize: int = 480  # 10 ms at 48 kHz; monitor budget is two blocks
    pace_realtime: bool = True
    classify_interval_s: float = 0.75
    ft8_window_s: float = 15.0
    max_seconds: float | None = None
    jt9_path: str | None = None
    ring_seconds: float = 60.0


class LiveUnavailableError(RuntimeError):
    """Raised when a real capture device cannot be opened."""


def _import_sounddevice():
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001 — missing lib or package
        raise LiveUnavailableError(
            "sounddevice/PortAudio is not available for live device capture. "
            "Install python3 package 'sounddevice' and libportaudio, "
            "or use --live-from-file for a virtual input. "
            f"({exc})"
        ) from exc
    return sd


def _query_devices(sd) -> list[dict]:
    devices = sd.query_devices()
    if isinstance(devices, dict):
        return [devices]
    return list(devices)


def list_devices() -> list[DeviceInfo]:
    """List capture devices. Empty if PortAudio is missing or has no inputs."""
    try:
        sd = _import_sounddevice()
    except LiveUnavailableError:
        return []
    out: list[DeviceInfo] = []
    for i, dev in enumerate(_query_devices(sd)):
        max_in = int(dev.get("max_input_channels") or 0)
        if max_in <= 0:
            continue
        rate = dev.get("default_samplerate") or 48000
        out.append(
            DeviceInfo(
                id=str(i),
                name=str(dev.get("name") or f"device {i}"),
                max_input_channels=max_in,
                default_sample_rate=int(rate),
            )
        )
    return out


def list_output_devices() -> list[DeviceInfo]:
    try:
        sd = _import_sounddevice()
    except LiveUnavailableError:
        return []
    out: list[DeviceInfo] = []
    for i, dev in enumerate(_query_devices(sd)):
        max_out = int(dev.get("max_output_channels") or 0)
        if max_out <= 0:
            continue
        rate = dev.get("default_samplerate") or 48000
        out.append(
            DeviceInfo(
                id=str(i),
                name=str(dev.get("name") or f"device {i}"),
                max_input_channels=0,
                default_sample_rate=int(rate),
            )
        )
    return out


def resolve_input_device(spec: str):
    """Return a PortAudio input device index, or raise LiveUnavailableError."""
    sd = _import_sounddevice()
    devices = _query_devices(sd)
    if spec in (None, "", "default"):
        try:
            idx = int(sd.default.device[0])
        except Exception as exc:  # noqa: BLE001
            raise LiveUnavailableError(
                "No default input device. Pass --device, or --live-from-file "
                "to feed a WAV through the same ring buffer."
            ) from exc
        if idx < 0:
            raise LiveUnavailableError(
                "No default input device. Pass --device, or --live-from-file "
                "to feed a WAV through the same ring buffer."
            )
        return idx
    text = str(spec)
    if text.lstrip("-").isdigit():
        return int(text)
    spec_l = text.lower()
    matches = [
        i
        for i, dev in enumerate(devices)
        if int(dev.get("max_input_channels") or 0) > 0 and spec_l in str(dev.get("name") or "").lower()
    ]
    if not matches:
        names = [d.name for d in list_devices()] or ["(none)"]
        raise LiveUnavailableError(
            f"No input device matching {spec!r}. Inputs: {', '.join(names)}. "
            "Use --list-devices or --live-from-file."
        )
    return matches[0]


def resolve_output_device(spec: str | None):
    """Return an output device index, or None if playback should no-op."""
    try:
        sd = _import_sounddevice()
    except LiveUnavailableError:
        return None
    devices = _query_devices(sd)
    if spec not in (None, "", "default"):
        text = str(spec)
        if text.lstrip("-").isdigit():
            return int(text)
        spec_l = text.lower()
        for i, dev in enumerate(devices):
            if int(dev.get("max_output_channels") or 0) > 0 and spec_l in str(dev.get("name") or "").lower():
                return i
        return None
    try:
        idx = int(sd.default.device[1])
    except Exception:  # noqa: BLE001
        return None
    if idx < 0:
        return None
    return idx


class RingBuffer:
    """Mono float32 ring addressed by absolute frame counts.

    Independent readers each keep their own cursor. A reader that falls
    more than ``capacity`` behind skips forward (overrun) instead of
    blocking the capture callback.
    """

    def __init__(self, capacity: int):
        if capacity < 2:
            raise ValueError("ring capacity too small")
        self.cap = int(capacity)
        self.buf = np.zeros(self.cap, dtype=np.float32)
        self.write_abs = 0
        self.overruns = 0
        self._lock = threading.Lock()

    def write(self, samples: NDArray[np.floating]) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = int(x.size)
        if n <= 0:
            return
        if n > self.cap:
            raise ValueError("capture block larger than ring buffer")
        with self._lock:
            start = self.write_abs % self.cap
            first = min(n, self.cap - start)
            self.buf[start : start + first] = x[:first]
            if first < n:
                self.buf[: n - first] = x[first:]
            self.write_abs += n

    def read(self, read_abs: int, n: int) -> tuple[NDArray[np.float32], int, bool]:
        """Copy up to ``n`` frames. Returns (data, new_cursor, overrun)."""
        n = int(n)
        with self._lock:
            overrun = False
            if self.write_abs - read_abs > self.cap:
                overrun = True
                self.overruns += 1
                read_abs = self.write_abs - self.cap
            avail = self.write_abs - read_abs
            take = min(max(n, 0), max(avail, 0))
            if take <= 0:
                return np.zeros(0, dtype=np.float32), read_abs, overrun
            start = read_abs % self.cap
            first = min(take, self.cap - start)
            if first == take:
                out = self.buf[start : start + take].copy()
            else:
                out = np.empty(take, dtype=np.float32)
                out[:first] = self.buf[start:]
                out[first:] = self.buf[: take - first]
            return out, read_abs + take, overrun

    def available(self, read_abs: int) -> int:
        with self._lock:
            if self.write_abs - read_abs > self.cap:
                return self.cap
            return max(0, self.write_abs - read_abs)


class NoiseSquelch:
    """Noise-floor squelch with hysteresis and hang.

    Floor falls quickly and rises slowly so a transmission does not
    immediately become the new floor. Open / close thresholds are dB
    above that floor. Hang keeps the gate open after the level drops
    below the close threshold (monitor and gate-off decode).
    """

    def __init__(self, cfg: SquelchConfig, sample_rate: int):
        self.cfg = cfg
        self.sr = int(sample_rate)
        self.open = False
        self.ever_open = False
        self.level_db = -120.0
        self.floor_db = -90.0
        self._floor_init = False
        self._hang_left = 0
        self.frames = 0
        self.frames_while_open = 0
        # Run-length history: (exclusive_end_frame, was_open) so monitor
        # playback follows the decision made when those samples arrived,
        # even if a fast file feed finishes before playback starts.
        self._runs: list[tuple[int, bool]] = []
        self._lock = threading.Lock()

    def process(self, samples: NDArray[np.floating]) -> bool:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = int(x.size)
        if n <= 0:
            return self.open
        rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-20))
        db = float(20.0 * math.log10(rms))
        dt = n / float(self.sr)
        with self._lock:
            self.level_db = db
            if not self._floor_init:
                self.floor_db = db
                self._floor_init = True
            else:
                if db < self.floor_db:
                    tau = 0.05  # seconds — fall fast onto quiet audio
                else:
                    tau = 5.0  # seconds — rise slowly so signals stay above floor
                alpha = 1.0 - math.exp(-dt / tau)
                self.floor_db += alpha * (db - self.floor_db)
            open_db = float(self.cfg.open_db)
            close_db = min(float(self.cfg.close_db), open_db)
            hang_frames = max(0, int(round(float(self.cfg.hang_ms) * 1e-3 * self.sr)))
            open_th = self.floor_db + open_db
            close_th = self.floor_db + close_db
            if self.open:
                if db >= close_th:
                    self._hang_left = hang_frames
                else:
                    self._hang_left -= n
                    if self._hang_left <= 0:
                        self.open = False
                        self._hang_left = 0
            elif db >= open_th:
                self.open = True
                self.ever_open = True
                self._hang_left = hang_frames
            self.frames += n
            if self.open:
                self.frames_while_open += n
            if self._runs and self._runs[-1][1] is self.open:
                self._runs[-1] = (self.frames, self.open)
            else:
                self._runs.append((self.frames, self.open))
            if len(self._runs) > 200_000:
                del self._runs[:100_000]
            return self.open

    def is_open(self) -> bool:
        with self._lock:
            return self.open

    def snapshot(self) -> tuple[bool, bool, float, float, int]:
        with self._lock:
            floor = self.floor_db if self._floor_init else None
            return self.open, self.ever_open, floor, self.level_db, self.frames_while_open

    def open_at(self, frame: int) -> bool:
        """Squelch state for a previously captured frame index."""
        with self._lock:
            runs = self._runs
            if not runs:
                return False
            lo, hi = 0, len(runs)
            while lo < hi:
                mid = (lo + hi) // 2
                if runs[mid][0] <= frame:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= len(runs):
                return False
            return runs[lo][1]

    def any_open(self, start: int, end: int) -> bool:
        """True if squelch was open for any captured frame in [start, end)."""
        if end <= start:
            return False
        with self._lock:
            prev = 0
            for run_end, is_open in self._runs:
                lo = start if start > prev else prev
                hi = end if end < run_end else run_end
                if hi > lo and is_open:
                    return True
                prev = run_end
                if prev >= end:
                    break
            return False


class PcmWavWriter:
    """Incremental 16-bit mono WAV. Pre-squelch: caller passes raw capture."""

    def __init__(self, path: Path, sample_rate: int):
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._w = wave.open(str(self.path), "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(int(sample_rate))
        self.frames = 0
        self._lock = threading.Lock()
        self._closed = False

    def write(self, samples: NDArray[np.floating]) -> None:
        x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16)
        with self._lock:
            if self._closed:
                return
            self._w.writeframes(pcm.tobytes())
            self.frames += int(pcm.size)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._w.close()


class MonitorPlayer:
    """Squelch-aware playback. No-ops (but still counts) when no output exists."""

    def __init__(self, sample_rate: int, blocksize: int, output_device: str | None):
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        self.backend = "noop"
        self.detail = "not opened"
        self.stream = None
        self.latency_ms = 2000.0 * self.blocksize / float(self.sample_rate)
        self._open(output_device)

    def _open(self, output_device: str | None) -> None:
        idx = resolve_output_device(output_device)
        if idx is None:
            self.backend = "noop"
            self.detail = "no output device; monitor path muted locally"
            return
        try:
            sd = _import_sounddevice()
            stream = sd.OutputStream(
                device=idx,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.blocksize,
                latency="low",
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001 — playback is optional
            self.backend = "noop"
            self.detail = f"output open failed ({exc}); monitor path muted locally"
            self.stream = None
            return
        self.stream = stream
        self.backend = "sounddevice"
        self.detail = f"output device {idx}"
        try:
            reported = float(stream.latency) * 1000.0
            # capture block + reported output latency
            self.latency_ms = (1000.0 * self.blocksize / float(self.sample_rate)) + reported
        except Exception:  # noqa: BLE001
            pass

    def write(self, samples: NDArray[np.floating], audible: bool) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        out = x if audible else np.zeros_like(x)
        stream = self.stream
        if stream is None:
            return
        stream.write(out.reshape(-1, 1))

    def close(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class _DecodeJob:
    samples: NDArray[np.float32]
    sample_rate: int
    force_mode: str


def format_status(st: LiveStatus) -> str:
    sq = "open" if st.squelch_open else "closed"
    if st.noise_floor_db is None:
        floor = "n/a"
        level = "n/a"
    else:
        floor = f"{st.noise_floor_db:.1f}dB"
        level = f"{st.level_db:.1f}dB" if st.level_db is not None else "n/a"
    rec = st.record_path if st.recording and st.record_path else "off"
    if st.last_mode_label:
        if st.last_confidence is None:
            cls = st.last_mode_label
        else:
            cls = f"{st.last_mode_label}:{st.last_confidence:.2f}"
    else:
        cls = "-"
    dec = st.last_decode_line or "-"
    mode = MODE_LABEL.get(st.mode, str(st.mode.value if isinstance(st.mode, LiveMode) else st.mode))
    return (
        f"status: mode={mode} squelch={sq} floor={floor} level={level} "
        f"record={rec} class={cls} decode={dec} "
        f"device={st.device or '-'} monitor={st.monitor_backend} "
        f"latency_ms={st.monitor_latency_ms:.1f}"
    )


class LiveSession:
    """
    Single-capture session: ring buffer → monitor + WAV writer + decode windows.

    ``start()`` opens the input at most once (real device) or starts one file
    feeder. Both call ``_on_capture``, which only updates the squelch and the
    ring. Classify and jt9 run on other threads.
    """

    def __init__(self, config: LiveConfig | None = None):
        self.config = config or LiveConfig()
        self._status = LiveStatus(
            mode=self.config.mode,
            recording=self.config.record_path is not None,
            record_path=str(self.config.record_path) if self.config.record_path else None,
            device=self.config.device,
            message="ready",
        )
        self._status_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._started = False
        self._stop_called = False
        self._stop = threading.Event()
        self._capture_done = threading.Event()
        self._done = threading.Event()
        self._threads: list[threading.Thread] = []
        self._ring: RingBuffer | None = None
        self.squelch: NoiseSquelch | None = None
        self._wav: PcmWavWriter | None = None
        self._player: MonitorPlayer | None = None
        self._in_stream = None
        self._file_samples: NDArray[np.float32] | None = None
        self._jobs: queue.Queue[_DecodeJob | None] = queue.Queue()
        self._decode_thread: threading.Thread | None = None
        self._captured = 0
        self._mon_frames = 0
        self._mon_audible = 0
        self._wav_frames = 0
        self._an_frames = 0
        self._capture_overruns = 0
        self._capture_opens = 0
        self._input_backend = ""
        self._ft8_chunks: list[NDArray[np.float32]] = []
        self._ft8_n = 0
        self._ft8_need = 0
        self._ft8_latch_mode = LiveMode.BOTH
        self._ft8_latch_force = "auto"
        self._ft8_latch_policy = SquelchPolicy.GATE_OFF
        self._ft8_frame0 = 0
        self._ft8_window_open = False
        self._class_since = 0
        self._class_need = 0
        self._class_latch_mode = LiveMode.BOTH
        self._class_latch_policy = SquelchPolicy.GATE_OFF
        self._class_frame0 = 0
        self._class_interval_open = False
        self._roll: list[NDArray[np.float32]] = []
        self._roll_n = 0
        self._roll_max = 0
        self._decoder: Ft8Jt9Decoder | None = None

    def status(self) -> LiveStatus:
        self._publish_meters()
        with self._status_lock:
            return replace(self._status)

    def start(self) -> None:
        """Open device once (or start the file feeder) and spawn consumers."""
        if self._started:
            raise RuntimeError("LiveSession already started")
        self._started = True
        self._stop.clear()
        self._capture_done.clear()
        self._done.clear()
        self._stop_called = False

        cfg = self.config
        sr = int(cfg.sample_rate)
        if sr <= 0:
            raise ValueError("sample_rate must be positive")
        bs = int(cfg.blocksize)
        if bs <= 0:
            raise ValueError("blocksize must be positive")

        if cfg.from_file is not None:
            audio = load_wav(cfg.from_file, channel=cfg.channel)
            if audio.sample_rate != sr:
                audio = audio.resample(sr)
            samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
            if cfg.max_seconds is not None and cfg.max_seconds > 0:
                samples = samples[: int(cfg.max_seconds * sr)]
            self._file_samples = samples
            ring_frames = max(int(sr * cfg.ring_seconds), int(samples.size) + sr, bs * 4)
            self._input_backend = "file"
            dev_label = f"file:{cfg.from_file}"
        else:
            self._file_samples = None
            ring_frames = max(int(sr * cfg.ring_seconds), bs * 8)
            self._input_backend = "sounddevice"
            dev_label = cfg.device

        self._ring = RingBuffer(ring_frames)
        self.squelch = NoiseSquelch(cfg.squelch, sr)
        self._ft8_need = max(1, int(round(cfg.ft8_window_s * sr)))
        self._class_need = max(1, int(round(cfg.classify_interval_s * sr)))
        self._roll_max = self._ft8_need
        self._captured = 0
        self._mon_frames = 0
        self._mon_audible = 0
        self._wav_frames = 0
        self._an_frames = 0
        self._capture_overruns = 0
        self._capture_opens = 0
        self._ft8_chunks = []
        self._ft8_n = 0
        self._ft8_window_open = False
        self._class_since = 0
        self._class_interval_open = False
        self._roll = []
        self._roll_n = 0

        latency_ms = 2000.0 * bs / float(sr)
        with self._status_lock:
            self._status.mode = cfg.mode
            self._status.recording = cfg.record_path is not None
            self._status.record_path = str(cfg.record_path) if cfg.record_path else None
            self._status.device = dev_label
            self._status.input_backend = self._input_backend
            self._status.monitor_latency_ms = latency_ms
            self._status.message = "starting"
            self._status.decode_lines = ()
            self._status.last_decode_line = None
            self._status.last_mode_label = None
            self._status.last_confidence = None
            self._status.decode_attempts = 0
            self._status.decode_skipped_gate = 0

        if cfg.record_path is not None:
            self._wav = PcmWavWriter(Path(cfg.record_path), sr)
        self._player = MonitorPlayer(sr, bs, cfg.output_device)
        with self._status_lock:
            self._status.monitor_backend = self._player.backend
            self._status.monitor_detail = self._player.detail
            self._status.monitor_latency_ms = self._player.latency_ms

        if latency_ms > 80.0:
            with self._status_lock:
                self._status.message = (
                    f"warning: software monitor queue {latency_ms:.0f} ms exceeds 80 ms target; "
                    "reduce --blocksize"
                )

        self._threads = []
        if self._wav is not None:
            self._threads.append(threading.Thread(target=self._wav_loop, name="digimode-wav", daemon=False))
        self._threads.append(threading.Thread(target=self._monitor_loop, name="digimode-monitor", daemon=False))
        self._decode_thread = threading.Thread(target=self._decode_loop, name="digimode-decode", daemon=False)
        self._threads.append(threading.Thread(target=self._analysis_loop, name="digimode-analysis", daemon=False))
        try:
            self._decode_thread.start()
            for t in self._threads:
                t.start()
            if cfg.from_file is not None:
                feeder = threading.Thread(target=self._file_loop, name="digimode-file", daemon=False)
                self._threads.append(feeder)
                self._capture_opens = 1
                feeder.start()
            else:
                self._open_input_stream()
        except Exception:
            self.stop()
            raise
        with self._status_lock:
            if not str(self._status.message).startswith("warning"):
                extra = ""
                if self._player and self._player.backend == "noop":
                    extra = f"; monitor noop ({self._player.detail})"
                self._status.message = "running" + extra

    def stop(self) -> None:
        if not self._started:
            with self._status_lock:
                self._status.message = "stopped"
            return
        with self._stop_lock:
            if self._stop_called:
                return
            self._stop_called = True
            self._stop.set()
            stream = self._in_stream
            self._in_stream = None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            self._capture_done.set()
            # Analysis posts the decode sentinel and joins the decode thread.
            # Join analysis before the decode thread so the sentinel is sent.
            ordered: list[threading.Thread] = []
            for t in self._threads:
                if t.name == "digimode-analysis":
                    ordered.append(t)
            for t in self._threads:
                if t.name != "digimode-analysis":
                    ordered.append(t)
            for t in ordered:
                if t.is_alive():
                    t.join(timeout=180.0)
            if self._decode_thread is not None and self._decode_thread.is_alive():
                try:
                    self._jobs.put_nowait(None)
                except Exception:  # noqa: BLE001
                    pass
                self._decode_thread.join(timeout=180.0)
            if self._player is not None:
                self._player.close()
            if self._wav is not None:
                self._wav.close()
            self._publish_meters()
            with self._status_lock:
                self._status.message = "stopped"
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the session finishes. True if finished (file EOF or stop)."""
        return self._done.wait(timeout)

    def set_mode(self, mode: LiveMode | str) -> None:
        """Monitor follows immediately; decode uses the new mode on the next window."""
        parsed = mode if isinstance(mode, LiveMode) else parse_live_mode(str(mode))
        self.config.mode = parsed
        with self._status_lock:
            self._status.mode = parsed

    def set_mute(self, mute: bool) -> None:
        """Panic-mute the monitor immediately. Does not affect record or decode."""
        self.config.mute = bool(mute)

    def set_squelch(
        self,
        open_db: float | None = None,
        close_db: float | None = None,
        hang_ms: float | None = None,
        policy: str | SquelchPolicy | None = None,
    ) -> None:
        cfg = self.config.squelch
        if open_db is not None:
            cfg.open_db = float(open_db)
        if close_db is not None:
            cfg.close_db = float(close_db)
        if hang_ms is not None:
            cfg.hang_ms = float(hang_ms)
        if policy is not None:
            cfg.policy = policy if isinstance(policy, SquelchPolicy) else parse_squelch_policy(str(policy))

    def _publish_meters(self) -> None:
        sq = self.squelch
        player = self._player
        with self._status_lock:
            st = self._status
            st.mode = self.config.mode
            st.recording = self._wav is not None and self.config.record_path is not None
            st.record_path = str(self.config.record_path) if self.config.record_path else None
            st.captured_frames = self._captured
            st.monitor_frames = self._mon_frames
            st.monitor_audible_frames = self._mon_audible
            st.wav_frames = self._wav_frames
            st.analysis_frames = self._an_frames
            st.capture_opens = self._capture_opens
            st.input_backend = self._input_backend
            ring = self._ring
            st.overruns = self._capture_overruns + (ring.overruns if ring is not None else 0)
            if player is not None:
                st.monitor_backend = player.backend
                st.monitor_detail = player.detail
                st.monitor_latency_ms = player.latency_ms
            if sq is not None:
                is_open, ever, floor, level, _frames = sq.snapshot()
                st.squelch_open = is_open
                st.squelch_ever_open = ever
                st.noise_floor_db = floor
                st.level_db = level

    def _on_capture(self, samples: NDArray[np.floating]) -> None:
        """Single producer ingest: squelch meter + ring write. No disk, no jt9."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        assert self.squelch is not None and self._ring is not None
        self.squelch.process(x)
        self._ring.write(x)
        self._captured += int(x.size)

    def _file_loop(self) -> None:
        assert self._file_samples is not None
        samples = self._file_samples
        bs = int(self.config.blocksize)
        sr = int(self.config.sample_rate)
        pos = 0
        n = int(samples.size)
        next_t = time.perf_counter()
        try:
            while pos < n and not self._stop.is_set():
                end = min(pos + bs, n)
                self._on_capture(samples[pos:end])
                step = end - pos
                pos = end
                if self.config.pace_realtime and step > 0:
                    next_t += step / float(sr)
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        finally:
            self._capture_done.set()

    def _open_input_stream(self) -> None:
        sd = _import_sounddevice()
        idx = resolve_input_device(self.config.device)
        info = sd.query_devices(idx)
        max_in = int(info.get("max_input_channels") or 1)
        if self.config.channel is None:
            channels = 1
            ch = 0
        else:
            ch = int(self.config.channel)
            if ch < 0 or ch >= max_in:
                raise LiveUnavailableError(
                    f"channel {ch} out of range for device {idx} ({max_in} input channels)"
                )
            channels = ch + 1
        self._capture_ch = ch

        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if self._stop.is_set():
                return
            if status:
                self._capture_overruns += 1
            if getattr(indata, "ndim", 1) == 2:
                use = self._capture_ch if self._capture_ch < indata.shape[1] else 0
                mono = indata[:, use]
            else:
                mono = indata
            # PortAudio reuses the buffer after the callback returns.
            self._on_capture(np.ascontiguousarray(mono, dtype=np.float32))

        try:
            stream = sd.InputStream(
                device=idx,
                channels=channels,
                samplerate=int(self.config.sample_rate),
                dtype="float32",
                blocksize=int(self.config.blocksize),
                callback=callback,
                latency="low",
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001
            raise LiveUnavailableError(
                f"Failed to open input device {self.config.device!r} (index {idx}): {exc}. "
                "Use --list-devices or --live-from-file."
            ) from exc
        self._in_stream = stream
        self._capture_opens = 1
        with self._status_lock:
            name = str(info.get("name") or idx)
            self._status.device = f"{idx}:{name}"

    def _reader_loop(self, consume, on_frames) -> None:
        assert self._ring is not None
        pos = 0
        bs = max(int(self.config.blocksize), 1)
        while True:
            data, pos, overrun = self._ring.read(pos, bs)
            if overrun:
                pass
            if data.size == 0:
                if self._capture_done.is_set() or self._stop.is_set():
                    # Drain anything that arrived between the empty read and the flag.
                    data, pos, _ = self._ring.read(pos, bs)
                    if data.size == 0:
                        break
                else:
                    time.sleep(0.001)
                    continue
            consume(data, pos - int(data.size))
            on_frames(int(data.size))

    def _wav_loop(self) -> None:
        wav = self._wav
        assert wav is not None

        def consume(data: NDArray[np.float32], start: int) -> None:  # noqa: ARG001
            wav.write(data)

        def on_frames(n: int) -> None:
            self._wav_frames += n

        self._reader_loop(consume, on_frames)

    def _monitor_loop(self) -> None:
        player = self._player
        assert player is not None and self.squelch is not None

        def consume(data: NDArray[np.float32], start: int) -> None:
            listen = (not self.config.mute) and self.config.mode != LiveMode.DECODE
            audible = bool(listen and self.squelch.open_at(start))
            player.write(data, audible)
            self._mon_frames += int(data.size)
            if audible:
                self._mon_audible += int(data.size)

        self._reader_loop(consume, lambda _n: None)

    def _analysis_loop(self) -> None:
        assert self._ring is not None
        pos = 0
        # Larger reads keep classify responsive without a syscall per 10 ms,
        # but still far below the ring so monitor/WAV are not starved.
        read_n = max(int(self.config.sample_rate // 10), int(self.config.blocksize))
        try:
            while True:
                data, pos, _over = self._ring.read(pos, read_n)
                if data.size == 0:
                    if self._capture_done.is_set() or self._stop.is_set():
                        data, pos, _over = self._ring.read(pos, read_n)
                        if data.size == 0:
                            break
                    else:
                        time.sleep(0.002)
                        continue
                start = self._an_frames
                self._an_frames += int(data.size)
                self._consume_analysis(data, start)
            self._flush_analysis()
        except Exception as exc:  # noqa: BLE001 — keep capture alive; report on status
            with self._status_lock:
                self._status.message = f"analysis error: {exc}"
        finally:
            self._jobs.put(None)
            if self._decode_thread is not None:
                self._decode_thread.join()
            self._publish_meters()
            self._done.set()

    def _append_roll(self, data: NDArray[np.float32]) -> None:
        self._roll.append(data)
        self._roll_n += int(data.size)
        while self._roll_n > self._roll_max and self._roll:
            extra = self._roll_n - self._roll_max
            head = self._roll[0]
            if head.size <= extra:
                self._roll.pop(0)
                self._roll_n -= int(head.size)
            else:
                self._roll[0] = head[extra:]
                self._roll_n -= extra
                break

    def _roll_audio(self) -> NDArray[np.float32]:
        if not self._roll:
            return np.zeros(0, dtype=np.float32)
        if len(self._roll) == 1:
            return self._roll[0]
        return np.concatenate(self._roll)

    def _consume_analysis(self, data: NDArray[np.float32], frame0: int) -> None:
        self._append_roll(data)
        if not self._class_interval_open:
            self._class_latch_mode = self.config.mode
            self._class_latch_policy = self.config.squelch.policy
            self._class_frame0 = frame0
            self._class_interval_open = True
        self._class_since += int(data.size)
        if self._class_since >= self._class_need:
            self._run_classify(self._class_frame0, self._class_frame0 + self._class_since)
            self._class_since = 0
            self._class_interval_open = False

        offset = 0
        while offset < data.size:
            if not self._ft8_window_open:
                self._begin_ft8_window(frame0 + offset)
            space = self._ft8_need - self._ft8_n
            take = min(space, int(data.size) - offset)
            piece = data[offset : offset + take]
            self._ft8_chunks.append(piece)
            self._ft8_n += take
            offset += take
            if self._ft8_n >= self._ft8_need:
                self._finish_ft8_window(partial=False)

    def _begin_ft8_window(self, frame0: int) -> None:
        self._ft8_latch_mode = self.config.mode
        self._ft8_latch_force = self.config.force_mode
        self._ft8_latch_policy = self.config.squelch.policy
        self._ft8_frame0 = int(frame0)
        self._ft8_window_open = True

    def _window_saw_open(self, start: int, end: int) -> bool:
        if self.squelch is None:
            return False
        return self.squelch.any_open(start, end)

    def _run_classify(self, start: int, end: int) -> None:
        if not decode_window_allowed(
            self._class_latch_mode,
            self._class_latch_policy,
            self._window_saw_open(start, end),
        ):
            return
        samples = self._roll_audio()
        if samples.size < max(1, self.config.sample_rate // 5):
            return
        audio = AudioBuffer(samples=samples, sample_rate=int(self.config.sample_rate))
        result = classify(audio)
        with self._status_lock:
            self._status.last_mode_label = result.mode
            self._status.last_confidence = float(result.confidence)

    def _finish_ft8_window(self, partial: bool) -> None:
        chunks = self._ft8_chunks
        n = self._ft8_n
        mode = self._ft8_latch_mode
        force = self._ft8_latch_force
        policy = self._ft8_latch_policy
        frame0 = self._ft8_frame0
        self._ft8_chunks = []
        self._ft8_n = 0
        self._ft8_window_open = False
        if n <= 0 or not chunks:
            return
        # A short tail at EOF is not an FT8 period. Keep near-full windows.
        if partial and n < int(0.9 * self._ft8_need):
            return
        saw = self._window_saw_open(frame0, frame0 + n)
        if mode == LiveMode.LISTEN:
            return
        if not decode_window_allowed(mode, policy, saw):
            with self._status_lock:
                self._status.decode_skipped_gate += 1
                self._status.message = "decode skipped (squelch gate-off)"
            return
        samples = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        self._jobs.put(
            _DecodeJob(
                samples=np.asarray(samples, dtype=np.float32).copy(),
                sample_rate=int(self.config.sample_rate),
                force_mode=force,
            )
        )

    def _flush_analysis(self) -> None:
        if self._class_since > 0:
            self._run_classify(self._class_frame0, self._class_frame0 + self._class_since)
            self._class_since = 0
            self._class_interval_open = False
        if self._ft8_n > 0:
            self._finish_ft8_window(partial=True)

    def _decode_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self._run_decode(job)
            except Exception as exc:  # noqa: BLE001
                with self._status_lock:
                    self._status.message = f"decode error: {exc}"

    def _decoder_for(self) -> Ft8Jt9Decoder | StubDecoder:
        if self.config.jt9_path:
            return Ft8Jt9Decoder(jt9_path=self.config.jt9_path)
        if self._decoder is None:
            dec = get_ft8_decoder()
            if isinstance(dec, Ft8Jt9Decoder):
                self._decoder = dec
            return dec
        return self._decoder

    def _run_decode(self, job: _DecodeJob) -> None:
        audio = AudioBuffer(samples=job.samples, sample_rate=job.sample_rate)
        if job.force_mode == "auto":
            classification = classify(audio)
            mode = classification.mode
            with self._status_lock:
                self._status.last_mode_label = classification.mode
                self._status.last_confidence = float(classification.confidence)
        else:
            mode = job.force_mode
        if mode == "FT8":
            result = self._decoder_for().decode(audio)
        elif mode in ("RTTY", "PSK31"):
            result = StubDecoder(
                mode=mode,
                reason=f"{mode} decode not implemented (classifier only)",
            ).decode(audio)
        else:
            with self._status_lock:
                self._status.decode_attempts += 1
                self._status.message = f"no decoder for mode {mode}"
            return
        with self._status_lock:
            self._status.decode_attempts += 1
            if result.lines:
                self._status.decode_lines = self._status.decode_lines + tuple(result.lines)
                self._status.last_decode_line = result.lines[-1]
            if result.ok:
                self._status.message = "decoded"
            else:
                self._status.message = result.error or "no decode"


def live_listen_record(
    device: str = "default",
    mode: str = "both",
    record_path: str | Path | None = None,
    force_mode: str = "auto",
    squelch_open_db: float = 6.0,
    squelch_close_db: float = 3.0,
    squelch_hang_ms: float = 500.0,
    squelch_policy: str = "gate-off",
    *,
    from_file: str | Path | None = None,
    channel: int | None = None,
    mute: bool = False,
    sample_rate: int = 48000,
    blocksize: int = 480,
    pace_realtime: bool = True,
    output_device: str | None = None,
    max_seconds: float | None = None,
    jt9_path: str | None = None,
) -> LiveSession:
    """
    Convenience constructor for listen + optional WAV record + decode tee.

    Does not open the device until ``LiveSession.start()``.
    """
    cfg = LiveConfig(
        device=device,
        mode=parse_live_mode(mode),
        record_path=Path(record_path) if record_path else None,
        force_mode=force_mode,
        sample_rate=sample_rate,
        mute=mute,
        from_file=Path(from_file) if from_file else None,
        channel=channel,
        output_device=output_device,
        blocksize=blocksize,
        pace_realtime=pace_realtime,
        max_seconds=max_seconds,
        jt9_path=jt9_path,
        squelch=SquelchConfig(
            open_db=squelch_open_db,
            close_db=squelch_close_db,
            hang_ms=squelch_hang_ms,
            policy=parse_squelch_policy(squelch_policy),
        ),
    )
    return LiveSession(cfg)
