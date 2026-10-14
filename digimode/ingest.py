"""WAV ingest: load mono float32, optional resample."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioBuffer:
    """In-memory mono audio."""

    samples: NDArray[np.floating]
    sample_rate: int
    path: Path | None = None

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)

    def resample(self, target_rate: int) -> "AudioBuffer":
        if target_rate == self.sample_rate:
            return self
        if target_rate <= 0:
            raise ValueError("target_rate must be positive")
        n_out = int(round(len(self.samples) * target_rate / self.sample_rate))
        if n_out <= 0:
            return AudioBuffer(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=target_rate,
                path=self.path,
            )
        x_old = np.linspace(0.0, 1.0, num=len(self.samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(x_new, x_old, self.samples.astype(np.float64))
        return AudioBuffer(
            samples=y.astype(np.float32),
            sample_rate=target_rate,
            path=self.path,
        )

    def write_wav(self, path: Path | str, sampwidth: int = 2) -> Path:
        """Write 16-bit PCM WAV (common for jt9)."""
        path = Path(path)
        pcm = np.clip(self.samples, -1.0, 1.0)
        if sampwidth == 2:
            data = (pcm * 32767.0).astype(np.int16)
        else:
            raise ValueError("only 16-bit PCM supported")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(sampwidth)
            w.setframerate(self.sample_rate)
            w.writeframes(data.tobytes())
        return path


def load_wav(path: Path | str, channel: int | None = None) -> AudioBuffer:
    """Load a WAV file as mono float32 in [-1, 1].

    channel: 0-based channel to keep. None averages all channels (file default).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"WAV not found: {path}")

    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        # 24-bit packed little-endian
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        # sign-extend to int32
        b = (
            a[:, 0].astype(np.int32)
            | (a[:, 1].astype(np.int32) << 8)
            | (a[:, 2].astype(np.int32) << 16)
        )
        b = np.where(b & 0x800000, b - 0x1000000, b)
        data = b.astype(np.float32) / 8388608.0
    elif sw == 4:
        # try int32; if floats, wave usually won't use IEEE without extensible
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")

    if nch > 1:
        data = data.reshape(-1, nch)
        if channel is None:
            data = data.mean(axis=1)
        else:
            if channel < 0 or channel >= nch:
                raise ValueError(f"channel {channel} out of range for {nch}-channel WAV")
            data = data[:, channel]
    elif channel not in (None, 0):
        raise ValueError(f"channel {channel} out of range for mono WAV")

    return AudioBuffer(samples=data.astype(np.float32), sample_rate=rate, path=path)
