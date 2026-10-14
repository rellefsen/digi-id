"""FT8 decode windows locked to UTC slots.

On-air FT8 periods are 15 seconds beginning at Unix epoch multiples
(:00, :15, :30, :45). A window that starts mid-slot splits the
transmission and jt9 misses it. Device capture drops the leading
partial slot, then cuts every following period on that boundary.
File playback keeps stream-relative windows unless UTC alignment is
asked for explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


def frames_until_ft8_slot(
    instant_utc: float,
    sample_rate: int,
    *,
    period_s: float = 15.0,
    tolerance_s: float = 0.05,
) -> int:
    """Frames to drop so the next kept sample starts on an FT8 UTC boundary.

    ``instant_utc`` is the Unix time of the sample under consideration.
    A sample up to ``tolerance_s`` after a boundary counts as aligned.
    Waiting out the other 14.95 s would drop the transmission that
    just started. Samples still before the boundary are not aligned:
    small capture blocks must not open the period early on every slot.
    """
    if period_s <= 0:
        raise ValueError("period_s must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    offset = float(instant_utc) % float(period_s)
    if offset <= tolerance_s:
        return 0
    frames = int(round((period_s - offset) * int(sample_rate)))
    return max(0, frames)


def capture_epoch_utc(
    now: float,
    captured_before: int,
    sample_rate: int,
    adc_delay_s: float,
) -> float:
    """Unix time of stream sample 0.

    ``now`` is the callback time. ``adc_delay_s`` is how much earlier the
    first sample of the current block was captured. ``captured_before``
    is the number of samples already written before this block.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    delay = float(adc_delay_s)
    if not np.isfinite(delay) or delay < 0.0 or delay > 1.0:
        delay = 0.0
    first_of_block = float(now) - delay
    return first_of_block - (int(captured_before) / float(sample_rate))


@dataclass
class Ft8Window:
    """One decode period, starting at ``frame0`` on the capture timeline."""

    samples: NDArray[np.float32]
    frame0: int
    latch: object = None


class Ft8WindowCutter:
    """Slice a capture timeline into FT8 periods.

    Stream mode (``align_utc=False``) starts at the first sample and
    takes ``period_s`` chunks, which is the right cut for a file that
    is already one period.

    UTC mode drops samples until the next slot boundary, using
    ``epoch_utc`` as the Unix time of capture frame 0. A start that is
    already a few tens of milliseconds into the slot is kept. Later
    periods stay contiguous with that first boundary. A gap in the
    frame index abandons the open period and waits for the next
    boundary, so a ring overrun cannot splice two parts of the timeline.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        period_s: float = 15.0,
        align_utc: bool = False,
        tolerance_s: float = 0.05,
        on_begin: Callable[[int], object] | None = None,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self.sample_rate = int(sample_rate)
        self.period_s = float(period_s)
        self.align_utc = bool(align_utc)
        self.tolerance_s = float(tolerance_s)
        self._on_begin = on_begin
        self._need = max(1, int(round(self.period_s * self.sample_rate)))
        self._epoch: float | None = None
        self._chunks: list[NDArray[np.float32]] = []
        self._n = 0
        self._open = False
        self._frame0 = 0
        self._latch: object = None
        self._expect: int | None = None
        self.lead_in_frames = 0
        self.hold_frames = 0
        self.slots = 0
        self.resyncs = 0

    @property
    def epoch_utc(self) -> float | None:
        return self._epoch

    @property
    def window_frames(self) -> int:
        return self._need

    def set_epoch(self, epoch_utc: float) -> None:
        """Unix time of capture frame 0. Ignored once a period is open."""
        if self._open:
            return
        self._epoch = float(epoch_utc)

    def push(self, samples: NDArray[np.floating], frame0: int) -> list[Ft8Window]:
        """Take one contiguous capture chunk. ``frame0`` is its first frame."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return []
        frame0 = int(frame0)
        if self._open and self._expect is not None and frame0 != self._expect:
            self._abandon()
        finished: list[Ft8Window] = []
        offset = 0
        abs_frame = frame0
        while offset < data.size:
            if not self._open:
                if self.align_utc:
                    if self._epoch is None:
                        drop = data.size - offset
                        self.lead_in_frames += drop
                        offset += drop
                        abs_frame += drop
                        break
                    utc = self._epoch + (abs_frame / float(self.sample_rate))
                    skip = frames_until_ft8_slot(
                        utc,
                        self.sample_rate,
                        period_s=self.period_s,
                        tolerance_s=self.tolerance_s,
                    )
                    if skip > 0:
                        drop = min(skip, data.size - offset)
                        self.lead_in_frames += drop
                        offset += drop
                        abs_frame += drop
                        continue
                self._open_at(abs_frame)
            space = self._need - self._n
            take = min(space, data.size - offset)
            self._chunks.append(data[offset : offset + take])
            self._n += take
            offset += take
            abs_frame += take
            if self._n >= self._need:
                finished.append(self._close())
        self._expect = frame0 + int(data.size)
        self._update_hold(self._expect)
        return finished

    def flush(self) -> Ft8Window | None:
        """Finish a near-full tail. A short tail is not an FT8 period."""
        if not self._open or self._n <= 0:
            self._update_hold(self._expect or 0)
            return None
        if self._n < int(0.9 * self._need):
            self._chunks = []
            self._n = 0
            self._open = False
            self._latch = None
            self.hold_frames = 0
            return None
        window = self._close()
        self.hold_frames = 0
        return window

    def _open_at(self, frame0: int) -> None:
        self._frame0 = int(frame0)
        self._chunks = []
        self._n = 0
        self._open = True
        self._latch = self._on_begin(self._frame0) if self._on_begin is not None else None
        self.hold_frames = 0

    def _close(self) -> Ft8Window:
        if len(self._chunks) == 1:
            samples = np.asarray(self._chunks[0], dtype=np.float32).copy()
        else:
            samples = np.concatenate(self._chunks).astype(np.float32, copy=False)
        window = Ft8Window(samples=samples, frame0=self._frame0, latch=self._latch)
        self._chunks = []
        self._n = 0
        self._open = False
        self._latch = None
        self.slots += 1
        return window

    def _abandon(self) -> None:
        self._chunks = []
        self._n = 0
        self._open = False
        self._latch = None
        self.resyncs += 1

    def _update_hold(self, next_frame: int) -> None:
        if self._open or not self.align_utc or self._epoch is None:
            self.hold_frames = 0
            return
        utc = self._epoch + (int(next_frame) / float(self.sample_rate))
        self.hold_frames = frames_until_ft8_slot(
            utc,
            self.sample_rate,
            period_s=self.period_s,
            tolerance_s=self.tolerance_s,
        )
