"""Heuristic mode classifier from spectrogram / tone fingerprints.

M1 cues (offline, no ML):
  FT8  — ~6.25 Hz tone spacing within ~50 Hz BW, ~15 s cadence energy
  RTTY — two strong tones ~170 Hz apart (FSK mark/space)
  PSK31 — single very narrow (~31 Hz) dominant carrier
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from digimode.ingest import AudioBuffer


@dataclass
class Classification:
    mode: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "confidence": round(self.confidence, 3),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "notes": list(self.notes),
        }


def _stft_mag(
    samples: NDArray[np.floating],
    sr: int,
    n_fft: int = 8192,
    hop: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    hop = hop or max(n_fft // 8, 1)
    window = np.hanning(n_fft).astype(np.float64)
    samples = samples.astype(np.float64)
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))

    n_frames = 1 + (len(samples) - n_fft) // hop
    if n_frames < 1:
        n_frames = 1
    specs = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = samples[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        specs[:, i] = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = (np.arange(n_frames) * hop + n_fft / 2) / sr
    return specs, freqs, times


def _band_mask(freqs: NDArray, lo: float, hi: float) -> NDArray[np.bool_]:
    return (freqs >= lo) & (freqs <= hi)


def _top_local_peaks(
    spectrum: NDArray[np.floating],
    freqs: NDArray[np.floating],
    k: int = 12,
    min_hz: float = 8.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return (peak_freqs, peak_mags) for strongest local maxima, min spaced."""
    x = spectrum.astype(np.float64)
    if x.max() <= 0:
        return np.array([]), np.array([])
    cand: list[int] = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            cand.append(i)
    if not cand:
        return np.array([]), np.array([])
    cand = sorted(cand, key=lambda i: x[i], reverse=True)
    chosen: list[int] = []
    for i in cand:
        if all(abs(freqs[i] - freqs[j]) >= min_hz for j in chosen):
            chosen.append(i)
        if len(chosen) >= k:
            break
    chosen = sorted(chosen, key=lambda i: freqs[i])
    idx = np.array(chosen, dtype=int)
    return freqs[idx], x[idx]


def _occupied_bw(spectrum: NDArray, freqs: NDArray, rel: float = 0.25) -> tuple[float, float, float]:
    """Return (bw_hz, f_lo, f_hi) of bins above rel * max in-band."""
    if spectrum.size == 0 or spectrum.max() <= 0:
        return 0.0, 0.0, 0.0
    thr = spectrum.max() * rel
    strong = freqs[spectrum >= thr]
    if strong.size == 0:
        return 0.0, 0.0, 0.0
    return float(strong.max() - strong.min()), float(strong.min()), float(strong.max())


def _best_narrowband_center(
    spectrum: NDArray, freqs: NDArray, width_hz: float = 50.0
) -> tuple[float, float, float]:
    """Sliding window: return (center_hz, window_energy, local_fraction).

    local_fraction = energy in `width_hz` / energy in a 400 Hz neighborhood
    around the same center (avoids dilution by the full HF band).
    """
    band = _band_mask(freqs, 200.0, 3000.0)
    f = freqs[band]
    s = spectrum[band]
    if f.size < 4 or s.max() <= 0:
        return 0.0, 0.0, 0.0
    df = float(f[1] - f[0]) if f.size > 1 else 1.0
    half = max(int(round((width_hz / 2) / df)), 1)
    half_local = max(int(round(200.0 / df)), half + 1)  # ±200 Hz
    csum = np.concatenate([[0.0], np.cumsum(s)])
    best_frac = -1.0
    best_e = 0.0
    best_i = half
    for i in range(half, len(s) - half):
        e = float(csum[i + half + 1] - csum[i - half])
        lo = max(0, i - half_local)
        hi = min(len(s), i + half_local + 1)
        local = float(csum[hi] - csum[lo]) + 1e-12
        frac = e / local
        if frac > best_frac:
            best_frac = frac
            best_e = e
            best_i = i
    return float(f[best_i]), best_e, float(best_frac)


def _tone_spacing_score(
    spectrum: NDArray, freqs: NDArray, center: float, width_hz: float = 55.0
) -> tuple[float, list[str]]:
    """Look for ~6.25 Hz peak spacing inside a narrow window around center."""
    notes: list[str] = []
    lo, hi = center - width_hz / 2, center + width_hz / 2
    mask = _band_mask(freqs, lo, hi)
    seg = spectrum.copy()
    seg[~mask] = 0.0
    peaks, pmags = _top_local_peaks(seg, freqs, k=10, min_hz=4.0)
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]
    if peaks.size < 2:
        return 0.0, notes
    diffs = np.diff(np.sort(peaks))
    near = diffs[(diffs >= 5.0) & (diffs <= 8.0)]
    if near.size >= 3:
        notes.append(f"{near.size} tone spacings ~6.25 Hz near {center:.0f} Hz")
        return 0.45, notes
    if near.size >= 1:
        notes.append(f"{near.size} tone spacing(s) ~{float(np.median(near)):.1f} Hz")
        return 0.25, notes
    return 0.0, notes


def _score_ft8(
    mag: NDArray, freqs: NDArray, times: NDArray, duration_s: float
) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 0.0
    band = _band_mask(freqs, 200.0, 3000.0)
    env = mag[band].sum(axis=0)
    env_n = env / (env.max() + 1e-12)
    active = env_n > 0.2

    # Prefer median of active frames (less smear than mean over whole file)
    if np.any(active):
        frame_specs = mag[:, active]
    else:
        frame_specs = mag
    # Use a strong mid-TX frame if available
    if frame_specs.shape[1] >= 3:
        strengths = frame_specs[band].sum(axis=0)
        idx = int(np.argsort(strengths)[len(strengths) // 2])
        spec = frame_specs[:, idx] * band.astype(float)
    else:
        spec = frame_specs.mean(axis=1) * band.astype(float)

    center, win_e, frac = _best_narrowband_center(spec, freqs, width_hz=50.0)
    if frac >= 0.45:
        score += 0.3
        notes.append(f"~50 Hz energy cluster @ {center:.0f} Hz (local_frac={frac:.2f})")
    elif frac >= 0.28:
        score += 0.15
        notes.append(f"partial 50 Hz cluster @ {center:.0f} Hz (local_frac={frac:.2f})")

    spacing_s, spacing_n = _tone_spacing_score(spec, freqs, center if center else float(freqs[np.argmax(spec)]))
    score += spacing_s
    notes.extend(spacing_n)

    # Cadence: FT8 TX ~12.6/15
    if duration_s >= 12.0:
        t_mask = times <= min(15.0, duration_s)
        if np.any(t_mask):
            act = env_n[t_mask] > 0.2
            afrac = float(act.mean())
            if 0.55 <= afrac <= 1.0:
                score += 0.3
                notes.append(f"15s-slot activity fraction {afrac:.2f}")
            elif 0.35 <= afrac < 0.55:
                score += 0.12
                notes.append(f"partial slot activity {afrac:.2f}")
    else:
        notes.append("duration < 12s; cadence cue skipped")
        if score > 0:
            score *= 0.5
            notes.append("FT8 score halved (short clip)")

    # Veto: single ultra-narrow carrier (PSK-like)
    peaks, pmags = _top_local_peaks(spec, freqs, k=5, min_hz=8.0)
    if peaks.size >= 1:
        pmax = float(pmags.max())
        second = float(np.partition(pmags, -2)[-2]) if pmags.size >= 2 else 0.0
        local = spec.copy()
        c = float(freqs[int(np.argmax(spec))])
        local[~_band_mask(freqs, c - 40, c + 40)] = 0
        bw3_bins = local >= (pmax / np.sqrt(2))
        if np.any(bw3_bins):
            bw3 = float(freqs[bw3_bins].max() - freqs[bw3_bins].min())
        else:
            bw3 = 0.0
        if second < 0.2 * pmax and bw3 < 15.0:
            score *= 0.2
            notes.append("single ultra-narrow carrier (FT8 veto)")

    return float(min(score, 1.0)), notes


def _score_rtty(mag: NDArray, freqs: NDArray) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 0.0
    band = _band_mask(freqs, 400.0, 2800.0)
    mean_spec = mag.mean(axis=1) * band.astype(float)
    peaks, pmags = _top_local_peaks(mean_spec, freqs, k=8, min_hz=40.0)
    if peaks.size < 2:
        return 0.0, ["fewer than 2 strong tones"]

    # Require the TWO strongest tones (by magnitude) to be ~170 Hz apart
    order = np.argsort(pmags)[::-1]
    f1, f2 = float(peaks[order[0]]), float(peaks[order[1]])
    m1, m2 = float(pmags[order[0]]), float(pmags[order[1]])
    shift = abs(f1 - f2)
    strength_ratio = m2 / (m1 + 1e-12)

    if 155 <= shift <= 185 and strength_ratio >= 0.35:
        score += 0.55
        notes.append(f"dominant FSK pair shift {shift:.0f} Hz (mark/space)")
    elif 75 <= shift <= 95 and strength_ratio >= 0.35:
        score += 0.35
        notes.append(f"dominant FSK pair shift {shift:.0f} Hz (~85 Hz)")
    elif 150 <= shift <= 190 and strength_ratio >= 0.2:
        score += 0.25
        notes.append(f"weak FSK pair shift {shift:.0f} Hz")
    else:
        notes.append(f"top-2 tone shift {shift:.0f} Hz (not RTTY-like)")

    # Two comparable tones overall
    if strength_ratio >= 0.4 and 100 <= shift <= 250:
        score += 0.2
        notes.append("two comparable tones (FSK-like)")

    # Occupied BW should be roughly the shift (not a solid 50 Hz blob)
    bw, _, _ = _occupied_bw(mean_spec, freqs, rel=0.3)
    if 120 <= bw <= 250:
        score += 0.15
        notes.append(f"FSK occupied BW ~{bw:.0f} Hz")

    return float(min(score, 1.0)), notes


def _score_psk31(mag: NDArray, freqs: NDArray) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 0.0
    band = _band_mask(freqs, 200.0, 3000.0)
    mean_spec = mag.mean(axis=1) * band.astype(float)
    if mean_spec.max() <= 0:
        return 0.0, notes

    # 3 dB bandwidth around strongest peak
    peak_i = int(np.argmax(mean_spec))
    f0 = float(freqs[peak_i])
    half = mean_spec[peak_i] / np.sqrt(2)  # ~-3 dB
    # walk left/right
    lo = peak_i
    while lo > 0 and mean_spec[lo] >= half:
        lo -= 1
    hi = peak_i
    while hi < len(mean_spec) - 1 and mean_spec[hi] >= half:
        hi += 1
    bw3 = float(freqs[hi] - freqs[lo])

    if 8 <= bw3 <= 45:
        score += 0.5
        notes.append(f"narrow -3dB BW ~{bw3:.0f} Hz at {f0:.0f} Hz")
    elif 5 <= bw3 <= 70:
        score += 0.25
        notes.append(f"moderately narrow BW ~{bw3:.0f} Hz at {f0:.0f} Hz")

    peaks, pmags = _top_local_peaks(mean_spec, freqs, k=5, min_hz=20.0)
    if peaks.size >= 1:
        if peaks.size == 1 or pmags.max() > 3.0 * np.partition(pmags, -2)[-2]:
            score += 0.3
            notes.append(f"single dominant carrier ~{f0:.0f} Hz")
        else:
            # multiple strong tones → not PSK31
            score *= 0.4
            notes.append("multiple strong tones (penalized for PSK31)")

    # Occupied BW at 25% should also be narrow
    bw, _, _ = _occupied_bw(mean_spec, freqs, rel=0.25)
    if 10 <= bw <= 55:
        score += 0.15
        notes.append(f"occupied BW ~{bw:.0f} Hz")
    elif bw > 100:
        score *= 0.5
        notes.append(f"wide occupied BW ~{bw:.0f} Hz (not PSK31)")

    return float(min(max(score, 0.0), 1.0)), notes


def classify(audio: AudioBuffer, n_fft: int | None = None) -> Classification:
    """Classify digimode from a mono AudioBuffer."""
    buf = audio.resample(12000) if audio.sample_rate != 12000 else audio
    # Fine bins: 12000/8192 ≈ 1.46 Hz (resolves FT8 6.25 Hz spacing)
    n_fft = n_fft or 8192
    mag, freqs, times = _stft_mag(buf.samples, buf.sample_rate, n_fft=n_fft)

    ft8_s, ft8_n = _score_ft8(mag, freqs, times, buf.duration_s)
    rtty_s, rtty_n = _score_rtty(mag, freqs)
    psk_s, psk_n = _score_psk31(mag, freqs)

    scores = {"FT8": ft8_s, "RTTY": rtty_s, "PSK31": psk_s}
    mode = max(scores, key=scores.get)
    best = scores[mode]
    second = sorted(scores.values(), reverse=True)[1]

    margin = best - second
    confidence = float(np.clip(0.55 * best + 0.45 * max(margin, 0.0), 0.0, 1.0))
    # Prefer UNKNOWN when everything is weak
    if best < 0.28:
        mode = "UNKNOWN"
        confidence = float(best)
        notes = ["low scores for all modes"] + ft8_n + rtty_n + psk_n
    else:
        note_map = {"FT8": ft8_n, "RTTY": rtty_n, "PSK31": psk_n}
        notes = note_map.get(mode, [])

    return Classification(mode=mode, confidence=confidence, scores=scores, notes=notes)
