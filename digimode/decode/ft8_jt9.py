"""FT8 decode via WSJT-X `jt9 -8` (offline CLI)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from digimode.decode.base import DecodeResult, Decoder
from digimode.ingest import AudioBuffer


def find_jt9() -> str | None:
    """Locate jt9 executable (WSJT-X package)."""
    env = os.environ.get("DIGIMODE_JT9")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    return shutil.which("jt9")


_DECODE_LINE = re.compile(
    r"^\s*\d{4,6}\s+"  # HHMMSS or similar timestamp from jt9
    r".+"
)


def _parse_jt9_lines(stdout: str) -> list[str]:
    """Extract human-readable FT8 message lines from jt9 stdout."""
    lines: list[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("<DecodeFinished>") or s.startswith("DecodeFinished"):
            continue
        if s.startswith("Usage:") or s.startswith("OPTIONS"):
            continue
        # Typical: "000001 -10  0.0 1500 ~  CQ K1ABC FN42"
        if "~" in s:
            msg = s.split("~", 1)[1].strip()
            if msg:
                lines.append(msg)
            else:
                lines.append(s)
        elif _DECODE_LINE.match(s) and not s.startswith("-"):
            lines.append(s)
    return lines


class Ft8Jt9Decoder(Decoder):
    """Decode FT8 from WAV using WSJT-X jt9."""

    mode = "FT8"

    def __init__(self, jt9_path: str | None = None, depth: int = 3):
        self.jt9_path = jt9_path or find_jt9()
        self.depth = depth
        if not self.jt9_path:
            raise FileNotFoundError("jt9 not found; install wsjtx or set DIGIMODE_JT9")

    def decode(self, audio: AudioBuffer, work_dir: Path | None = None) -> DecodeResult:
        # jt9 expects 12 kHz mono 16-bit WAVs for FT8 (ft8sim produces these)
        buf = audio.resample(12000) if audio.sample_rate != 12000 else audio

        own_tmp = work_dir is None
        if work_dir is None:
            tmp = tempfile.TemporaryDirectory(prefix="digimode_ft8_")
            work = Path(tmp.name)
        else:
            tmp = None
            work = Path(work_dir)
            work.mkdir(parents=True, exist_ok=True)

        try:
            wav_path = work / "input_12000.wav"
            buf.write_wav(wav_path)

            # -8 = FT8, -d depth, -a/-t writable paths
            cmd = [
                self.jt9_path,
                "-8",
                "-d",
                str(self.depth),
                "-a",
                str(work),
                "-t",
                str(work),
                str(wav_path),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(work),
                    env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
                )
            except subprocess.TimeoutExpired:
                return DecodeResult(
                    mode=self.mode,
                    backend=f"jt9:{self.jt9_path}",
                    ok=False,
                    error="jt9 timed out after 120s",
                )
            except FileNotFoundError:
                return DecodeResult(
                    mode=self.mode,
                    backend="jt9",
                    ok=False,
                    error=f"jt9 executable missing: {self.jt9_path}",
                )

            raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            lines = _parse_jt9_lines(proc.stdout or "")
            ok = proc.returncode == 0 and len(lines) > 0
            err = None
            if proc.returncode != 0 and not lines:
                err = f"jt9 exit {proc.returncode}"
            elif not lines:
                err = "jt9 produced no decode lines (signal may be too weak or not FT8)"

            return DecodeResult(
                mode=self.mode,
                lines=lines,
                raw_output=raw.strip(),
                backend=f"jt9:{self.jt9_path}",
                ok=ok,
                error=err,
            )
        finally:
            if own_tmp and tmp is not None:
                tmp.cleanup()
