"""Audio ingest: cut a fragment and normalize it to mono WAV."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class AudioError(RuntimeError):
    pass


def extract_fragment(
    src: Path,
    dst: Path,
    start: float | None = None,
    end: float | None = None,
    sample_rate: int = 22_050,
) -> Path:
    """Extract ``[start, end)`` from ``src`` into a mono WAV at ``dst``.

    Prefers ``ffmpeg`` (handles any container); falls back to libsndfile via
    soundfile when ffmpeg is unavailable.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error"]
        if start:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(src)]
        if end is not None:
            duration = max(0.0, end - (start or 0.0))
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-ac", "1", "-ar", str(sample_rate), "-vn", str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return dst
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown error"]
        raise AudioError(f"ffmpeg failed: {detail[0]}")

    import librosa
    import soundfile as sf

    offset = start or 0.0
    duration = None if end is None else max(0.0, end - offset)
    try:
        y, _ = librosa.load(str(src), sr=sample_rate, mono=True, offset=offset, duration=duration)
    except Exception as exc:  # pragma: no cover - depends on local codecs
        raise AudioError(f"could not read {src}: {exc}") from exc
    if y.size == 0:
        raise AudioError(f"no audio extracted from {src}")
    sf.write(str(dst), y, sample_rate, subtype="PCM_16")
    return dst


def load_audio(path: Path, sample_rate: int = 22_050) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32)


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)
