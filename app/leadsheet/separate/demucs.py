"""Stem separation through Demucs (optional dependency)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

STEMS = ("vocals", "drums", "bass", "other", "piano", "guitar")
HARMONIC_STEMS = ("bass", "other", "piano", "guitar")


class SeparationUnavailable(RuntimeError):
    pass


def separate_stems(
    wav: Path,
    out_root: Path,
    model: str = "htdemucs_6s",
) -> dict[str, Path]:
    """Run Demucs as a subprocess so torch is never imported in-process."""
    wav, out_root = Path(wav), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "--out",
        str(out_root),
        str(wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
        raise SeparationUnavailable(f"demucs failed: {' | '.join(detail)}")

    base = out_root / model / wav.stem
    stems = {name: base / f"{name}.wav" for name in STEMS if (base / f"{name}.wav").exists()}
    if not stems:
        raise SeparationUnavailable(f"demucs produced no stems under {base}")
    return stems


def harmonic_mix(stems: dict[str, Path], dst: Path, sample_rate: int = 22_050) -> Path:
    """Sum the pitched stems so the chord recognizer ignores vocals and drums."""
    import librosa
    import soundfile as sf

    parts = [path for name, path in stems.items() if name in HARMONIC_STEMS]
    if not parts:
        parts = list(stems.values())

    total: np.ndarray | None = None
    for path in parts:
        y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        total = y if total is None else total + y
    if total is None or total.size == 0:
        raise SeparationUnavailable("no harmonic stems to mix")

    peak = float(np.max(np.abs(total)))
    if peak > 1.0:
        total = total / peak
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), total, sample_rate, subtype="PCM_16")
    return dst
