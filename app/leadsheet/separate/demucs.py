"""Stem separation through Demucs (optional dependency)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

STEMS = ("vocals", "drums", "bass", "other", "piano", "guitar")
HARMONIC_STEMS = ("bass", "other", "piano", "guitar")


class SeparationUnavailable(RuntimeError):
    pass


class SeparationCancelled(Exception):
    pass


def separate_stems(
    wav: Path,
    out_root: Path,
    model: str = "htdemucs_6s",
    *,
    device: str | None = None,
    two_stems: str | None = None,
    cancel: threading.Event | None = None,
    timeout: float | None = None,
) -> dict[str, Path]:
    """Run Demucs as a subprocess so torch is never imported in-process."""
    wav, out_root = Path(wav), Path(out_root)
    if two_stems is not None and two_stems not in STEMS:
        raise ValueError(f"unsupported Demucs stem: {two_stems}")
    if cancel is not None and cancel.is_set():
        raise SeparationCancelled()
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
    ]
    if device is not None:
        cmd += ["--device", device]
    if two_stems is not None:
        cmd += ["--two-stems", two_stems]
    cmd += ["--out", str(out_root), str(wav)]

    with tempfile.TemporaryFile() as stderr:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
        except OSError as exc:
            raise SeparationUnavailable(f"demucs could not start: {exc}") from exc

        started = time.monotonic()
        try:
            while proc.poll() is None:
                if cancel is not None and cancel.wait(0.25):
                    raise SeparationCancelled()
                if cancel is None:
                    time.sleep(0.25)
                if timeout is not None and time.monotonic() - started > timeout:
                    raise SeparationUnavailable(f"demucs timed out after {timeout:g} seconds")
            return_code = proc.wait()
        finally:
            if proc.poll() is None:
                _terminate(proc)

        stderr.seek(0, 2)
        stderr_size = stderr.tell()
        stderr.seek(max(0, stderr_size - 8192))
        detail = stderr.read().decode("utf-8", errors="replace").strip().splitlines()[-2:]
    if return_code != 0:
        raise SeparationUnavailable(
            f"demucs failed: {' | '.join(detail)}" if detail else "demucs failed"
        )

    base = out_root / model / wav.stem
    names = (two_stems, f"no_{two_stems}") if two_stems is not None else STEMS
    stems = {name: base / f"{name}.wav" for name in names if (base / f"{name}.wav").exists()}
    if not stems:
        raise SeparationUnavailable(f"demucs produced no stems under {base}")
    return stems


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


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
