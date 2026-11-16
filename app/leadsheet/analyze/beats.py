"""Tempo, beat, downbeat and key estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_BEATS_PER_MEASURE, HOP_LENGTH
from ..transcribe.events import root_name

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_FLAT_MAJORS = {1, 3, 5, 6, 8, 10}  # Db, Eb, F, Gb, Ab, Bb
_FLAT_MINORS = {0, 2, 5, 7, 10}  # c, d, f, g, bb


@dataclass(frozen=True, slots=True)
class BeatGrid:
    tempo: float
    beats: list[float]
    downbeats: list[float]
    beats_per_measure: int = DEFAULT_BEATS_PER_MEASURE
    time_signature: str = "4/4"


@dataclass(frozen=True, slots=True)
class KeyEstimate:
    tonic: int
    mode: str
    confidence: float

    @property
    def flats(self) -> bool:
        return self.tonic in (_FLAT_MAJORS if self.mode == "major" else _FLAT_MINORS)

    @property
    def name(self) -> str:
        suffix = "m" if self.mode == "minor" else ""
        return f"{root_name(self.tonic, self.flats)}{suffix}"


def estimate_beats(
    y: np.ndarray,
    sr: int,
    hop_length: int = HOP_LENGTH,
    beats_per_measure: int = DEFAULT_BEATS_PER_MEASURE,
) -> BeatGrid:
    import librosa

    tempo_raw, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="frames"
    )
    tempo = float(np.atleast_1d(tempo_raw)[0])
    duration = len(y) / sr

    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()
    if tempo <= 0:
        tempo = 120.0
    if len(beats) < beats_per_measure:
        step = 60.0 / tempo
        beats = [i * step for i in range(int(duration / step) + 1)]

    downbeats = _pick_downbeats(y, sr, beats, hop_length, beats_per_measure)
    return BeatGrid(
        tempo=tempo,
        beats=beats,
        downbeats=downbeats,
        beats_per_measure=beats_per_measure,
        time_signature=f"{beats_per_measure}/4",
    )


def _pick_downbeats(
    y: np.ndarray,
    sr: int,
    beats: list[float],
    hop_length: int,
    beats_per_measure: int,
) -> list[float]:
    import librosa

    if not beats:
        return []
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if onset.size == 0:
        return beats[::beats_per_measure]
    frames = librosa.time_to_frames(np.array(beats), sr=sr, hop_length=hop_length)
    frames = np.clip(frames, 0, onset.size - 1)
    energies = onset[frames]
    phases = [float(energies[phase::beats_per_measure].sum()) for phase in range(beats_per_measure)]
    phase = int(np.argmax(phases))
    return beats[phase::beats_per_measure]


def estimate_key(chroma: np.ndarray) -> KeyEstimate:
    """Krumhansl-Schmuckler key finding on a chromagram."""
    profile = np.asarray(chroma, dtype=np.float64).mean(axis=1)
    if profile.sum() <= 0:
        return KeyEstimate(0, "major", 0.0)
    profile = profile / profile.sum()

    scores: list[tuple[float, int, str]] = []
    for mode, template in (("major", KRUMHANSL_MAJOR), ("minor", KRUMHANSL_MINOR)):
        template = template / template.sum()
        for tonic in range(12):
            rotated = np.roll(template, tonic)
            scores.append((_cosine(profile, rotated), tonic, mode))
    scores.sort(reverse=True)
    best, tonic, mode = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    confidence = 0.0 if best <= 0 else float(np.clip((best - runner_up) / best, 0.0, 1.0))
    return KeyEstimate(tonic, mode, confidence)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)
