"""Chord recognition by chroma template matching.

M0 engine. The vocabulary is deliberately small (triads, sevenths, sus4) and
is meant to be swapped for a learned recognizer (BTC-style) behind the same
``chords_from_chroma`` interface.
"""

from __future__ import annotations

import numpy as np

from ..config import HOP_LENGTH
from .events import NO_CHORD, ChordEvent

QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus4": (0, 5, 7),
}

_SILENCE_THRESHOLD = 0.12
_MIN_SEGMENT_SECONDS = 0.05


def build_templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows: list[np.ndarray] = []
    labels: list[tuple[int, str]] = []
    for root in range(12):
        for quality, intervals in QUALITY_INTERVALS.items():
            vec = np.zeros(12, dtype=np.float64)
            for interval in intervals:
                vec[(root + interval) % 12] = 1.0
            rows.append(vec)
            labels.append((root, quality))
    return np.vstack(rows), labels


def detect_chords(
    y: np.ndarray,
    sr: int,
    beat_times: list[float],
    hop_length: int = HOP_LENGTH,
) -> list[ChordEvent]:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    duration = float(len(y)) / sr
    return chords_from_chroma(chroma, beat_times, duration, sr=sr, hop_length=hop_length)


def chords_from_chroma(
    chroma: np.ndarray,
    beat_times: list[float],
    duration: float,
    sr: int = 22_050,
    hop_length: int = HOP_LENGTH,
) -> list[ChordEvent]:
    import librosa

    if chroma.size == 0:
        return []
    times = librosa.times_like(chroma[0], sr=sr, hop_length=hop_length)
    boundaries = [*beat_times, max(duration, float(times[-1]))]
    templates, labels = build_templates()

    events: list[ChordEvent] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start < _MIN_SEGMENT_SECONDS:
            continue
        mask = (times >= start) & (times < end)
        frame = (
            np.median(chroma[:, mask], axis=1)
            if mask.any()
            else chroma[:, min(int(np.searchsorted(times, start)), chroma.shape[1] - 1)]
        )
        root, quality, confidence = _best_match(frame, templates, labels)
        events.append(ChordEvent(start, end, root, quality, None, confidence))
    return _merge(events)


def _best_match(
    frame: np.ndarray,
    templates: np.ndarray,
    labels: list[tuple[int, str]],
) -> tuple[int, str, float]:
    energy = float(np.linalg.norm(frame))
    if energy < 1e-6 or float(np.max(frame)) < _SILENCE_THRESHOLD:
        return 0, NO_CHORD, 0.0
    template_norms = np.linalg.norm(templates, axis=1)
    scores = templates @ frame / (template_norms * energy)
    best = int(np.argmax(scores))
    root, quality = labels[best]
    return root, quality, float(np.clip(scores[best], 0.0, 1.0))


def _merge(events: list[ChordEvent]) -> list[ChordEvent]:
    merged: list[ChordEvent] = []
    for event in events:
        if merged and _same_chord(merged[-1], event):
            previous = merged[-1]
            merged[-1] = ChordEvent(
                previous.start,
                event.end,
                previous.root,
                previous.quality,
                previous.bass,
                max(previous.confidence, event.confidence),
            )
        else:
            merged.append(event)
    return merged


def _same_chord(a: ChordEvent, b: ChordEvent) -> bool:
    return a.root == b.root and a.quality == b.quality and a.bass == b.bass
