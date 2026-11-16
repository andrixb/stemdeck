"""Snap chord segments onto beats and derive measure positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..transcribe.events import ChordEvent


@dataclass(frozen=True, slots=True)
class MeasureChord:
    measure: int
    beat: float
    root: int
    quality: str
    symbol: str


def place_chords(
    chords: list[ChordEvent],
    beat_times: list[float],
    beats_per_measure: int = 4,
    min_beats: float = 0.75,
) -> list[MeasureChord]:
    """Map time-based chords to measure/beat positions.

    Boundaries are snapped to the nearest beat, segments shorter than
    ``min_beats`` are merged into their predecessor, and repeated symbols are
    collapsed, so the resulting chart reads cleanly.
    """
    if not chords or not beat_times:
        return []

    beats = np.asarray(beat_times, dtype=float)
    beat_length = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5

    snapped: list[tuple[int, float, int, str]] = []
    for chord in chords:
        beat_index = int(np.argmin(np.abs(beats - chord.start)))
        if snapped and chord.end - chord.start < beat_length * min_beats:
            continue
        snapped.append((beat_index, float(beats[beat_index]), chord.root, chord.quality))

    collapsed: list[tuple[int, float, int, str]] = []
    for entry in snapped:
        if collapsed and (collapsed[-1][2], collapsed[-1][3]) == (entry[2], entry[3]):
            continue
        collapsed.append(entry)

    return [
        MeasureChord(
            measure=index // beats_per_measure,
            beat=float(index % beats_per_measure),
            root=root,
            quality=quality,
            symbol=_symbol(root, quality),
        )
        for index, _time, root, quality in collapsed
    ]


def _symbol(root: int, quality: str) -> str:
    from ..transcribe.events import chord_symbol

    return chord_symbol(root, quality)
