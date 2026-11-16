"""Note and chord event types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

NO_CHORD = "N"


def root_name(pitch_class: int, flats: bool = False) -> str:
    names = FLAT_NAMES if flats else SHARP_NAMES
    return names[pitch_class % 12]


def chord_symbol(
    root: int,
    quality: str = "",
    bass: int | None = None,
    flats: bool = False,
) -> str:
    if quality == NO_CHORD:
        return "N.C."
    symbol = f"{root_name(root, flats)}{quality}"
    if bass is not None:
        symbol += f"/{root_name(bass, flats)}"
    return symbol


@dataclass(frozen=True, slots=True)
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: int = 90
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True, slots=True)
class ChordEvent:
    start: float
    end: float
    root: int
    quality: str = ""
    bass: int | None = None
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def symbol(self) -> str:
        return chord_symbol(self.root, self.quality, self.bass)
