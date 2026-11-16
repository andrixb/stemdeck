"""Build and write a lead-sheet score with MusicXML chord symbols."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import music21 as m21

from ..quantize.chords_to_measures import MeasureChord
from ..transcribe.events import NoteEvent, root_name

KIND_MAP = {
    "": "major",
    "m": "minor",
    "7": "dominant-seventh",
    "maj7": "major-seventh",
    "m7": "minor-seventh",
    "dim": "diminished",
    "aug": "augmented",
    "sus4": "suspended-fourth",
}


def build_score(
    notes: Sequence[NoteEvent],
    measure_chords: Sequence[MeasureChord],
    *,
    tempo: float,
    time_signature: tuple[int, int] = (4, 4),
    key: tuple[int, str] = (0, "major"),
    flats: bool = False,
    title: str = "Untitled",
) -> m21.stream.Score:
    spq = 60.0 / tempo
    beats_per_measure, beat_unit = time_signature

    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = ""

    part = m21.stream.Part()
    part.partName = "Melody"
    part.insert(0, m21.meter.TimeSignature(f"{beats_per_measure}/{beat_unit}"))

    tonic = m21.pitch.Pitch(root_name(key[0], flats))
    part.insert(0, m21.key.KeySignature(m21.key.Key(tonic, key[1]).sharps))
    part.insert(0, m21.tempo.MetronomeMark(number=tempo))

    for offset, duration, note in _layout(notes, spq):
        element = m21.note.Note(note.pitch)
        element.volume.velocity = int(note.velocity)
        element.quarterLength = duration
        part.insert(offset, element)

    for chord in measure_chords:
        if chord.quality == "N" or chord.quality not in KIND_MAP:
            continue
        root = m21.pitch.Pitch(root_name(chord.root, flats))
        symbol = m21.harmony.ChordSymbol(f"{root.name}{chord.quality}")
        offset = chord.measure * beats_per_measure + chord.beat
        part.insert(offset, symbol)

    score.insert(0, part)
    return score


def _layout(
    notes: Sequence[NoteEvent],
    seconds_per_quarter: float,
) -> list[tuple[float, float, NoteEvent]]:
    ordered = sorted(notes, key=lambda n: n.start)
    layout: list[tuple[float, float, NoteEvent]] = []
    for index, note in enumerate(ordered):
        offset = max(0.0, round(note.start / seconds_per_quarter * 4) / 4)
        end = offset + max(0.25, round(note.duration / seconds_per_quarter * 4) / 4)
        if index + 1 < len(ordered):
            next_offset = max(0.0, round(ordered[index + 1].start / seconds_per_quarter * 4) / 4)
            end = min(end, next_offset)
        layout.append((offset, max(0.25, end - offset), note))
    return layout


def write_musicxml(score: m21.stream.Score, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(path))
    return path


def write_midi(score: m21.stream.Score, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    copy.deepcopy(score).write("midi", fp=str(path))
    return path
