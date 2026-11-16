"""Build and write a lead-sheet score with MusicXML chord symbols."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
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
    melody_parts: Mapping[str, Sequence[NoteEvent]] | None = None,
) -> m21.stream.Score:
    spq = 60.0 / tempo
    beats_per_measure, beat_unit = time_signature

    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = ""

    parts = melody_parts if melody_parts is not None else {"Melody": notes}
    chord_part = next((name for name, part_notes in parts.items() if part_notes), None)
    if chord_part is None:
        chord_part = next(iter(parts), None)
    for part_name, part_notes in parts.items():
        part = m21.stream.Part()
        part.partName = part_name
        if part_name.casefold() == "bass":
            part.insert(0, m21.instrument.AcousticBass())
        part.insert(0, m21.meter.TimeSignature(f"{beats_per_measure}/{beat_unit}"))

        tonic = m21.pitch.Pitch(root_name(key[0], flats))
        part.insert(0, m21.key.KeySignature(m21.key.Key(tonic, key[1]).sharps))
        part.insert(0, m21.tempo.MetronomeMark(number=tempo))

        melody = m21.stream.Voice()
        melody.id = 1
        for offset, duration, note in _layout(part_notes, spq):
            element = m21.note.Note(note.pitch)
            element.volume.velocity = int(note.velocity)
            element.quarterLength = duration
            melody.insert(offset, element)
        if melody:
            part.insert(0, melody)

        if part_name == chord_part:
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


def write_midi(
    score: m21.stream.Score,
    path: Path,
    *,
    chords: Sequence[Mapping[str, float | str]] | None = None,
    tempo: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    marks = list(score.recurse().getElementsByClass(m21.tempo.MetronomeMark))
    if tempo is None and marks:
        tempo = float(marks[0].number)
    if chords is None and tempo:
        chords = [
            {
                "start": float(chord.getOffsetInHierarchy(score)) * 60.0 / tempo,
                "symbol": chord.figure,
            }
            for part in score.parts
            for chord in part.recurse().getElementsByClass(m21.harmony.ChordSymbol)
        ]
    if chords and tempo:
        _write_midi_with_chord_markers(score, path, chords, tempo)
    else:
        copy.deepcopy(score).write("midi", fp=str(path))
    return path


def replace_chord_symbols(
    musicxml_path: Path,
    midi_path: Path,
    chords: Sequence[Mapping[str, float | str]],
    *,
    tempo: float,
) -> None:
    score = m21.converter.parse(str(musicxml_path))
    parts = list(score.parts)
    if not parts:
        raise ValueError("the score contains no parts")

    original_chord_part = next(
        (part for part in parts if part.recurse().getElementsByClass(m21.harmony.ChordSymbol)),
        parts[0],
    )
    for part in parts:
        for chord in list(part.recurse().getElementsByClass(m21.harmony.ChordSymbol)):
            if chord.activeSite is not None:
                chord.activeSite.remove(chord)

    measures = list(original_chord_part.getElementsByClass(m21.stream.Measure))
    for item in sorted(chords, key=lambda chord: float(chord["start"])):
        symbol = str(item["symbol"])
        if symbol == "N.C.":
            continue
        offset = float(item["start"]) * tempo / 60.0
        measure = next(
            (
                candidate
                for candidate in measures
                if candidate.offset <= offset < candidate.offset + candidate.duration.quarterLength
            ),
            None,
        )
        if measure is not None:
            measure.insert(offset - measure.offset, m21.harmony.ChordSymbol(symbol))

    score.write("musicxml", fp=str(musicxml_path))
    _write_midi_with_chord_markers(score, midi_path, chords, tempo)


def _write_midi_with_chord_markers(
    score: m21.stream.Score,
    path: Path,
    chords: Sequence[Mapping[str, float | str]],
    tempo: float,
) -> None:
    midi_file = m21.midi.translate.streamToMidiFile(copy.deepcopy(score))
    track = m21.midi.MidiTrack(len(midi_file.tracks))

    delta = m21.midi.DeltaTime(track)
    delta.time = 0
    track.events.append(delta)
    name = m21.midi.MidiEvent(track)
    name.type = m21.midi.MetaEvents.SEQUENCE_TRACK_NAME
    name.data = "Chords"
    track.events.append(name)

    previous_tick = 0
    for item in sorted(chords, key=lambda chord: float(chord["start"])):
        symbol = str(item["symbol"])
        if symbol == "N.C.":
            continue
        tick = round(float(item["start"]) * tempo / 60.0 * midi_file.ticksPerQuarterNote)
        delta = m21.midi.DeltaTime(track)
        delta.time = tick - previous_tick
        marker = m21.midi.MidiEvent(track)
        marker.type = m21.midi.MetaEvents.MARKER
        marker.data = symbol
        track.events.extend((delta, marker))
        previous_tick = tick

    delta = m21.midi.DeltaTime(track)
    delta.time = 0
    end = m21.midi.MidiEvent(track)
    end.type = m21.midi.MetaEvents.END_OF_TRACK
    end.data = b""
    track.events.extend((delta, end))
    midi_file.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.open(str(path), "wb")
    try:
        midi_file.write()
    finally:
        midi_file.close()
