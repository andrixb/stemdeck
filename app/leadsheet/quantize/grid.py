"""Rhythm quantization against a constant tempo grid."""

from __future__ import annotations

from ..config import MIN_NOTE_SECONDS
from ..transcribe.events import NoteEvent


def seconds_per_quarter(tempo: float) -> float:
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    return 60.0 / tempo


def quantize_notes(
    notes: list[NoteEvent],
    tempo: float,
    subdivision: int = 4,
    min_duration: float = MIN_NOTE_SECONDS,
) -> list[NoteEvent]:
    """Snap note boundaries to a 1/subdivision-of-a-quarter grid."""
    note_step = seconds_per_quarter(tempo) / subdivision
    snapped: list[NoteEvent] = []
    for note in notes:
        start = round(note.start / note_step) * note_step
        end = round(note.end / note_step) * note_step
        if end - start < min_duration:
            continue
        snapped.append(NoteEvent(start, end, note.pitch, note.velocity, note.confidence))
    return monophonic(snapped, min_duration=min_duration)


def monophonic(
    notes: list[NoteEvent],
    min_duration: float = MIN_NOTE_SECONDS,
) -> list[NoteEvent]:
    """Trim overlaps so the melody reads as a single voice, dropping scraps."""
    ordered = sorted(notes, key=lambda n: (n.start, -n.pitch))
    result: list[NoteEvent] = []
    for index, note in enumerate(ordered):
        end = note.end
        if index + 1 < len(ordered):
            end = min(end, ordered[index + 1].start)
        if end - note.start < min_duration:
            continue
        if result and result[-1].pitch == note.pitch and note.start <= result[-1].end + 1e-6:
            previous = result.pop()
            end = max(previous.end, end)
            note = NoteEvent(
                previous.start, end, previous.pitch, previous.velocity, previous.confidence
            )
        result.append(NoteEvent(note.start, end, note.pitch, note.velocity, note.confidence))
    return result


def time_to_quarter(seconds: float, tempo: float, origin: float = 0.0) -> float:
    return (seconds - origin) / seconds_per_quarter(tempo)


def quarter_to_time(quarters: float, tempo: float, origin: float = 0.0) -> float:
    return origin + quarters * seconds_per_quarter(tempo)
