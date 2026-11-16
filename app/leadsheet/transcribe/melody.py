"""Melody extraction: Basic Pitch when installed, pyin otherwise."""

from __future__ import annotations

import numpy as np

from ..config import HOP_LENGTH, MELODY_FMAX, MELODY_FMIN, MIN_NOTE_SECONDS
from .events import NoteEvent

BASS_FMIN = 41.2
BASS_FMAX = 392.0
BASS_FRAME_LENGTH = 4096


def transcribe_melody(
    y: np.ndarray,
    sr: int,
    hop_length: int = HOP_LENGTH,
    *,
    instrument: str | None = None,
) -> tuple[list[NoteEvent], str]:
    """Return ``(notes, engine_name)`` for a selected stem."""
    if instrument is not None and instrument.casefold() == "bass":
        return (
            _pyin(
                y,
                sr,
                hop_length,
                fmin=BASS_FMIN,
                fmax=BASS_FMAX,
                frame_length=BASS_FRAME_LENGTH,
            ),
            "pyin-bass",
        )
    notes = _basic_pitch(y, sr)
    if notes:
        return notes, "basic-pitch"
    return _pyin(y, sr, hop_length), "pyin"


def _basic_pitch(y: np.ndarray, sr: int) -> list[NoteEvent]:
    try:
        from basic_pitch.inference import predict
    except Exception:
        return []
    try:
        _, _, note_events = predict(
            y,
            sr,
            minimum_note_length=80,
            minimum_frequency=MELODY_FMIN,
            maximum_frequency=MELODY_FMAX,
        )
    except Exception:
        return []
    notes = []
    for start, end, pitch, amplitude, _bend in note_events:
        if end - start < MIN_NOTE_SECONDS:
            continue
        notes.append(
            NoteEvent(
                start=float(start),
                end=float(end),
                pitch=int(round(float(pitch))),
                velocity=int(np.clip(amplitude * 127, 1, 127)),
                confidence=float(np.clip(amplitude, 0.0, 1.0)),
            )
        )
    return notes


def _pyin(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    fmin: float = MELODY_FMIN,
    fmax: float = MELODY_FMAX,
    frame_length: int = 2048,
) -> list[NoteEvent]:
    import librosa

    if y.size == 0:
        return []
    f0, voiced, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    midi = librosa.hz_to_midi(f0)
    return _segments_to_notes(midi, voiced, times)


def _segments_to_notes(
    midi: np.ndarray,
    voiced: np.ndarray,
    times: np.ndarray,
) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    current_pitch: int | None = None
    current_start = 0.0

    for time, value, is_voiced in zip(times, midi, voiced, strict=True):
        pitch = int(round(float(value))) if is_voiced and np.isfinite(value) else None
        if pitch != current_pitch:
            _close(notes, current_pitch, current_start, float(time))
            current_pitch = pitch
            current_start = float(time)
    if len(times):
        _close(notes, current_pitch, current_start, float(times[-1]))

    return notes


def _close(notes: list[NoteEvent], pitch: int | None, start: float, end: float) -> None:
    if pitch is not None and end - start >= MIN_NOTE_SECONDS:
        notes.append(NoteEvent(start, end, pitch, velocity=90, confidence=0.5))
