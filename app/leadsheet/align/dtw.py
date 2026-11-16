"""Align generated score time to the original audio via chroma DTW."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ALIGN_HOP_LENGTH, SAMPLE_RATE
from ..transcribe.events import NoteEvent

LOW_CONFIDENCE = 0.5


def synthesize(
    notes: list[NoteEvent],
    duration: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render note events as a simple sine tone so chroma can be compared."""
    y = np.zeros(max(1, int(duration * sr)), dtype=np.float32)
    for note in notes:
        start = max(0, int(note.start * sr))
        end = min(len(y), int(note.end * sr))
        if end <= start:
            continue
        t = np.arange(end - start, dtype=np.float32) / sr
        freq = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
        tone = 0.25 * np.sin(2.0 * np.pi * freq * t)
        fade = min(len(tone), int(0.01 * sr))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            tone[:fade] *= ramp
            tone[-fade:] *= ramp[::-1]
        y[start:end] += tone
    return y


def align(
    reference: np.ndarray,
    notes: list[NoteEvent],
    beats: list[float],
    tempo: float,
    *,
    sr: int = SAMPLE_RATE,
    fragment_start: float = 0.0,
    beats_per_measure: int = 4,
    hop_length: int = ALIGN_HOP_LENGTH,
) -> dict[str, Any]:
    duration = float(len(reference)) / sr
    synth = synthesize(notes, duration, sr)
    if not notes or float(np.max(np.abs(synth))) < 1e-6:
        return _linear(notes, beats, tempo, duration, fragment_start, beats_per_measure)

    import librosa

    ref_chroma = librosa.feature.chroma_cens(y=reference, sr=sr, hop_length=hop_length)
    syn_chroma = librosa.feature.chroma_cens(y=synth, sr=sr, hop_length=hop_length)
    if ref_chroma.shape[1] < 2 or syn_chroma.shape[1] < 2 or float(np.max(ref_chroma)) < 1e-6:
        return _linear(notes, beats, tempo, duration, fragment_start, beats_per_measure)

    _cost, path = librosa.sequence.dtw(X=syn_chroma, Y=ref_chroma, metric="cosine")
    path = path[::-1]  # time order: (score frame, audio frame)

    syn_times = librosa.frames_to_time(np.arange(syn_chroma.shape[1]), sr=sr, hop_length=hop_length)
    ref_times = librosa.frames_to_time(np.arange(ref_chroma.shape[1]), sr=sr, hop_length=hop_length)
    score_axis = syn_times[path[:, 0]]
    audio_axis = ref_times[path[:, 1]]

    mean_cost = float(np.mean(_local_costs(syn_chroma, ref_chroma, path)))
    confidence = float(np.clip(np.exp(-mean_cost), 0.0, 1.0))

    beat_times = _map_times(np.asarray(beats, dtype=float), score_axis, audio_axis)
    note_times = _map_times(
        np.asarray([note.start for note in notes], dtype=float), score_axis, audio_axis
    )

    return {
        "version": 1,
        "method": "dtw",
        "confidence": round(confidence, 4),
        "low_confidence": confidence < LOW_CONFIDENCE,
        "fragment_start": fragment_start,
        "duration": round(duration, 4),
        "tempo": tempo,
        "beats_per_measure": beats_per_measure,
        "beats": [
            {
                "index": index,
                "measure": index // beats_per_measure,
                "beat": index % beats_per_measure,
                "score_time": round(float(score_t), 4),
                "time": round(float(audio_t), 4),
            }
            for index, (score_t, audio_t) in enumerate(zip(beats, beat_times, strict=True))
        ],
        "notes": [
            {
                "start": round(note.start, 4),
                "pitch": note.pitch,
                "time": round(float(audio_t), 4),
            }
            for note, audio_t in zip(notes, note_times, strict=True)
        ],
    }


def _local_costs(
    syn_chroma: np.ndarray,
    ref_chroma: np.ndarray,
    path: np.ndarray,
) -> np.ndarray:
    """Per-step cosine distance, unlike the cumulative DTW cost matrix."""
    syn_frames = syn_chroma[:, path[:, 0]].T
    ref_frames = ref_chroma[:, path[:, 1]].T
    numerator = np.sum(syn_frames * ref_frames, axis=1)
    denominator = np.linalg.norm(syn_frames, axis=1) * np.linalg.norm(ref_frames, axis=1)
    similarity = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-9,
    )
    return np.clip(1.0 - similarity, 0.0, 2.0)


def _map_times(
    targets: np.ndarray,
    score_axis: np.ndarray,
    audio_axis: np.ndarray,
) -> np.ndarray:
    if targets.size == 0:
        return np.zeros(0)
    indices = np.searchsorted(score_axis, targets)
    indices = np.clip(indices, 1, max(1, len(score_axis) - 1))
    left = indices - 1
    right = np.minimum(indices, len(score_axis) - 1)
    choose_left = np.abs(score_axis[left] - targets) <= np.abs(score_axis[right] - targets)
    return np.where(choose_left, audio_axis[left], audio_axis[right])


def _linear(
    notes: list[NoteEvent],
    beats: list[float],
    tempo: float,
    duration: float,
    fragment_start: float,
    beats_per_measure: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "method": "linear",
        "confidence": 0.0,
        "low_confidence": True,
        "fragment_start": fragment_start,
        "duration": round(duration, 4),
        "tempo": tempo,
        "beats_per_measure": beats_per_measure,
        "beats": [
            {
                "index": index,
                "measure": index // beats_per_measure,
                "beat": index % beats_per_measure,
                "score_time": round(float(beat), 4),
                "time": round(min(float(beat), duration), 4),
            }
            for index, beat in enumerate(beats)
        ],
        "notes": [
            {
                "start": round(note.start, 4),
                "pitch": note.pitch,
                "time": round(min(note.start, duration), 4),
            }
            for note in notes
        ],
    }


def write_alignment(alignment: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    return path
