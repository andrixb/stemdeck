"""End-to-end orchestration: audio in, lead sheet + alignment out."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .align.dtw import align, write_alignment
from .analyze.beats import estimate_beats, estimate_key
from .config import HOP_LENGTH, SAMPLE_RATE
from .export.musicxml import build_score, write_midi, write_musicxml
from .ingest.audio import audio_duration, extract_fragment, load_audio
from .ingest.youtube import download_audio, is_url
from .quantize.chords_to_measures import place_chords
from .quantize.grid import quantize_notes
from .separate.demucs import SeparationUnavailable, harmonic_mix, separate_stems
from .transcribe.chords import chords_from_chroma
from .transcribe.events import root_name
from .transcribe.melody import transcribe_melody

logger = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


class PipelineError(RuntimeError):
    pass


def run(
    source: str,
    out_dir: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    separate: bool = True,
    title: str | None = None,
    tempo: float | None = None,
    beats_per_measure: int = 4,
    progress: Progress | None = None,
    stems: dict[str, Path] | None = None,
    melody_stems: dict[str, Path] | None = None,
    harmonic: Path | None = None,
) -> dict[str, Any]:
    """Transcribe ``source`` to a lead sheet.

    ``stems`` lets a caller supply already separated audio (for example the
    stems a library keeps for a track) instead of running Demucs again.
    ``melody_stems`` selects named sources to transcribe as separate score
    parts. When omitted, the legacy single "Melody" source is used.
    ``harmonic``, when given, is the summed pitched audio used for chords.
    ``source`` is still the full mix, used for tempo, beats and alignment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    def report(value: float, message: str) -> None:
        logger.debug("[%5.1f%%] %s", value * 100, message)
        if progress is not None:
            progress(value, message)

    report(0.02, "resolving source")
    raw = (
        download_audio(source, out_dir / "download")
        if is_url(source)
        else Path(source).expanduser()
    )
    if not raw.exists():
        raise PipelineError(f"source not found: {raw}")

    report(0.08, "extracting fragment")
    fragment = extract_fragment(raw, out_dir / "fragment.wav", start, end, SAMPLE_RATE)
    duration = audio_duration(fragment)

    vocal_path = fragment
    harmonic_path = fragment
    stem_files: dict[str, str] = {}

    if stems:
        report(0.18, "preparing stems")
        stem_files = {name: str(Path(path)) for name, path in stems.items()}
        vocal_path = Path(stems.get("vocals", fragment))
        if harmonic is not None:
            harmonic_path = Path(harmonic)
        else:
            try:
                harmonic_path = harmonic_mix(dict(stems), out_dir / "harmonic.wav", SAMPLE_RATE)
            except SeparationUnavailable as exc:
                warnings.append(f"harmonic mix skipped: {exc}")
    elif separate:
        report(0.18, "separating stems (demucs)")
        try:
            separated = separate_stems(fragment, out_dir / "stems")
            stem_files = {name: _rel(path, out_dir) for name, path in separated.items()}
            vocal_path = separated.get("vocals", fragment)
            harmonic_path = harmonic_mix(separated, out_dir / "harmonic.wav", SAMPLE_RATE)
        except SeparationUnavailable as exc:
            warnings.append(f"stem separation skipped: {exc}")

    report(0.32, "loading audio")
    y_mix = load_audio(fragment, SAMPLE_RATE)
    y_harmonic = load_audio(harmonic_path, SAMPLE_RATE) if harmonic_path != fragment else y_mix

    report(0.4, "estimating tempo, beats and key")
    grid = estimate_beats(y_mix, SAMPLE_RATE, beats_per_measure=beats_per_measure)
    if tempo is not None:
        grid = replace(grid, tempo=float(tempo))

    if melody_stems is None:
        y_vocal = load_audio(vocal_path, SAMPLE_RATE) if vocal_path != fragment else y_mix
        sources = {"Melody": y_vocal}
    else:
        sources = {
            _part_name(name): load_audio(Path(path), SAMPLE_RATE)
            for name, path in melody_stems.items()
        }
    melody_parts = {}
    melody_engines = {}
    note_entries = []
    for index, (part_name, y_part) in enumerate(sources.items()):
        report(
            0.48 + 0.1 * index / max(1, len(sources)),
            f"transcribing {part_name}",
        )
        melody, melody_engine = transcribe_melody(y_part, SAMPLE_RATE)
        part_notes = quantize_notes(melody, grid.tempo)
        melody_parts[part_name] = part_notes
        melody_engines[part_name] = melody_engine
        if not part_notes:
            warnings.append(f"no melody detected in {part_name}; score part is empty")
            continue
        note_entries.extend((note.start, part_name, note) for note in part_notes)

    if not note_entries:
        raise PipelineError(
            "no melody detected in the selected stem; try another stem or a "
            "different section of the song"
        )
    note_entries.sort(key=lambda item: item[0])
    notes = [item[2] for item in note_entries]

    report(0.6, "recognizing chords")
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    chords = chords_from_chroma(chroma, grid.beats, duration, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    key = estimate_key(chroma)
    measure_chords = place_chords(chords, grid.beats, grid.beats_per_measure)

    report(0.72, "building score")
    score_title = title or raw.stem
    score = build_score(
        notes,
        measure_chords,
        tempo=grid.tempo,
        time_signature=(grid.beats_per_measure, 4),
        key=(key.tonic, key.mode),
        flats=key.flats,
        title=score_title,
        melody_parts=melody_parts,
    )
    xml_path = write_musicxml(score, out_dir / "lead-sheet.musicxml")
    midi_path = write_midi(score, out_dir / "lead-sheet.mid")

    report(0.85, "aligning score to audio")
    alignment = align(
        y_mix,
        notes,
        grid.beats,
        grid.tempo,
        sr=SAMPLE_RATE,
        fragment_start=start or 0.0,
        beats_per_measure=grid.beats_per_measure,
    )
    for aligned_note, (_start, part_name, _note) in zip(
        alignment["notes"], note_entries, strict=True
    ):
        aligned_note["part"] = part_name
    alignment_path = write_alignment(alignment, out_dir / "alignment.json")
    if alignment["low_confidence"]:
        warnings.append("audio/score alignment confidence is low; timing may drift")

    measures = max(
        max((chord.measure for chord in measure_chords), default=0) + 1,
        max(1, len(grid.beats) // grid.beats_per_measure),
    )
    metadata: dict[str, Any] = {
        "source": source,
        "title": score_title,
        "duration": round(duration, 3),
        "fragment": {
            "start": start or 0.0,
            "end": end if end is not None else round(duration, 3),
            "duration": round(duration, 3),
        },
        "tempo": round(grid.tempo, 3),
        "beats_per_measure": grid.beats_per_measure,
        "time_signature": grid.time_signature,
        "key": {
            "tonic": key.tonic,
            "mode": key.mode,
            "name": key.name,
            "confidence": round(key.confidence, 3),
        },
        "counts": {
            "notes": len(notes),
            "chords": len(chords),
            "measures": measures,
            "beats": len(grid.beats),
        },
        "melody_stems": (
            list(melody_stems)
            if melody_stems is not None
            else ["vocals"]
            if "vocals" in stem_files
            else ["mix"]
        ),
        "parts": [
            {"name": name, "notes": len(part_notes)} for name, part_notes in melody_parts.items()
        ],
        "engines": {
            "melody": (
                next(iter(melody_engines.values())) if len(melody_engines) == 1 else melody_engines
            ),
            "chords": "chroma-template",
            "separation": "demucs" if stem_files else "none",
            "alignment": alignment["method"],
        },
        "alignment_confidence": alignment["confidence"],
        "chords": [
            {
                "start": round(chord.start, 3),
                "end": round(chord.end, 3),
                "symbol": chord.symbol,
                "confidence": round(chord.confidence, 3),
            }
            for chord in chords
        ],
        "notes": [
            {
                "start": round(note.start, 3),
                "end": round(note.end, 3),
                "duration": round(note.duration, 3),
                "pitch": note.pitch,
                "name": f"{root_name(note.pitch % 12)}{note.pitch // 12 - 1}",
                "part": part_name,
            }
            for _start, part_name, note in note_entries
        ],
        "warnings": warnings,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    files: dict[str, Any] = {
        "audio": _rel(fragment, out_dir),
        "musicxml": _rel(xml_path, out_dir),
        "midi": _rel(midi_path, out_dir),
        "alignment": _rel(alignment_path, out_dir),
        "metadata": _rel(metadata_path, out_dir),
    }
    if harmonic_path != fragment:
        files["harmonic"] = _rel(harmonic_path, out_dir)
    if stem_files:
        files["stems"] = stem_files

    report(1.0, "done")
    return {
        "out_dir": str(out_dir),
        "files": files,
        "metadata": metadata,
        "warnings": warnings,
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _part_name(stem: str) -> str:
    return stem.replace("_", " ").title()


__all__ = ["PipelineError", "run"]
