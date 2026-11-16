"""Lead-sheet transcription from a finished song's stems.

A lead sheet is one melody line with chord symbols above it, exported as
MusicXML and MIDI at ``<job>/lead-sheet/``. The analysis itself is the
LeadSheet engine vendored under ``app/leadsheet`` (melody transcription,
chord recognition, beat/key estimation and MusicXML/MIDI export); this module
is the bridge between StemDeck's separated stems and that engine.

Running on the stems the song already has, rather than on its mix, is the
whole point: melody comes from the pitched stems the user selects, chords from
the summed pitched stems (bass, guitar, piano, other), with drums kept out of
the harmony. The full mix is still the reference for tempo and for the final
audio/score alignment.

The run is a post-hoc action on a "done" song, like a stem fill or a vocal
split. It writes into a staging folder first and swaps it in only on success,
so a failed or interrupted run never takes a previous lead sheet with it.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

from app.core.config import TIMEOUT_FFMPEG, ffmpeg_executable
from app.core.models import Job
from app.core.stem_files import decoded_wav, stem_exists, stem_path
from app.leadsheet.config import SAMPLE_RATE
from app.leadsheet.pipeline import PipelineError
from app.leadsheet.pipeline import run as run_engine
from app.leadsheet.separate.demucs import SeparationUnavailable, harmonic_mix

logger = logging.getLogger("stemdeck.lead_sheet")

LEAD_SHEET_DIR = "lead-sheet"
# What GET .../lead-sheet/files/{name} may serve. The engine writes exactly
# these; anything else in the folder stays internal.
FILES = (
    "lead-sheet.musicxml",
    "lead-sheet.mid",
    "alignment.json",
    "metadata.json",
)
# Stems summed into the full mix, in Demucs' own order. "original" is not one
# of them: it is the complement of the stems the user picked, so adding it
# would double-count part of the song.
_MIX_STEMS = ("vocals", "drums", "bass", "other", "piano", "guitar")
# Drums are not a melodic source; every other supported separated stem can
# contribute a score part.
MELODY_STEMS = (
    "vocals",
    "lead_vocals",
    "backing_vocals",
    "bass",
    "guitar",
    "piano",
    "other",
)
_HARMONIC_STEMS = ("bass", "other", "piano", "guitar")


class LeadSheetError(RuntimeError):
    """The lead sheet cannot be made; the message is safe to show."""


class LeadSheetCancelled(Exception):
    """The run was asked to stop by DELETE .../lead-sheet."""


def output_dir(job_dir: Path) -> Path:
    return job_dir / LEAD_SHEET_DIR


def read_metadata(job_dir: Path) -> dict | None:
    """The last successful run's metadata, or None when there is none."""
    path = output_dir(job_dir) / "metadata.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def existing_files(job_dir: Path) -> list[str]:
    folder = output_dir(job_dir)
    return [name for name in FILES if (folder / name).is_file()]


def available_melody_stems(job_dir: Path) -> list[str]:
    """Return selectable pitched stems that exist for this song."""
    stems_dir = job_dir / "stems"
    return [name for name in MELODY_STEMS if stem_exists(stems_dir, name)]


def _prepare_stems(
    job_dir: Path, melody_stems: list[str] | None = None
) -> tuple[dict[str, Path], dict[str, Path]]:
    """The stems on disk and the selected melody stems, or raise.

    MP3 songs are decoded to WAV first (decoded_wav), because the engine's
    analysis reads through librosa/soundfile and must not depend on which
    codec the song happens to keep.
    """
    stems_dir = job_dir / "stems"
    names = [name for name in _MIX_STEMS if stem_exists(stems_dir, name)]
    if not names:
        raise LeadSheetError("this song has no stems to transcribe")
    available = available_melody_stems(job_dir)
    if melody_stems is None:
        selected = next((name for name in ("vocals", "lead_vocals") if name in available), None)
        # Preserve the old no-selection API behavior for existing clients.
        return (
            {name: stem_path(stems_dir, name) for name in names},
            {selected: stem_path(stems_dir, selected)} if selected else {},
        )
    if not melody_stems:
        raise LeadSheetError("select at least one stem to transcribe")
    if len(set(melody_stems)) != len(melody_stems):
        raise LeadSheetError("a stem may only be selected once")
    unsupported = [name for name in melody_stems if name not in MELODY_STEMS]
    if unsupported:
        raise LeadSheetError(f"unsupported melody stem: {unsupported[0]}")
    unavailable = [name for name in melody_stems if name not in available]
    if unavailable:
        raise LeadSheetError(f"selected stem is not available: {unavailable[0]}")
    return (
        {name: stem_path(stems_dir, name) for name in names},
        {name: stem_path(stems_dir, name) for name in melody_stems},
    )


def _sum_stems(inputs: list[Path], dest: Path, label: str) -> Path:
    """Sum stems to a mono WAV at the engine's sample rate with ffmpeg."""
    if not inputs:
        raise LeadSheetError(f"no {label} stems to mix")
    cmd = [ffmpeg_executable(), "-y", "-nostdin", "-loglevel", "error"]
    for path in inputs:
        cmd += ["-i", str(path)]
    if len(inputs) == 1:
        cmd += ["-map", "0:a:0"]
    else:
        graph = "".join(f"[{index}:a]" for index in range(len(inputs)))
        cmd += [
            "-filter_complex",
            f"{graph}amix=inputs={len(inputs)}:normalize=0",
        ]
    cmd += [
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    try:
        proc = subprocess.run(  # noqa: S603 -- list args, no shell, trusted ffmpeg
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_FFMPEG,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeadSheetError(f"could not mix the {label} stems: {exc}") from exc
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        tail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:]
        raise LeadSheetError(tail[0] if tail else f"could not mix the {label} stems")
    return dest


def generate(
    job: Job,
    job_dir: Path,
    *,
    tempo: float | None = None,
    beats_per_measure: int = 4,
    melody_stems: list[str] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    """Make the lead sheet for ``job`` and swap it into ``lead-sheet/``.

    Blocking; the caller runs it in a thread. Raises LeadSheetCancelled when
    ``cancel`` is set, and LeadSheetError/PipelineError for anything the user
    can act on. Returns the result metadata.
    """
    stems, selected_melody_stems = _prepare_stems(job_dir, melody_stems)
    staging = job_dir / f".lead-sheet-{uuid.uuid4().hex[:8]}"
    staging.mkdir()

    def report(value: float, message: str) -> None:
        if cancel is not None and cancel.is_set():
            raise LeadSheetCancelled()
        if progress is not None:
            progress(max(0.0, min(1.0, value)), message)

    try:
        report(0.02, "Preparing stems...")
        with ExitStack() as stack:
            stem_paths = {**stems, **selected_melody_stems}
            decoded = {
                name: stack.enter_context(decoded_wav(path)) for name, path in stem_paths.items()
            }
            decoded_melodies = {name: decoded[name] for name in selected_melody_stems}

            report(0.08, "Mixing stems...")
            mix = _sum_stems(
                [decoded[name] for name in _MIX_STEMS if name in decoded],
                staging / "mix.wav",
                "mix",
            )

            harmonic: Path | None = None
            harmonic_inputs = [decoded[name] for name in _HARMONIC_STEMS if name in decoded]
            if harmonic_inputs:
                try:
                    harmonic = harmonic_mix(
                        {name: decoded[name] for name in _HARMONIC_STEMS if name in decoded},
                        staging / "harmonic.wav",
                        SAMPLE_RATE,
                    )
                except SeparationUnavailable as exc:
                    logger.info("[%s] harmonic mix unavailable: %s", job.id, exc)

            def engine_progress(value: float, message: str) -> None:
                # The last 5% is this module's own save step, reported below.
                report(0.15 + 0.8 * value, message)

            run_engine(
                str(mix),
                staging,
                separate=False,
                title=job.title or "Untitled",
                tempo=tempo or job.bpm,
                beats_per_measure=beats_per_measure,
                stems=decoded,
                melody_stems=decoded_melodies or None,
                harmonic=harmonic,
                progress=engine_progress,
            )

        metadata = _read_staging_metadata(staging)
        metadata["source"] = job.source_url or job.title or ""
        metadata["generated_by"] = "stemdeck"
        metadata["melody_stems"] = list(decoded_melodies) or ["mix"]
        metadata["available_melody_stems"] = available_melody_stems(job_dir)
        # The stems came from the library, not from a Demucs run inside this
        # transcription; say so rather than let the engine claim otherwise.
        metadata.setdefault("engines", {})["separation"] = "stemdeck-stems"
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        report(0.99, "Saving lead sheet...")
        final = output_dir(job_dir)
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
        report(1.0, "Lead sheet ready")
        return metadata
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _read_staging_metadata(staging: Path) -> dict:
    path = staging / "metadata.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LeadSheetError("the transcription produced no metadata") from exc
    if not isinstance(data, dict):
        raise LeadSheetError("the transcription produced no metadata")
    return data


def remove(job_dir: Path) -> None:
    """Delete the lead sheet and everything the engine wrote with it."""
    shutil.rmtree(output_dir(job_dir), ignore_errors=True)


__all__ = [
    "FILES",
    "LEAD_SHEET_DIR",
    "MELODY_STEMS",
    "LeadSheetCancelled",
    "LeadSheetError",
    "PipelineError",
    "available_melody_stems",
    "existing_files",
    "generate",
    "output_dir",
    "read_metadata",
    "remove",
]
