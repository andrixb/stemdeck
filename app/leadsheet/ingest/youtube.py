"""YouTube ingest via ``yt-dlp``.

Transcripts are meant for local, personal study of material you have the
right to use. Downloading is opt-in and only happens when a URL is passed
explicitly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class IngestError(RuntimeError):
    pass


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def download_audio(url: str, out_dir: Path) -> Path:
    exe = shutil.which("yt-dlp")
    if not exe:
        raise IngestError(
            "yt-dlp not found; install it (`brew install yt-dlp`) or pass a local file"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    template = out_dir / "source.%(ext)s"
    cmd = [
        exe,
        "--no-playlist",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        str(template),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
        raise IngestError(f"yt-dlp failed: {' | '.join(detail)}")

    files = sorted(out_dir.glob("source.*"))
    if not files:
        raise IngestError("yt-dlp produced no output file")
    return files[0]
