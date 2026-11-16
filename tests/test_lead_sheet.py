"""Lead-sheet transcription from a finished song's stems.

The engine under app/leadsheet is covered by its own tests upstream; what
matters here is the bridge: that StemDeck's stems become a MusicXML/MIDI pair
in the song's folder, that a run with no stems fails with a message worth
showing, and that a cancelled run leaves neither a partial folder nor a
staging directory behind.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app.api.jobs as jobs_mod
from app.core.models import Job
from app.core.registry import _jobs
from app.pipeline import lead_sheet
from tests.ffmpeg_probe import skip_without_ffmpeg

SR = 44100
DURATION = 12.0


def _tone(midi: int, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return 0.4 * np.sin(2 * np.pi * 440.0 * 2 ** ((midi - 69) / 12) * t)


def _write_stems(folder) -> None:
    """A vocal melody over a C major triad, separated like a real song: the
    melody in vocals, the harmony in the pitched stems, and drums that must
    stay out of chord recognition."""
    stems = folder / "stems"
    stems.mkdir()
    melody = np.concatenate([_tone(midi) for midi in (60, 64, 67, 69, 67, 64)])
    melody = np.tile(melody, 4)[: int(SR * DURATION)]
    harmony = (
        np.concatenate([_tone(midi, DURATION) for midi in (48, 60, 64, 67)])
        .reshape(4, -1)
        .sum(axis=0)
    )
    drums = np.zeros(int(SR * DURATION))
    drums[:: int(SR * 0.25)] = 0.4

    sf.write(stems / "vocals.wav", melody, SR, subtype="PCM_16")
    sf.write(stems / "bass.wav", _tone(36, DURATION), SR, subtype="PCM_16")
    sf.write(stems / "other.wav", harmony, SR, subtype="PCM_16")
    sf.write(stems / "piano.wav", harmony * 0.5, SR, subtype="PCM_16")
    sf.write(stems / "guitar.wav", harmony * 0.3, SR, subtype="PCM_16")
    sf.write(stems / "drums.wav", drums, SR, subtype="PCM_16")


def _job() -> Job:
    return Job(id="test", title="Synthetic Test", bpm=120, source_url="local:test")


def test_generate_writes_a_lead_sheet_beside_the_song(tmp_path):
    skip_without_ffmpeg()
    _write_stems(tmp_path)

    result = lead_sheet.generate(_job(), tmp_path)

    assert set(lead_sheet.existing_files(tmp_path)) == set(lead_sheet.FILES)
    assert (tmp_path / "lead-sheet" / "lead-sheet.musicxml").stat().st_size > 100
    assert (tmp_path / "lead-sheet" / "lead-sheet.mid").stat().st_size > 0
    assert result["counts"]["notes"] > 0
    assert result["notes"][0]["name"].endswith(("3", "4", "5"))
    assert result["key"]["name"]
    assert result["source"] == "local:test"
    assert lead_sheet.read_metadata(tmp_path)["counts"] == result["counts"]


def test_generate_refuses_a_song_with_no_stems(tmp_path):
    (tmp_path / "stems").mkdir()
    with pytest.raises(lead_sheet.LeadSheetError):
        lead_sheet.generate(_job(), tmp_path)
    assert not lead_sheet.output_dir(tmp_path).exists()


def test_a_cancelled_run_leaves_nothing_behind(tmp_path):
    skip_without_ffmpeg()
    _write_stems(tmp_path)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(lead_sheet.LeadSheetCancelled):
        lead_sheet.generate(_job(), tmp_path, cancel=cancel)

    assert not lead_sheet.output_dir(tmp_path).exists()
    assert not list(tmp_path.glob(".lead-sheet-*"))


# ── the API around it ──


@pytest.fixture
def client():
    with patch("app.api.jobs.jobqueue.enqueue", lambda job_id: None):
        from app.main import app

        with TestClient(app) as c:
            yield c


def _api_job(**fields) -> Job:
    job = Job(id="abcdef123456", status="done", title="API Test", duration_sec=12.0, **fields)
    _jobs[job.id] = job
    stems = jobs_mod.JOBS_DIR / job.id / "stems"
    stems.mkdir(parents=True)
    (stems / "vocals.wav").write_bytes(b"RIFF")
    (stems / "bass.wav").write_bytes(b"RIFF")
    return job


def _stub_generate(monkeypatch, calls: list | None = None) -> None:
    def fake(job, job_dir, **kwargs):
        if calls is not None:
            calls.append(job.id)
        out = lead_sheet.output_dir(job_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "lead-sheet.musicxml").write_text("<score-partwise/>", encoding="utf-8")
        (out / "lead-sheet.mid").write_bytes(b"MThd")
        (out / "alignment.json").write_text("{}", encoding="utf-8")
        (out / "metadata.json").write_text(
            json.dumps({"counts": {"notes": 1}, "key": {"name": "C"}}), encoding="utf-8"
        )
        return {"counts": {"notes": 1}, "key": {"name": "C"}}

    monkeypatch.setattr(jobs_mod.lead_sheet, "generate", fake)


def _wait(client, url: str) -> dict:
    deadline = time.monotonic() + 20
    while True:
        body = client.get(url).json()
        if body["status"] != "running" or time.monotonic() > deadline:
            return body
        time.sleep(0.05)


API_ID = "abcdef123456"
URL = f"/api/jobs/{API_ID}/lead-sheet"


def test_post_runs_the_engine_and_serves_its_files(client, monkeypatch):
    _api_job()
    _stub_generate(monkeypatch)
    assert client.get(URL).json()["status"] == "none"

    r = client.post(URL)
    assert r.status_code == 202
    body = _wait(client, URL)
    assert body["status"] == "done" and body["has_lead_sheet"]
    assert set(body["files"]) == set(lead_sheet.FILES)

    file_r = client.get(body["files"]["lead-sheet.musicxml"])
    assert file_r.status_code == 200
    assert "musicxml" in file_r.headers["content-type"]


def test_a_second_post_keeps_the_existing_lead_sheet(client, monkeypatch):
    _api_job()
    calls: list[str] = []
    _stub_generate(monkeypatch, calls)
    client.post(URL)
    _wait(client, URL)

    r = client.post(URL)
    assert r.status_code == 202
    assert r.json()["status"] == "done"
    assert calls == [API_ID]


def test_delete_removes_the_folder_and_an_unfinished_song_is_refused(client, monkeypatch):
    _api_job()
    _stub_generate(monkeypatch)
    client.post(URL)
    _wait(client, URL)

    assert client.delete(URL).json()["status"] == "none"
    assert not lead_sheet.output_dir(jobs_mod.JOBS_DIR / API_ID).exists()
    assert client.get(URL).json()["has_lead_sheet"] is False

    _jobs[API_ID].status = "running"
    assert client.post(URL).status_code == 409
