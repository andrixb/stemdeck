"""Regression tests for lead-sheet audio alignment."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from app.leadsheet.align.dtw import align
from app.leadsheet.transcribe.events import NoteEvent


def test_silent_chroma_frames_do_not_abort_dtw(monkeypatch):
    chroma = np.zeros((12, 3), dtype=float)
    chroma[0, 0] = chroma[0, 2] = 1.0
    feature_calls = 0

    def chroma_cens(*, y, sr, hop_length):
        nonlocal feature_calls
        feature_calls += 1
        return chroma.copy()

    def dtw(*, X, Y, metric):
        assert np.isfinite(X).all() and np.isfinite(Y).all()
        assert (np.linalg.norm(X, axis=0) > 0).all()
        assert (np.linalg.norm(Y, axis=0) > 0).all()
        return np.zeros((3, 3)), np.array([[2, 2], [1, 1], [0, 0]])

    fake_librosa = SimpleNamespace(
        feature=SimpleNamespace(chroma_cens=chroma_cens),
        sequence=SimpleNamespace(dtw=dtw),
        frames_to_time=lambda frames, *, sr, hop_length: frames * hop_length / sr,
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    result = align(
        np.ones(22_050, dtype=np.float32),
        [NoteEvent(start=0.0, end=0.5, pitch=60)],
        [0.0, 0.1, 0.2],
        120.0,
        sr=22_050,
        hop_length=2_205,
    )

    assert feature_calls == 2
    assert result["method"] == "dtw"
