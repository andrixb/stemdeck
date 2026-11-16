"""Timestamp parsing and formatting helpers."""

from __future__ import annotations


def parse_time(value: str | float | int | None) -> float | None:
    """Parse ``SS``, ``MM:SS`` or ``HH:MM:SS`` into seconds.

    Comma decimals are accepted (``1:02,5``) so timestamps copied from
    subtitles or spreadsheets work unchanged.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip().replace(",", ".")
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid timestamp: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
