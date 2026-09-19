"""Shared timestamp helpers: ``MM:SS.ss`` <-> seconds.

Used by every script that reads or writes an events CSV (annotate.py,
inspect_video.py, detect_touches.py, evaluate.py) so the format stays
consistent everywhere.
"""


def format_timestamp(seconds: float) -> str:
    """Seconds -> ``MM:SS.ss`` string, e.g. 41.77 -> ``00:41.77``."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def parse_timestamp(text: str) -> float:
    """``MM:SS.ss`` string -> seconds, e.g. ``00:41.77`` -> 41.77.

    Inverse of format_timestamp. Raises ValueError on malformed input.
    """
    minutes_str, _, secs_str = text.strip().partition(":")
    return int(minutes_str) * 60 + float(secs_str)
