from __future__ import annotations

import hashlib
from typing import Any

from app.config import get_settings


def _phash(b64_jpeg: str) -> str:
    return hashlib.md5(b64_jpeg[:4000].encode()).hexdigest()


def select_keyframes(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop near-duplicates and keep a small verified set."""
    settings = get_settings()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame in frames:
        data = frame.get("data") or frame.get("image") or ""
        if not data:
            continue
        digest = _phash(data)
        if digest in seen:
            continue
        seen.add(digest)
        selected.append(frame)
        if len(selected) >= settings.max_frames_per_turn:
            break
    return selected
