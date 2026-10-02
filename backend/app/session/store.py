from __future__ import annotations

import json
from typing import Any, Optional

from app.config import get_settings

_memory = {}
_redis = None


def get_redis():
    global _redis
    if _redis is not None:
        return _redis
    settings = get_settings()
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        _redis = client
        return _redis
    except Exception:
        return None


def session_set(session_id: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
    settings = get_settings()
    payload = json.dumps(value)
    client = get_redis()
    namespaced = f"mavic:{session_id}:{key}"
    if client is not None:
        client.set(namespaced, payload, ex=ttl or settings.frame_ttl_seconds)
        return
    _memory[namespaced] = payload


def session_get(session_id: str, key: str, default: Any = None) -> Any:
    namespaced = f"mavic:{session_id}:{key}"
    client = get_redis()
    raw = client.get(namespaced) if client is not None else _memory.get(namespaced)
    if raw is None:
        return default
    return json.loads(raw)


def session_delete(session_id: str, key: str) -> None:
    namespaced = f"mavic:{session_id}:{key}"
    client = get_redis()
    if client is not None:
        client.delete(namespaced)
        return
    _memory.pop(namespaced, None)
