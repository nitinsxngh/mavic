from __future__ import annotations

import hashlib
import hmac
import time

from typing import Optional

from fastapi import Header, HTTPException

from app.config import get_settings


def current_user_id(x_user_id: Optional[str] = Header(default=None)) -> str:
    """Identity is always injected by the backend, never by the LLM."""
    settings = get_settings()
    if settings.app_env == "development":
        return x_user_id or settings.default_user_id
    raise HTTPException(status_code=401, detail="Identity required")


def hashed_safety_id(user_id: str) -> str:
    settings = get_settings()
    secret = settings.jwt_secret or "mavic-dev-safety"
    digest = hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()
    return digest[:32]


def new_session_id() -> str:
    return f"sess_{int(time.time() * 1000)}"
