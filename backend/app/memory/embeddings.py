from __future__ import annotations

from openai import OpenAI
from typing import Optional

from app.config import get_settings

_client: Optional[OpenAI] = None


def openai_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_text(text: str) -> list[float]:
    settings = get_settings()
    response = openai_client().embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding
