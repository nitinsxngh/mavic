from __future__ import annotations

import json
from typing import Any

from app.config import get_settings
from app.context.models import VisionContext, VisualObject
from app.memory.embeddings import openai_client
from app.perception.frames import select_keyframes

VISION_PROMPT = """You are the visual perception layer for Video Agent Buddy.
Analyze ALL provided frames together. Do not trust a single frame.
Look for phones, screens, objects in hand, clothing, and the room.
Return JSON only:
{
  "scene": "short scene description of what is actually visible",
  "objects": [{"label": "...", "color": "...", "confidence": 0.0}],
  "confidence": 0.0,
  "verified_from_frames": 0,
  "notes": "what agreed across frames vs what was uncertain"
}
If a phone or device is visible, name the brand/model only if you can see it clearly; otherwise describe color, size, and what is on screen.
"""


def analyze_frames(frames: list[dict[str, Any]], question: str = "") -> VisionContext:
    selected = select_keyframes(frames)
    if not selected:
        return VisionContext(scene="no frames available", confidence=0.0, notes="camera context missing")

    settings = get_settings()
    asked = f"\nUser question: {question}\nAnswer that question from the frames." if question else ""
    content: list[dict[str, Any]] = [{"type": "text", "text": VISION_PROMPT + asked}]
    for frame in selected:
        data = frame.get("data") or ""
        if data.startswith("data:"):
            data = data.split(",", 1)[-1]
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{data}"},
            }
        )

    response = openai_client().chat.completions.create(
        model=settings.openai_vision_model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = json.loads(response.choices[0].message.content or "{}")
    objects = [
        VisualObject(
            label=item.get("label", "unknown"),
            color=item.get("color"),
            confidence=float(item.get("confidence") or 0),
        )
        for item in raw.get("objects", [])
    ]
    return VisionContext(
        scene=raw.get("scene", ""),
        objects=objects,
        confidence=float(raw.get("confidence") or 0),
        verified_from_frames=int(raw.get("verified_from_frames") or len(selected)),
        notes=raw.get("notes", ""),
        raw=raw,
    )
