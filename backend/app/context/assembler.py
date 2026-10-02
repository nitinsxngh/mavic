from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.context.models import (
    AgentContext,
    AudioContext,
    ConversationContext,
    EmotionEstimate,
    LocationContext,
    SpeechFeatures,
    VisionContext,
)
from app.perception.emotion import estimate_emotion


def assemble_context(
    *,
    session_id: str,
    user_id: str,
    transcript: str,
    speech_features: Optional[Dict[str, Any]] = None,
    location: Optional[Dict[str, Any]] = None,
    vision: Optional[Dict[str, Any]] = None,
    previous_turns: Optional[List[Dict[str, str]]] = None,
) -> AgentContext:
    features = SpeechFeatures(**(speech_features or {}))
    emotion = estimate_emotion(features, transcript)
    loc_data = {k: v for k, v in (location or {}).items() if k in LocationContext.model_fields}
    loc = LocationContext(**loc_data) if loc_data else LocationContext()
    vis = VisionContext(**(vision or {})) if vision else VisionContext()
    return AgentContext(
        session_id=session_id,
        user_id=user_id,
        conversation=ConversationContext(transcript=transcript, previous_turns=previous_turns or []),
        vision=vis,
        audio=AudioContext(transcript=transcript, speech_features=features, emotion=emotion),
        location=loc,
    )


def emotion_instruction(emotion: EmotionEstimate) -> str:
    return (
        f"Audio emotion is probabilistic context, not truth. {emotion.phrasing}. "
        "You may gently adapt tone. Never assert a psychological diagnosis."
    )
