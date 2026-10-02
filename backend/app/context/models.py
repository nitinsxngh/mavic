from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VisualObject(BaseModel):
    label: str
    color: Optional[str] = None
    confidence: float = 0.0


class VisionContext(BaseModel):
    scene: str = ""
    objects: List[VisualObject] = Field(default_factory=list)
    confidence: float = 0.0
    verified_from_frames: int = 0
    notes: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)


class EmotionEstimate(BaseModel):
    label: str = "unknown"
    confidence: float = 0.0
    phrasing: str = "emotion unknown"


class SpeechFeatures(BaseModel):
    energy: float = 0.0
    speech_rate: float = 0.0
    pitch_variance: float = 0.0


class AudioContext(BaseModel):
    transcript: str = ""
    speech_features: SpeechFeatures = Field(default_factory=SpeechFeatures)
    emotion: EmotionEstimate = Field(default_factory=EmotionEstimate)


class LocationContext(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    formatted: Optional[str] = None
    timestamp: Optional[str] = None
    source: str = "unknown"


class ConversationContext(BaseModel):
    transcript: str = ""
    previous_turns: List[Dict[str, str]] = Field(default_factory=list)


class AgentContext(BaseModel):
    session_id: str
    user_id: str
    conversation: ConversationContext = Field(default_factory=ConversationContext)
    vision: VisionContext = Field(default_factory=VisionContext)
    audio: AudioContext = Field(default_factory=AudioContext)
    location: LocationContext = Field(default_factory=LocationContext)
    memory: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
