from __future__ import annotations

from app.context.models import EmotionEstimate, SpeechFeatures


def estimate_emotion(features: SpeechFeatures, transcript: str = "") -> EmotionEstimate:
    """Probabilistic context only — never treat as ground truth."""
    energy = features.energy
    rate = features.speech_rate
    pitch = features.pitch_variance
    lowered = transcript.lower()

    label = "neutral"
    confidence = 0.45

    if energy < 0.28 and rate < 0.4:
        label, confidence = "possibly_tired", 0.62
    elif energy > 0.72 and pitch > 0.55:
        label, confidence = "possibly_excited", 0.66
    elif energy > 0.65 and rate > 0.7:
        label, confidence = "possibly_frustrated", 0.6
    elif energy > 0.5 and pitch < 0.25:
        label, confidence = "possibly_serious", 0.55

    if any(w in lowered for w in ("tired", "exhausted", "drained")):
        label, confidence = "possibly_tired", min(0.78, confidence + 0.15)
    if any(w in lowered for w in ("angry", "annoyed", "frustrated")):
        label, confidence = "possibly_frustrated", min(0.8, confidence + 0.15)

    phrasing = f"user MAY sound {label.replace('possibly_', '')} (confidence={confidence:.2f})"
    return EmotionEstimate(label=label, confidence=round(confidence, 2), phrasing=phrasing)
