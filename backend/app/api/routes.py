from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from app.agent.graph import agent_graph, gather_graph, is_junk_transcript, remember_turn, stream_reason
from app.agent.language import LANGUAGE_POLICY
from app.agent.tools import execute_tool
from app.config import get_settings
from app.context.assembler import assemble_context
from app.security.auth import current_user_id, hashed_safety_id, new_session_id
from app.session.store import session_get, session_set

router = APIRouter()


class FramePayload(BaseModel):
    timestamp_ms: Optional[int] = None
    offset_ms: Optional[int] = None
    data: str


class LocationPayload(BaseModel):
    latitude: float
    longitude: float
    accuracy_m: Optional[float] = None


class SpeechFeaturesPayload(BaseModel):
    energy: float = 0
    speech_rate: float = 0
    pitch_variance: float = 0


class ContextUpdate(BaseModel):
    session_id: str
    frames: List[FramePayload] = Field(default_factory=list)
    location: Optional[LocationPayload] = None
    speech_features: Optional[SpeechFeaturesPayload] = None
    transcript: Optional[str] = None


class ToolCall(BaseModel):
    session_id: str
    name: str
    arguments: Dict = Field(default_factory=dict)
    call_id: Optional[str] = None


class TurnRequest(BaseModel):
    session_id: str
    transcript: str
    speech_features: Optional[SpeechFeaturesPayload] = None
    location: Optional[LocationPayload] = None


class SessionOut(BaseModel):
    session_id: str
    user_id: str


@router.post("/session", response_model=SessionOut)
def create_session(user_id: str = Depends(current_user_id)):
    session_id = new_session_id()
    session_set(session_id, "user_id", user_id, ttl=60 * 60)
    return SessionOut(session_id=session_id, user_id=user_id)


@router.post("/context")
def update_context(body: ContextUpdate, user_id: str = Depends(current_user_id)):
    if body.frames:
        encoded = [frame.model_dump() for frame in body.frames]
        session_set(body.session_id, "frames", encoded)
    if body.location:
        session_set(body.session_id, "location", body.location.model_dump())
    if body.speech_features:
        session_set(body.session_id, "speech_features", body.speech_features.model_dump())
    if body.transcript is not None:
        session_set(body.session_id, "transcript", body.transcript)
    return {"ok": True, "frames": len(body.frames)}


@router.post("/tools/execute")
def run_tool(body: ToolCall, user_id: str = Depends(current_user_id)):
    result = execute_tool(body.name, body.arguments, user_id=user_id, session_id=body.session_id)
    return {"call_id": body.call_id, "name": body.name, "result": result}


@router.post("/turn")
async def run_turn(body: TurnRequest, user_id: str = Depends(current_user_id)):
    frames = session_get(body.session_id, "frames") or []
    location = body.location.model_dump() if body.location else session_get(body.session_id, "location")
    features = body.speech_features.model_dump() if body.speech_features else session_get(body.session_id, "speech_features")
    ctx = assemble_context(
        session_id=body.session_id,
        user_id=user_id,
        transcript=body.transcript,
        speech_features=features,
        location=location,
    )
    result = await agent_graph.ainvoke(
        {
            "session_id": body.session_id,
            "user_id": user_id,
            "transcript": body.transcript,
            "visual_context": ctx.vision.model_dump(),
            "audio_context": ctx.audio.model_dump(),
            "location_context": ctx.location.model_dump(),
        }
    )
    remember_turn(body.session_id, body.transcript, result.get("response") or "")
    return {
        "ack": result.get("ack"),
        "intent": result.get("intent"),
        "required_tools": result.get("required_tools"),
        "response": result.get("response"),
        "vision": result.get("visual_context"),
        "location": result.get("location_context"),
        "memories": result.get("retrieved_memories"),
        "entities": result.get("recognized_entities"),
        "saved_entity": result.get("saved_entity"),
        "emotion": ctx.audio.emotion.model_dump(),
        "frames_buffered": len(frames),
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/turn/stream")
async def run_turn_stream(body: TurnRequest, user_id: str = Depends(current_user_id)):
    frames = session_get(body.session_id, "frames") or []
    location = body.location.model_dump() if body.location else session_get(body.session_id, "location")
    features = body.speech_features.model_dump() if body.speech_features else session_get(body.session_id, "speech_features")
    ctx = assemble_context(
        session_id=body.session_id,
        user_id=user_id,
        transcript=body.transcript,
        speech_features=features,
        location=location,
    )
    emotion = ctx.audio.emotion.model_dump()

    async def events():
        if is_junk_transcript(body.transcript):
            yield _sse("done", {"response": "", "emotion": emotion, "junk": True})
            return
        gathered = await gather_graph.ainvoke(
            {
                "session_id": body.session_id,
                "user_id": user_id,
                "transcript": body.transcript,
                "visual_context": ctx.vision.model_dump(),
                "audio_context": ctx.audio.model_dump(),
                "location_context": ctx.location.model_dump(),
            }
        )
        yield _sse(
            "meta",
            {
                "ack": gathered.get("ack"),
                "intent": gathered.get("intent"),
                "required_tools": gathered.get("required_tools"),
                "vision": gathered.get("visual_context"),
                "location": gathered.get("location_context"),
                "memories": gathered.get("retrieved_memories"),
                "entities": gathered.get("recognized_entities"),
                "saved_entity": gathered.get("saved_entity"),
                "emotion": emotion,
                "frames_buffered": len(frames),
            },
        )
        text = ""
        async for piece in stream_reason(gathered):
            text += piece
            yield _sse("token", {"text": piece})
        reply = text.strip()
        remember_turn(body.session_id, body.transcript, reply)
        yield _sse("done", {"response": reply, "emotion": emotion})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/realtime/sdp")
async def realtime_sdp(request: Request, user_id: str = Depends(current_user_id)):
    """Unified WebRTC handshake: API key stays on the server."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing")

    sdp = (await request.body()).decode()
    instructions = f"""You are Mavic's voice. Only speak the reply you are given.
{LANGUAGE_POLICY}
Do not call tools. Do not invent extra facts.
"""
    session = {
        "type": "realtime",
        "model": settings.openai_realtime_model,
        "instructions": instructions,
        "audio": {
            "input": {
                "transcription": {
                    "model": "gpt-4o-mini-transcribe",
                    "language": "hi",
                    "prompt": "Hinglish. Hindi in Devanagari. English in Latin. No Urdu or Arabic script.",
                },
                "turn_detection": {
                    "type": "semantic_vad",
                    "create_response": False,
                    "interrupt_response": False,
                },
            },
            "output": {
                "voice": "marin",
            },
        },
    }

    import httpx

    files = {
        "sdp": (None, sdp, "application/sdp"),
        "session": (None, json.dumps(session), "application/json"),
    }
    timeout = httpx.Timeout(connect=25.0, read=60.0, write=60.0, pool=10.0)
    response: Optional[httpx.Response] = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    "https://api.openai.com/v1/realtime/calls",
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "OpenAI-Safety-Identifier": hashed_safety_id(user_id),
                    },
                    files=files,
                )
                break
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if attempt == 2:
                    raise HTTPException(
                        status_code=504,
                        detail="Voice handshake timed out reaching OpenAI. Check internet / VPN and tap Enable again.",
                    ) from exc
    if response is None:
        raise HTTPException(status_code=504, detail="Voice handshake failed.")
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return PlainTextResponse(content=response.text, media_type="application/sdp")
