from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.agent.language import LANGUAGE_POLICY
from app.agent.state import AgentState
from app.agent.tools import execute_tool
from app.config import get_settings
from app.context.assembler import emotion_instruction
from app.context.models import EmotionEstimate
from app.memory.entities import extract_introduction, is_introduction, is_recognition
from app.session.store import session_get, session_set

VISION_RE = re.compile(
    r"(hold|holding|look|looking|see|seeing|wearing|camera|this|that|in front|behind me|what am i|"
    r"phone|mobile|object|show|showing|"
    r"दिख|देख|दिखा|दिखाई|फोन|मोबाइल|कैमरा|ये क्या|यह क्या|क्या है|हाथ|पकड़|सामान)",
    re.I,
)
LOCATION_RE = re.compile(r"(where am i|location|which city|gps|near me|around here|कहाँ|कहा|लोकेशन)", re.I)
MEMORY_RE = re.compile(
    r"(my name|remember|who am i|what do you know|favorite|prefer|i like|last time|personal|नाम|याद|मेरा नाम)",
    re.I,
)
JUNK_TRANSCRIPT_RE = re.compile(
    r"(transcribe hindi|never use urdu|arabic script|keep english words|context:\s*#)",
    re.I,
)


def is_junk_transcript(transcript: str) -> bool:
    text = (transcript or "").strip()
    if len(text) < 2:
        return True
    if text.lower().startswith("context:"):
        return True
    return bool(JUNK_TRANSCRIPT_RE.search(text))


def detect_needs(transcript: str, has_frames: bool = False) -> list[str]:
    needs: list[str] = []
    if is_junk_transcript(transcript):
        return needs
    if VISION_RE.search(transcript) or has_frames or is_introduction(transcript) or is_recognition(transcript):
        needs.append("vision")
    if LOCATION_RE.search(transcript):
        needs.append("location")
    if MEMORY_RE.search(transcript):
        needs.append("memory")
    if is_introduction(transcript):
        needs.append("entity_save")
    if is_recognition(transcript) or has_frames:
        needs.append("entity_recall")
    return needs


def _llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key, temperature=0.4)


def normalize_context(state: AgentState) -> dict[str, Any]:
    session_id = state.get("session_id") or ""
    return {
        "transcript": (state.get("transcript") or "").strip(),
        "retrieved_memories": state.get("retrieved_memories") or [],
        "required_tools": state.get("required_tools") or [],
        "chat_history": state.get("chat_history") or session_get(session_id, "chat_history") or [],
        "ack": "",
        "response": "",
    }


def detect_intent(state: AgentState) -> dict[str, Any]:
    transcript = state.get("transcript") or ""
    frames = session_get(state.get("session_id") or "", "frames") or []
    needs = detect_needs(transcript, has_frames=bool(frames))
    intent = "casual"
    if needs == ["vision"]:
        intent = "vision"
    elif needs == ["location"]:
        intent = "location"
    elif needs == ["memory"]:
        intent = "memory"
    elif len(needs) > 1:
        intent = "mixed"
    ack = "Let me take a look." if "vision" in needs else ""
    return {"intent": intent, "required_tools": needs, "ack": ack}


async def gather_context(state: AgentState) -> dict[str, Any]:
    user_id = state["user_id"]
    session_id = state["session_id"]
    transcript = state.get("transcript") or ""
    needs = set(state.get("required_tools") or [])
    visual = dict(state.get("visual_context") or {})
    location = dict(state.get("location_context") or {})
    memories = list(state.get("retrieved_memories") or [])
    entities: list[dict] = []
    saved_entity: dict = {}

    async def run(name: str, arguments: dict) -> tuple[str, dict]:
        result = await asyncio.to_thread(execute_tool, name, arguments, user_id, session_id)
        return name, result

    jobs = []
    if "vision" in needs:
        jobs.append(run("analyze_current_scene", {"question": transcript}))
    if "location" in needs and not location.get("city"):
        jobs.append(run("get_current_location", {}))
    if "memory" in needs:
        jobs.append(run("search_personal_memory", {"query": transcript}))
        jobs.append(run("get_user_profile", {}))

    if jobs:
        results = await asyncio.gather(*[job for job in jobs])
        for name, result in results:
            if name == "analyze_current_scene":
                visual = result
            elif name == "get_current_location":
                location = result
            elif name == "search_personal_memory":
                memories = result.get("memories") or []
            elif name == "get_user_profile":
                memories = memories + [{"content": f"profile:{result}", "memory_type": "profile"}]

    if "entity_save" in needs:
        intro = extract_introduction(transcript, visual)
        if intro and intro.get("name"):
            _, saved_entity = await run(
                "remember_entity",
                {
                    "name": intro["name"],
                    "kind": intro.get("kind") or "object",
                    "visual_description": intro.get("visual_description") or visual.get("scene") or "",
                    "notes": intro.get("notes") or transcript,
                },
            )

    if "entity_recall" in needs or "entity_save" in needs or "vision" in needs:
        labels = ", ".join(obj.get("label", "") for obj in (visual.get("objects") or []) if isinstance(obj, dict))
        query = " ".join(part for part in [transcript, visual.get("scene") or "", labels] if part)
        if query.strip():
            _, recalled = await run("recall_entities", {"query": query})
            entities = recalled.get("entities") or []

    return {
        "visual_context": visual,
        "location_context": location,
        "retrieved_memories": memories,
        "recognized_entities": entities,
        "saved_entity": saved_entity,
    }


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts)
    return str(content)


def _reason_messages(state: AgentState) -> list:
    audio = state.get("audio_context") or {}
    emotion = EmotionEstimate(
        label=(audio.get("emotion") or {}).get("label", "unknown"),
        confidence=float((audio.get("emotion") or {}).get("confidence") or 0),
        phrasing=(audio.get("emotion") or {}).get("phrasing") or "",
    )
    system = f"""You are Mavic, a Video Agent Buddy. Camera is your eyes, microphone is your ears, speaker is your voice.
{LANGUAGE_POLICY}
You already received structured perception. Do not invent GPS, objects, or personal facts.
Use recent_conversation: if they refer to something they said earlier this session, answer from that. Do not claim they never told you if it is in recent_conversation.
If visual_context has a scene or objects, you ARE seeing through the camera right now — answer from that when they ask what you see or how they look. Never dump a full scene description for small talk (hello, how are you).
Never say the camera is off if a scene description exists.
If vision notes say no frames were available, ask the user to hold the object toward the camera and ask again.
If the user introduces a person, pet/animal, or personal object, confirm you will remember them by name.
If recognized_entities has a strong match (score above ~0.45), identify them by name. Do not invent names that are not in memory or the user's words.
If saved_entity is present, confirm: you now know that named person/object/animal.
{emotion_instruction(emotion)}
Keep spoken replies concise (1-4 sentences) unless the user asks for depth.
"""
    payload = {
        "transcript": state.get("transcript"),
        "recent_conversation": (state.get("chat_history") or [])[-12:],
        "vision": state.get("visual_context"),
        "location": state.get("location_context"),
        "memories": state.get("retrieved_memories"),
        "known_entities": state.get("recognized_entities"),
        "just_saved_entity": state.get("saved_entity"),
        "audio": audio,
    }
    return [
        SystemMessage(content=system),
        HumanMessage(content=f"AgentContext:\n{payload}"),
    ]


def agent_reason(state: AgentState) -> dict[str, Any]:
    llm = _llm()
    result = llm.invoke(_reason_messages(state))
    return {"response": result.content}


async def stream_reason(state: AgentState):
    llm = _llm()
    async for chunk in llm.astream(_reason_messages(state)):
        piece = _chunk_text(chunk)
        if piece:
            yield piece


def remember_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    if not session_id or is_junk_transcript(user_text):
        return
    history = list(session_get(session_id, "chat_history") or [])
    history.append({"role": "user", "text": user_text})
    if assistant_text:
        history.append({"role": "assistant", "text": assistant_text})
    session_set(session_id, "chat_history", history[-24:], ttl=60 * 60)


def response_prepare(state: AgentState) -> dict[str, Any]:
    return {"response": (state.get("response") or "").strip()}


def build_graph(include_reason: bool = True):
    graph = StateGraph(AgentState)
    graph.add_node("normalize_context", normalize_context)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("gather_context", gather_context)
    graph.add_edge(START, "normalize_context")
    graph.add_edge("normalize_context", "detect_intent")
    graph.add_edge("detect_intent", "gather_context")
    if include_reason:
        graph.add_node("agent_reason", agent_reason)
        graph.add_node("response_prepare", response_prepare)
        graph.add_edge("gather_context", "agent_reason")
        graph.add_edge("agent_reason", "response_prepare")
        graph.add_edge("response_prepare", END)
    else:
        graph.add_edge("gather_context", END)
    return graph.compile()


agent_graph = build_graph(include_reason=True)
gather_graph = build_graph(include_reason=False)
