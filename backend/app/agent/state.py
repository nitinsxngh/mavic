from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    transcript: str
    visual_context: dict[str, Any]
    audio_context: dict[str, Any]
    location_context: dict[str, Any]
    retrieved_memories: list[dict[str, Any]]
    recognized_entities: list[dict[str, Any]]
    saved_entity: dict[str, Any]
    chat_history: list[dict[str, Any]]
    intent: str
    required_tools: list[str]
    response: str
    ack: str
    messages: Annotated[list, add_messages]
