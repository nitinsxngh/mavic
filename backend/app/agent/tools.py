from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import models as db_models
from app.memory.store import get_or_create_profile, list_profile_facts, save_memory, search_memories
from app.memory.entities import list_entities, search_entities, upsert_entity
from app.perception.location import reverse_geocode
from app.perception.vision import analyze_frames
from app.session.store import session_get

TOOL_PERMISSIONS = {
    "get_current_location": "sensitive_read",
    "search_personal_memory": "private_read",
    "get_user_profile": "private_read",
    "save_memory": "private_write",
    "analyze_current_scene": "private_read",
    "remember_entity": "private_write",
    "recall_entities": "private_read",
    "list_known_entities": "private_read",
    "get_current_time": "read",
}


REALTIME_TOOLS = [
    {
        "type": "function",
        "name": "search_personal_memory",
        "description": "Retrieve the authenticated user's personal memories from the vector store. Always scoped to the current user.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up about the user"}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_user_profile",
        "description": "Get the user's profile and durable preferences.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "save_memory",
        "description": "Store a durable personal fact the user asked you to remember.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "memory_type": {
                    "type": "string",
                    "enum": ["profile", "preference", "episodic"],
                },
            },
            "required": ["content"],
        },
    },
    {
        "type": "function",
        "name": "get_current_location",
        "description": "Get the user's current GPS location from the device. Never invent coordinates.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "analyze_current_scene",
        "description": "Look through the camera using multiple recently captured frames and return a verified visual observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "What to look for in the frames"}
            },
        },
    },
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current UTC time.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def _db() -> Session:
    if db_models.SessionLocal is None:
        db_models.init_db()
    if db_models.SessionLocal is None:
        raise RuntimeError("Database is not initialized")
    return db_models.SessionLocal()


def execute_tool(name: str, arguments: dict[str, Any], user_id: str, session_id: str) -> dict[str, Any]:
    args = arguments or {}
    if name == "search_personal_memory":
        db = _db()
        try:
            hits = search_memories(db, user_id=user_id, query=args.get("query", ""))
            return {"memories": hits}
        finally:
            db.close()

    if name == "get_user_profile":
        db = _db()
        try:
            settings = get_settings()
            profile = get_or_create_profile(db, user_id, settings.default_user_name)
            facts = list_profile_facts(db, user_id)
            return {
                "user_id": profile.user_id,
                "display_name": profile.display_name,
                "preferences": profile.preferences or {},
                "facts": facts,
            }
        finally:
            db.close()

    if name == "save_memory":
        db = _db()
        try:
            record = save_memory(
                db,
                user_id=user_id,
                content=args.get("content", ""),
                memory_type=args.get("memory_type", "episodic"),
            )
            return {"saved": True, "id": record.id}
        finally:
            db.close()

    if name == "get_current_location":
        snapshot = session_get(session_id, "location") or {}
        lat = snapshot.get("latitude")
        lng = snapshot.get("longitude")
        if lat is None or lng is None:
            return {"error": "No GPS fix yet. Ask the user to enable location."}
        geo = reverse_geocode(float(lat), float(lng), snapshot.get("accuracy_m"))
        return geo.model_dump()

    if name == "analyze_current_scene":
        frames = session_get(session_id, "frames") or []
        vision = analyze_frames(frames, question=str(args.get("question") or ""))
        payload = vision.model_dump()
        if args.get("question"):
            payload["asked"] = args["question"]
        return payload

    if name == "remember_entity":
        db = _db()
        try:
            if not (args.get("name") or "").strip():
                return {"saved": False, "error": "name required"}
            record = upsert_entity(
                db,
                user_id=user_id,
                name=args.get("name") or "",
                kind=args.get("kind") or "object",
                visual_description=args.get("visual_description") or "",
                notes=args.get("notes") or "",
            )
            return {"saved": True, "entity": {"id": record.id, "name": record.name, "kind": record.kind}}
        finally:
            db.close()

    if name == "recall_entities":
        db = _db()
        try:
            hits = search_entities(
                db,
                user_id=user_id,
                query=args.get("query") or "",
                kind=args.get("kind"),
            )
            return {"entities": hits}
        finally:
            db.close()

    if name == "list_known_entities":
        db = _db()
        try:
            return {"entities": list_entities(db, user_id=user_id)}
        finally:
            db.close()

    if name == "get_current_time":
        return {"utc": datetime.now(timezone.utc).isoformat()}

    return {"error": f"Unknown tool {name}"}
