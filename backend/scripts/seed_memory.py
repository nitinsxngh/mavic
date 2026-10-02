from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import init_db
from app.memory.store import get_or_create_profile, save_memory, search_memories

SEED = [
    ("profile", "The user's name is Nitin Singh.", 0.95),
    ("preference", "User prefers concise technical explanations.", 0.9),
    ("preference", "User's favorite programming language is Python.", 0.85),
    ("episodic", "User is building Video Agent Buddy, also called Mavic Agent.", 0.88),
    ("episodic", "User is working on Bitconnect.", 0.7),
]


def seed(db: Session) -> None:
    settings = get_settings()
    user_id = settings.default_user_id
    get_or_create_profile(db, user_id, settings.default_user_name)
    existing = search_memories(db, user_id, "Video Agent Buddy", k=3)
    if existing and existing[0]["score"] > 0.8:
        print("Memory already seeded.")
        return
    for memory_type, content, importance in SEED:
        save_memory(db, user_id=user_id, content=content, memory_type=memory_type, importance=importance)
        print(f"saved {memory_type}: {content}")


if __name__ == "__main__":
    init_db()
    from app.db import models as db_models

    db = db_models.SessionLocal()
    try:
        seed(db)
    finally:
        db.close()
