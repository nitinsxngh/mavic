from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import MemoryRecord, UserProfile, is_sqlite
from app.memory.embeddings import embed_text


def get_or_create_profile(db: Session, user_id: str, display_name: str = "") -> UserProfile:
    profile = db.get(UserProfile, user_id)
    if profile:
        return profile
    profile = UserProfile(user_id=user_id, display_name=display_name, preferences={})
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def save_memory(
    db: Session,
    user_id: str,
    content: str,
    memory_type: str = "episodic",
    importance: float = 0.5,
    extra: Optional[dict] = None,
) -> MemoryRecord:
    embedding = embed_text(content)
    record = MemoryRecord(
        id=f"mem_{uuid.uuid4().hex[:16]}",
        user_id=user_id,
        memory_type=memory_type,
        content=content,
        importance=importance,
        extra=extra or {},
        embedding=embedding,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return float(dot / ((na * nb) + 1e-9))


def search_memories(db: Session, user_id: str, query: str, k: int = 5) -> list[dict]:
    """Always filter by authenticated user_id. Never search the whole corpus."""
    query_embedding = embed_text(query)
    if is_sqlite():
        records = db.scalars(select(MemoryRecord).where(MemoryRecord.user_id == user_id)).all()
        ranked = []
        for record in records:
            score = _cosine(query_embedding, record.embedding or [])
            ranked.append(
                {
                    "id": record.id,
                    "content": record.content,
                    "memory_type": record.memory_type,
                    "importance": record.importance,
                    "score": score,
                    "metadata": record.extra or {},
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:k]

    vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    rows = db.execute(
        text(
            """
            SELECT id, content, memory_type, importance, extra_json,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM personal_memories
            WHERE user_id = :user_id
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :k
            """
        ),
        {"embedding": vector_literal, "user_id": user_id, "k": k},
    ).mappings()
    return [
        {
            "id": row["id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "importance": row["importance"],
            "score": float(row["score"] or 0),
            "metadata": row["extra_json"] or {},
        }
        for row in rows
    ]


def list_profile_facts(db: Session, user_id: str, limit: int = 12) -> list[dict]:
    records = db.scalars(
        select(MemoryRecord)
        .where(MemoryRecord.user_id == user_id)
        .where(MemoryRecord.memory_type.in_(["profile", "preference"]))
        .order_by(MemoryRecord.importance.desc())
        .limit(limit)
    ).all()
    return [
        {
            "id": r.id,
            "content": r.content,
            "memory_type": r.memory_type,
            "importance": r.importance,
        }
        for r in records
    ]
