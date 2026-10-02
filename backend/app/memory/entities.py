from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.db.models import EntityRecord, is_sqlite
from app.memory.embeddings import embed_text, openai_client
from app.config import get_settings

KINDS = ("person", "object", "animal")

INTRODUCE_HINT = re.compile(
    r"(this is|that's|that is|meet |his name|her name|its name|remember (this|him|her|it)|"
    r"call (him|her|it)|my dog|my cat|my friend|this is my|"
    r"ये हैं|ये है|ये मेरा|ये मेरी|ये मेरे|इसका नाम|उसका नाम|नाम है|याद रख|मिलो)",
    re.I,
)

RECOGNIZE_HINT = re.compile(
    r"(who is|who's|do you know|do you remember|what is this|what's this|"
    r"कौन है|किसका|पहचान|याद है|ये कौन|क्या है ये)",
    re.I,
)

REL_TO_KIND = {
    "dog": "animal",
    "cat": "animal",
    "pet": "animal",
    "puppy": "animal",
    "bird": "animal",
    "कुत्ता": "animal",
    "बिल्ला": "animal",
    "बिल्ली": "animal",
    "friend": "person",
    "brother": "person",
    "sister": "person",
    "mom": "person",
    "dad": "person",
    "wife": "person",
    "husband": "person",
    "दोस्त": "person",
    "भाई": "person",
    "बहन": "person",
    "मम्मी": "person",
    "पापा": "person",
    "wallet": "object",
    "phone": "object",
    "mug": "object",
    "keys": "object",
    "फोन": "object",
}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return float(dot / ((na * nb) + 1e-9))


def _embed_blob(name: str, kind: str, visual: str, notes: str) -> list[float]:
    try:
        return embed_text(f"{kind} named {name}. {visual}. {notes}".strip())
    except Exception:
        return []


def _serialize(record: EntityRecord, score: Optional[float] = None) -> dict:
    payload = {
        "id": record.id,
        "name": record.name,
        "kind": record.kind,
        "visual_description": record.visual_description,
        "notes": record.notes,
        "last_seen_at": record.last_seen_at.isoformat() if record.last_seen_at else None,
        "metadata": record.extra or {},
    }
    if score is not None:
        payload["score"] = score
    return payload


def is_introduction(transcript: str) -> bool:
    return bool(INTRODUCE_HINT.search(transcript or ""))


def is_recognition(transcript: str) -> bool:
    return bool(RECOGNIZE_HINT.search(transcript or ""))


def infer_kind(transcript: str, visual: dict, fallback: str = "object") -> str:
    blob = f"{transcript} {visual.get('scene', '')}".lower()
    for token, kind in REL_TO_KIND.items():
        if token in blob:
            return kind
    labels = " ".join(obj.get("label", "") for obj in visual.get("objects") or []).lower()
    if any(word in labels for word in ("person", "man", "woman", "face", "people")):
        return "person"
    if any(word in labels for word in ("dog", "cat", "animal", "bird", "pet")):
        return "animal"
    return fallback


def extract_introduction(transcript: str, visual: dict) -> Optional[dict]:
    """Regex first; small LLM JSON extract if the user is clearly introducing someone/something."""
    text = (transcript or "").strip()
    if not text:
        return None

    patterns = [
        re.compile(
            r"this is my (?P<rel>dog|cat|pet|friend|brother|sister|mom|dad|wife|husband)\s+(?P<name>[A-Za-z][\w]{1,40})",
            re.I,
        ),
        re.compile(
            r"this is my (?:black |white |red |blue |new |old )?(?P<name>wallet|phone|mug|keys|bag|watch|laptop|bottle|headphones)",
            re.I,
        ),
        re.compile(r"(?:meet |this is )(?P<name>[A-Za-z][\w]{1,40})", re.I),
        re.compile(r"(?:his|her|its) name is (?P<name>[A-Za-z][\w]{1,40})", re.I),
        re.compile(r"remember (?:this|him|her|it) as (?P<name>[\w\u0900-\u097F]{2,40})", re.I),
        re.compile(r"this is my (?P<name>[A-Za-z][\w]{1,40})", re.I),
        re.compile(r"इस(?:का|की|के) नाम (?P<name>[\w\u0900-\u097F]{2,40})", re.I),
        re.compile(r"ये (?:मेरा|मेरी|मेरे) (?P<rel>कुत्ता|बिल्ला|बिल्ली|दोस्त|भाई|बहन)?\s*(?P<name>[\w\u0900-\u097F]{2,40})", re.I),
        re.compile(r"ये हैं (?P<name>[\w\u0900-\u097F]{2,40})", re.I),
        re.compile(r"नाम है (?P<name>[\w\u0900-\u097F]{2,40})", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        name = (match.groupdict().get("name") or "").strip(" .,!")
        if not name or name.lower() in {"my", "the", "this", "that", "ये", "यह", "black", "white", "new", "old"}:
            continue
        rel = (match.groupdict().get("rel") or "").lower()
        kind = REL_TO_KIND.get(rel) or infer_kind(text, visual)
        return {
            "name": name,
            "kind": kind if kind in KINDS else "object",
            "notes": text,
            "visual_description": visual.get("scene") or "",
        }

    if not is_introduction(text):
        return None

    settings = get_settings()
    prompt = (
        "Extract the named person, pet/animal, or personal object the user is introducing. "
        'JSON only: {"name": string|null, "kind": "person"|"object"|"animal", "notes": string}. '
        "If they are not introducing a named entity, name must be null.\n"
        f"User: {text}\nVision: {visual}"
    )
    try:
        response = openai_client().chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return None
    name = (data.get("name") or "").strip()
    if not name:
        return None
    kind = data.get("kind") if data.get("kind") in KINDS else infer_kind(text, visual)
    return {
        "name": name,
        "kind": kind,
        "notes": data.get("notes") or text,
        "visual_description": visual.get("scene") or "",
    }


def upsert_entity(
    db: Session,
    user_id: str,
    name: str,
    kind: str = "object",
    visual_description: str = "",
    notes: str = "",
    extra: Optional[dict] = None,
) -> EntityRecord:
    kind = kind if kind in KINDS else "object"
    existing = db.scalars(
        select(EntityRecord)
        .where(EntityRecord.user_id == user_id)
        .where(func.lower(EntityRecord.name) == name.lower())
    ).first()
    now = datetime.now(timezone.utc)
    if existing:
        if visual_description and visual_description not in (existing.visual_description or ""):
            existing.visual_description = (existing.visual_description + " | " + visual_description).strip(" |")
        if notes and notes not in (existing.notes or ""):
            existing.notes = (existing.notes + " | " + notes).strip(" |")
        existing.kind = kind or existing.kind
        meta = dict(existing.extra or {})
        meta.update(extra or {})
        existing.extra = meta
        existing.last_seen_at = now
        existing.embedding = _embed_blob(existing.name, existing.kind, existing.visual_description, existing.notes)
        db.commit()
        db.refresh(existing)
        return existing

    record = EntityRecord(
        id=f"ent_{uuid.uuid4().hex[:16]}",
        user_id=user_id,
        kind=kind,
        name=name.strip(),
        visual_description=visual_description,
        notes=notes,
        extra=extra or {},
        embedding=_embed_blob(name, kind, visual_description, notes),
        created_at=now,
        last_seen_at=now,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def search_entities(db: Session, user_id: str, query: str, k: int = 5, kind: Optional[str] = None) -> list[dict]:
    try:
        query_embedding = embed_text(query)
    except Exception:
        query_embedding = []
    stmt = select(EntityRecord).where(EntityRecord.user_id == user_id)
    if kind in KINDS:
        stmt = stmt.where(EntityRecord.kind == kind)

    if is_sqlite() or not query_embedding:
        records = db.scalars(stmt).all()
        ranked = []
        for record in records:
            score = _cosine(query_embedding, record.embedding or [])
            if name_hit(query, record.name):
                score = max(score, 0.92)
            ranked.append(_serialize(record, score))
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return [item for item in ranked[:k] if item["score"] >= 0.28]

    vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    kind_filter = "AND kind = :kind" if kind in KINDS else ""
    rows = db.execute(
        text(
            f"""
            SELECT id, name, kind, visual_description, notes, extra_json, last_seen_at,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM known_entities
            WHERE user_id = :user_id {kind_filter}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :k
            """
        ),
        {"embedding": vector_literal, "user_id": user_id, "kind": kind, "k": k},
    ).mappings()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "visual_description": row["visual_description"],
            "notes": row["notes"],
            "last_seen_at": str(row["last_seen_at"]),
            "metadata": row["extra_json"] or {},
            "score": float(row["score"] or 0),
        }
        for row in rows
    ]


def name_hit(query: str, name: str) -> bool:
    q = (query or "").lower()
    n = (name or "").lower()
    return bool(n) and n in q


def list_entities(db: Session, user_id: str, limit: int = 20) -> list[dict]:
    records = db.scalars(
        select(EntityRecord)
        .where(EntityRecord.user_id == user_id)
        .order_by(EntityRecord.last_seen_at.desc())
        .limit(limit)
    ).all()
    return [_serialize(record) for record in records]
