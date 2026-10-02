from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import JSON, DateTime, Float, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import get_settings

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    Vector = None  # type: ignore


def is_sqlite() -> bool:
    return get_settings().database_url.startswith("sqlite")


def _embedding_column():
    if is_sqlite() or Vector is None:
        return JSON
    return Vector(1536)


class Base(DeclarativeBase):
    pass


class MemoryRecord(Base):
    __tablename__ = "personal_memories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    memory_type: Mapped[str] = mapped_column(String(32), default="episodic")
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    extra: Mapped[dict] = mapped_column("extra_json", JSON, default=dict)
    embedding: Mapped[Optional[List[float]]] = mapped_column(_embedding_column(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EntityRecord(Base):
    __tablename__ = "known_entities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="object")
    name: Mapped[str] = mapped_column(String(128), index=True)
    visual_description: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    extra: Mapped[dict] = mapped_column("extra_json", JSON, default=dict)
    embedding: Mapped[Optional[List[float]]] = mapped_column(_embedding_column(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


engine = None
SessionLocal = None


def init_db() -> None:
    global engine, SessionLocal
    settings = get_settings()
    kwargs = {"pool_pre_ping": True}
    if is_sqlite():
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(settings.database_url, **kwargs)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    if not is_sqlite():
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    Base.metadata.create_all(engine)


def get_session():
    if SessionLocal is None:
        raise RuntimeError("Database is not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
