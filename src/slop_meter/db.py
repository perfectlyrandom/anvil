"""SQLAlchemy ORM models for slop_meter's local SQLite store.

Schema notes:
- One row per AI conversation in ``sessions``.
- One row per LLM turn in ``turns`` (request + response pair).
- ``prompt_buckets`` decomposes a turn's input tokens into named buckets (rule, skill, mcp, etc.) - populated in v0.2.
- ``github_events`` is independent - GitHub PRs/commits keyed by timestamp, joined to sessions via heuristics.
- ``tips`` and ``reflections`` are user-facing surfaces, not raw data.

All timestamps are stored as UTC ISO-8601 strings (SQLite has no native datetime).
Costs are stored as numeric strings (Decimal) to avoid float precision pain.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Engine, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all models."""


class SessionRecord(Base):
    """One AI conversation session (Cursor chat, Claude Code conversation, etc.)."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tool: Mapped[str] = mapped_column(String, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cwd: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    project: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    source_file: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    turns: Mapped[list[Turn]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Turn(Base):
    """One request/response pair within a session."""

    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str] = mapped_column(String)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    assistant_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    session: Mapped[SessionRecord] = relationship(back_populates="turns")
    buckets: Mapped[list[PromptBucket]] = relationship(back_populates="turn", cascade="all, delete-orphan")


class PromptBucket(Base):
    """v0.2 - per-turn attribution of input tokens to named buckets (rule / skill / mcp / context)."""

    __tablename__ = "prompt_buckets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[str] = mapped_column(String, ForeignKey("turns.id"), index=True)
    bucket_type: Mapped[str] = mapped_column(String, index=True)
    bucket_name: Mapped[str] = mapped_column(String, index=True)
    estimated_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    inferred: Mapped[bool] = mapped_column(default=True)

    turn: Mapped[Turn] = relationship(back_populates="buckets")


class GitHubEvent(Base):
    """A merged PR, commit, review, or issue pulled from GitHub GraphQL."""

    __tablename__ = "github_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    merged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    repo: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)
    changed_files: Mapped[int] = mapped_column(Integer, default=0)
    labels: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    pr_type: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)


class Tip(Base):
    """A surfaced optimization suggestion."""

    __tablename__ = "tips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tip_type: Mapped[str] = mapped_column(String, index=True)
    severity: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    estimated_savings_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ReflectionEntry(Base):
    """Optional weekly self-check-in - "how productive did this week feel? 1-5"."""

    __tablename__ = "reflections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_starting: Mapped[datetime] = mapped_column(DateTime, unique=True)
    felt_productivity: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime)


class ParserCheckpoint(Base):
    """Byte-offset checkpoints for incremental JSONL parsing.

    Tracked by (source_file_path, inode) so log rotation is detected via inode mismatch.
    """

    __tablename__ = "parser_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(String, index=True)
    inode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    byte_offset: Mapped[int] = mapped_column(Integer, default=0)
    last_parsed_at: Mapped[datetime] = mapped_column(DateTime)


def create_engine_for_path(db_path: Path, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine pointed at a SQLite file in WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=echo, future=True)
    return engine


def init_schema(engine: Engine) -> None:
    """Create all tables. Idempotent."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Yield a SQLAlchemy session that commits on success, rolls back on error."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
