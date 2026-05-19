"""Smoke tests for the SQLAlchemy schema."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import inspect

from slop_meter.db import (
    Base,
    GitHubEvent,
    ParserCheckpoint,
    PromptBucket,
    ReflectionEntry,
    SessionRecord,
    Tip,
    Turn,
    create_engine_for_path,
    init_schema,
    session_scope,
)


def test_init_schema_creates_all_tables(tmp_path) -> None:
    engine = create_engine_for_path(tmp_path / "data.db")
    init_schema(engine)
    inspector = inspect(engine)
    actual_tables = set(inspector.get_table_names())
    expected_tables = {table.__tablename__ for table in Base.__subclasses__()}
    assert expected_tables.issubset(actual_tables)


def test_insert_and_query_session(tmp_path) -> None:
    engine = create_engine_for_path(tmp_path / "data.db")
    init_schema(engine)
    started = datetime.now(UTC).replace(tzinfo=None)
    with session_scope(engine) as session:
        session.add(
            SessionRecord(
                id="sess-1",
                tool="claude_code",
                started_at=started,
                turn_count=0,
            )
        )
    with session_scope(engine) as session:
        row = session.get(SessionRecord, "sess-1")
        assert row is not None
        assert row.tool == "claude_code"


def test_models_register_with_base() -> None:
    """All models are registered with the declarative base."""
    table_names = {table.__tablename__ for table in Base.__subclasses__()}
    assert {
        SessionRecord.__tablename__,
        Turn.__tablename__,
        PromptBucket.__tablename__,
        GitHubEvent.__tablename__,
        Tip.__tablename__,
        ReflectionEntry.__tablename__,
        ParserCheckpoint.__tablename__,
    } <= table_names
