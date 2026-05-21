"""Tests for the Cursor bubbleId state.vscdb parser."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from anvil.parsers.cursor_bubbles import scan_cursor_bubbles


def _make_state_db(path: Path, entries: list[tuple[str, str]]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)")
        con.executemany("INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)", entries)
        con.commit()
    finally:
        con.close()


def test_returns_empty_when_db_missing(tmp_path: Path) -> None:
    result = scan_cursor_bubbles(tmp_path / "does-not-exist.vscdb")
    assert result.session_count == 0
    assert result.observations == {}


def test_aggregates_non_zero_bubbles_per_session(tmp_path: Path) -> None:
    db_path = tmp_path / "state.vscdb"
    entries = [
        (
            "bubbleId:session-a:bubble-1",
            json.dumps({"tokenCount": {"inputTokens": 100, "outputTokens": 20}}),
        ),
        (
            "bubbleId:session-a:bubble-2",
            json.dumps({"tokenCount": {"inputTokens": 50, "outputTokens": 5}}),
        ),
        (
            "bubbleId:session-b:bubble-1",
            json.dumps({"tokenCount": {"inputTokens": 200, "outputTokens": 40}}),
        ),
    ]
    _make_state_db(db_path, entries)

    result = scan_cursor_bubbles(db_path)
    assert result.session_count == 2
    a = result.tokens_for("session-a")
    assert a is not None
    assert a.input_tokens == 150
    assert a.output_tokens == 25
    assert a.bubble_count == 2
    b = result.tokens_for("session-b")
    assert b is not None
    assert b.input_tokens == 200
    assert b.bubble_count == 1


def test_skips_zero_token_bubbles_byok_marker(tmp_path: Path) -> None:
    """BYOK sessions record bubbles with tokenCount=0; we drop them so they don't show as 'measured'."""
    db_path = tmp_path / "state.vscdb"
    entries = [
        (
            "bubbleId:byok:bubble-1",
            json.dumps({"tokenCount": {"inputTokens": 0, "outputTokens": 0}}),
        ),
        (
            "bubbleId:byok:bubble-2",
            json.dumps({"tokenCount": {"inputTokens": 0, "outputTokens": 0}}),
        ),
        (
            "bubbleId:real:bubble-1",
            json.dumps({"tokenCount": {"inputTokens": 10, "outputTokens": 2}}),
        ),
    ]
    _make_state_db(db_path, entries)

    result = scan_cursor_bubbles(db_path)
    assert result.tokens_for("byok") is None
    assert result.tokens_for("real") is not None


@pytest.mark.parametrize(
    "value",
    [
        "not-json-at-all",
        json.dumps({"tokenCount": "string-not-dict"}),
        json.dumps({"no_token_count_field": True}),
        json.dumps({"tokenCount": {}}),
    ],
)
def test_tolerates_malformed_blobs(tmp_path: Path, value: str) -> None:
    db_path = tmp_path / "state.vscdb"
    _make_state_db(db_path, [("bubbleId:weird:b1", value)])
    result = scan_cursor_bubbles(db_path)
    assert result.session_count == 0


def test_ignores_unrelated_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "state.vscdb"
    _make_state_db(
        db_path,
        [
            ("agentKv:blob:abc", json.dumps({"tokenCount": {"inputTokens": 999}})),
            ("bcCachedDetails:bc-1", json.dumps({"tokenCount": {"inputTokens": 999}})),
            (
                "bubbleId:s:b",
                json.dumps({"tokenCount": {"inputTokens": 1, "outputTokens": 1}}),
            ),
        ],
    )
    result = scan_cursor_bubbles(db_path)
    assert result.session_count == 1
    assert result.tokens_for("s") is not None
