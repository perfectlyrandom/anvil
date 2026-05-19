"""Smoke tests for the Cursor agent-transcripts parser."""

from __future__ import annotations

import json
from pathlib import Path

from anvil.analysis.cursor_stats import aggregate
from anvil.parsers.cursor_transcripts import (
    estimate_tokens,
    parse_session_file,
    scan_cursor_projects,
)


def _make_user_turn(text: str) -> str:
    return json.dumps({"role": "user", "message": {"content": [{"type": "text", "text": text}]}})


def _make_assistant_turn(text: str) -> str:
    return json.dumps({"role": "assistant", "message": {"content": [{"type": "text", "text": text}]}})


def _write_session(root: Path, workspace: str, session_id: str, lines: list[str]) -> Path:
    path = root / "projects" / workspace / "agent-transcripts" / session_id / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_estimate_tokens_nonzero_for_real_text() -> None:
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0


def test_parse_single_session_attributes_buckets(tmp_path: Path) -> None:
    user_turn = _make_user_turn(
        "<user_query>\nfix this bug in the parser\n</user_query>\n"
        "<attached_files>\nlong attached file content " + "x" * 500 + "\n</attached_files>"
    )
    assistant_turn = _make_assistant_turn("ok, here's the fix")
    path = _write_session(tmp_path, "ws-1", "sess-1", [user_turn, assistant_turn])

    session = parse_session_file(path)

    assert session is not None
    assert session.session_id == "sess-1"
    assert session.workspace == "ws-1"
    assert session.user_turn_count == 1
    assert session.assistant_turn_count == 1
    assert session.first_user_query == "fix this bug in the parser"
    assert "user_query" in session.bucket_tokens
    assert "attached_files" in session.bucket_tokens
    assert session.bucket_tokens["attached_files"] > session.bucket_tokens["user_query"]


def test_scan_cursor_projects_finds_nested_and_subagents(tmp_path: Path) -> None:
    _write_session(tmp_path, "ws-1", "sess-a", [_make_user_turn("<user_query>hi</user_query>")])
    sub_dir = tmp_path / "projects" / "ws-1" / "agent-transcripts" / "sess-a" / "subagents"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "subagent-x.jsonl").write_text(
        _make_user_turn("<user_query>do the subtask</user_query>") + "\n",
        encoding="utf-8",
    )

    scan = scan_cursor_projects(tmp_path / "projects")

    assert scan.total_files_seen == 2
    assert len(scan.parent_sessions) == 1
    assert len(scan.subagent_sessions) == 1


def test_aggregate_rolls_up_buckets_and_workspaces(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "ws-1",
        "sess-a",
        [_make_user_turn("<user_query>do x</user_query>"), _make_assistant_turn("done")],
    )
    _write_session(
        tmp_path,
        "ws-2",
        "sess-b",
        [
            _make_user_turn("<user_query>another</user_query><attached_files>" + "a" * 1000 + "</attached_files>"),
            _make_assistant_turn("ok"),
        ],
    )

    scan = scan_cursor_projects(tmp_path / "projects")
    agg = aggregate(scan)

    assert agg.total_sessions == 2
    assert len(agg.workspaces) == 2
    bucket_names = {b.name for b in agg.buckets}
    assert "user_query" in bucket_names
    assert "attached_files" in bucket_names
