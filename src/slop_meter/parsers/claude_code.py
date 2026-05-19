"""Parser for Claude Code's transcript JSONL files.

Claude Code stores chat history under::

    ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl

Unlike Cursor, Claude Code transcripts include:

- Real ``timestamp`` (ISO-8601)
- Real ``model`` (e.g. ``claude-sonnet-4-5-20250929`` or ``<synthetic>``)
- Real ``usage`` block with ``input_tokens``, ``output_tokens``,
  ``cache_creation_input_tokens``, ``cache_read_input_tokens``, and a nested
  ``cache_creation`` breakdown (ephemeral_1h vs ephemeral_5m).
- ``cwd`` (decoded working directory, not just the encoded form).
- ``gitBranch``, ``version``, ``entrypoint``.
- ``isSidechain`` (Claude Code's equivalent of Cursor's subagent transcripts).
- ``parentUuid`` for reconstructing the turn graph.

These fields make Claude Code our richest data source - the cost dashboard
and cache-health analyses key off it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TurnUsage:
    """Per-turn token usage breakdown."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    web_search_requests: int = 0
    web_fetch_requests: int = 0


@dataclass
class ClaudeCodeTurn:
    """One turn in a Claude Code session."""

    turn_index: int
    role: str
    model: str | None
    timestamp: datetime | None
    usage: TurnUsage
    text_preview: str
    is_sidechain: bool
    is_meta: bool


@dataclass
class ClaudeCodeSession:
    """One Claude Code session, parsed from a single JSONL file."""

    session_id: str
    source_file: Path
    cwd: str
    git_branch: str | None
    version: str | None
    entrypoint: str | None
    started_at: datetime | None
    ended_at: datetime | None
    turn_count: int
    usage_by_model: dict[str, TurnUsage]
    total_usage: TurnUsage
    first_user_text: str | None
    turns: list[ClaudeCodeTurn]
    is_sidechain_session: bool


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Claude Code emits "2026-04-01T22:05:35.508Z" format.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _extract_text(content: object) -> str:
    """Pull a human-readable text preview out of various content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    pieces.append(str(item.get("text", "")))
                elif item.get("type") == "tool_use":
                    pieces.append(f"[tool_use: {item.get('name', '?')}]")
                elif item.get("type") == "tool_result":
                    pieces.append("[tool_result]")
        return "\n".join(pieces)
    return ""


def _parse_usage(usage: dict[str, Any] | None) -> TurnUsage:
    if not usage:
        return TurnUsage()
    server_tool_raw = usage.get("server_tool_use")
    server_tool: dict[str, Any] = server_tool_raw if isinstance(server_tool_raw, dict) else {}
    return TurnUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        web_search_requests=int(server_tool.get("web_search_requests") or 0),
        web_fetch_requests=int(server_tool.get("web_fetch_requests") or 0),
    )


def _sum_usage(a: TurnUsage, b: TurnUsage) -> TurnUsage:
    return TurnUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_creation_input_tokens=a.cache_creation_input_tokens + b.cache_creation_input_tokens,
        cache_read_input_tokens=a.cache_read_input_tokens + b.cache_read_input_tokens,
        web_search_requests=a.web_search_requests + b.web_search_requests,
        web_fetch_requests=a.web_fetch_requests + b.web_fetch_requests,
    )


def parse_session_file(path: Path) -> ClaudeCodeSession | None:
    """Parse a single Claude Code JSONL transcript. Returns None if empty / unreadable."""
    if not path.exists() or path.stat().st_size == 0:
        return None

    turns: list[ClaudeCodeTurn] = []
    usage_by_model: dict[str, TurnUsage] = defaultdict(TurnUsage)
    total_usage = TurnUsage()
    first_user_text: str | None = None
    cwd = ""
    git_branch: str | None = None
    version: str | None = None
    entrypoint: str | None = None
    session_id = path.stem
    started_at: datetime | None = None
    ended_at: datetime | None = None
    any_sidechain = False
    all_sidechain = True

    with path.open("r", encoding="utf-8") as fh:
        for turn_index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = obj.get("type", "")
            # file-history-snapshot, last-prompt etc. are metadata, not chat turns.
            if entry_type not in {"user", "assistant"}:
                continue

            timestamp = _parse_timestamp(obj.get("timestamp"))
            if timestamp is not None:
                if started_at is None or timestamp < started_at:
                    started_at = timestamp
                if ended_at is None or timestamp > ended_at:
                    ended_at = timestamp

            cwd = cwd or str(obj.get("cwd", ""))
            git_branch = git_branch or obj.get("gitBranch")
            version = version or obj.get("version")
            entrypoint = entrypoint or obj.get("entrypoint")
            session_id = session_id or str(obj.get("sessionId", path.stem))

            is_sidechain = bool(obj.get("isSidechain", False))
            is_meta = bool(obj.get("isMeta", False))
            any_sidechain = any_sidechain or is_sidechain
            if not is_sidechain:
                all_sidechain = False

            msg = obj.get("message", {}) or {}
            assert isinstance(msg, dict)
            role = str(msg.get("role", entry_type))
            content = msg.get("content")
            text = _extract_text(content)
            model = msg.get("model")
            usage = _parse_usage(msg.get("usage") if isinstance(msg.get("usage"), dict) else None)

            if role == "user" and first_user_text is None and not is_meta and text:
                first_user_text = text

            if model and role == "assistant":
                usage_by_model[model] = _sum_usage(usage_by_model[model], usage)
            total_usage = _sum_usage(total_usage, usage)

            preview = text[:200] + ("..." if len(text) > 200 else "")
            turns.append(
                ClaudeCodeTurn(
                    turn_index=turn_index,
                    role=role,
                    model=model if isinstance(model, str) else None,
                    timestamp=timestamp,
                    usage=usage,
                    text_preview=preview,
                    is_sidechain=is_sidechain,
                    is_meta=is_meta,
                )
            )

    if not turns:
        return None

    return ClaudeCodeSession(
        session_id=session_id,
        source_file=path,
        cwd=cwd or path.parent.name,
        git_branch=git_branch,
        version=version,
        entrypoint=entrypoint,
        started_at=started_at,
        ended_at=ended_at,
        turn_count=len(turns),
        usage_by_model=dict(usage_by_model),
        total_usage=total_usage,
        first_user_text=first_user_text,
        turns=turns,
        is_sidechain_session=any_sidechain and all_sidechain,
    )


@dataclass
class ClaudeCodeScanResult:
    """Aggregate result of scanning all Claude Code transcripts."""

    sessions: list[ClaudeCodeSession]
    total_files_seen: int
    projects_dir: Path
    has_any_real_usage: bool = field(init=False)

    def __post_init__(self) -> None:
        self.has_any_real_usage = any(
            s.total_usage.input_tokens + s.total_usage.output_tokens > 0 for s in self.sessions
        )


def iter_transcript_paths(projects_dir: Path) -> list[Path]:
    """Yield every JSONL transcript path under ``<projects_dir>/*/*.jsonl``."""
    if not projects_dir.exists():
        return []
    return sorted(projects_dir.glob("*/*.jsonl"))


def scan_claude_code_projects(projects_dir: Path) -> ClaudeCodeScanResult:
    """Walk every Claude Code transcript and parse it."""
    paths = iter_transcript_paths(projects_dir)
    sessions: list[ClaudeCodeSession] = []
    for path in paths:
        session = parse_session_file(path)
        if session is not None:
            sessions.append(session)
    return ClaudeCodeScanResult(sessions=sessions, total_files_seen=len(paths), projects_dir=projects_dir)
