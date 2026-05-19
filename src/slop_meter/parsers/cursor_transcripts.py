"""Parser for Cursor's agent-transcripts JSONL files.

Cursor stores chat history under::

    ~/.cursor/projects/<encoded-cwd>/agent-transcripts/<session-id>/<session-id>.jsonl

Each line is a JSON object with::

    {"role": "user|assistant", "message": {"content": [{"type": "text", "text": "..."}]}}

User messages contain Cursor-injected sections wrapped in XML-style tags
(``<user_query>``, ``<system_reminder>``, ``<always_applied_workspace_rule>``,
``<attached_files>``, ``<code_selection>``, ``<external_links>``, ``<agent_skill>``,
``<open_and_recently_viewed_files>``, etc.). Splitting on these tags lets us attribute
tokens to named buckets - the v0.2 wedge.

Cursor's JSONL does NOT include timestamps, model identifiers, or real token counts;
we estimate tokens with ``tiktoken`` (cl100k_base) which is close enough for relative
comparison across buckets but not authoritative.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

_ENCODER: tiktoken.Encoding | None = None


def _encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """Estimate token count via cl100k_base. Approximate for cross-bucket comparison only."""
    if not text:
        return 0
    return len(_encoder().encode(text, disallowed_special=()))


# Buckets we attribute user-side tokens to. Order matters - earlier patterns are tried first.
_TAG_BUCKETS: list[tuple[str, str]] = [
    ("user_query", "user_query"),
    ("always_applied_workspace_rule", "workspace_rule"),
    ("always_applied_workspace_rules", "workspace_rule"),
    ("agent_requestable_workspace_rule", "workspace_rule"),
    ("agent_requestable_workspace_rules", "workspace_rule"),
    ("user_rule", "user_rule"),
    ("user_rules", "user_rule"),
    ("agent_skill", "agent_skill"),
    ("agent_skills", "agent_skill"),
    ("available_skills", "agent_skill"),
    ("attached_files", "attached_files"),
    ("code_selection", "code_selection"),
    ("terminal_selection", "terminal_selection"),
    ("external_links", "external_links"),
    ("image_files", "image_files"),
    ("open_and_recently_viewed_files", "open_files"),
    ("system_reminder", "system_reminder"),
    ("git_status", "git_status"),
    ("agent_transcripts", "agent_transcripts"),
    ("mcp_file_system", "mcp"),
    ("mcp_instructions", "mcp"),
    ("agent_transcripts_for_other_session", "agent_transcripts"),
]

BUCKET_NAMES = sorted({bucket for _, bucket in _TAG_BUCKETS})

# Matches <tag>...</tag> (greedy across newlines). Non-greedy on inner content.
_TAG_RE = re.compile(r"<([a-z][a-z0-9_]*)(?:\s[^>]*)?>(.*?)</\1>", re.DOTALL)


@dataclass
class TurnRecord:
    """One assistant-or-user turn."""

    turn_index: int
    role: str
    raw_text: str
    estimated_tokens: int
    bucket_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class SessionRecord:
    """One Cursor chat session, parsed from a single JSONL file."""

    session_id: str
    source_file: Path
    workspace: str
    turn_count: int
    user_turn_count: int
    assistant_turn_count: int
    user_tokens_est: int
    assistant_tokens_est: int
    bucket_tokens: dict[str, int]
    first_user_query: str | None
    turns: list[TurnRecord]
    file_bytes: int
    is_subagent: bool


def _attribute_user_message(text: str) -> tuple[dict[str, int], int]:
    """Walk a user message, attribute tokens to buckets by tag.

    Returns (bucket_tokens, total_tokens). Unattributed content goes to ``other``.
    """
    buckets: dict[str, int] = defaultdict(int)
    total = estimate_tokens(text)
    remaining = text

    tag_to_bucket = {tag: bucket for tag, bucket in _TAG_BUCKETS}

    # Pull out every matched tag block, sum tokens per bucket, strip from remaining.
    for match in _TAG_RE.finditer(text):
        tag = match.group(1).lower()
        inner = match.group(2)
        bucket = tag_to_bucket.get(tag)
        if bucket is None:
            continue
        buckets[bucket] += estimate_tokens(inner)
        remaining = remaining.replace(match.group(0), "", 1)

    # Anything left (after stripping known tag blocks) is "other" - small framing,
    # markdown formatting, or unknown tags. Likely <10% of a typical user message.
    other_tokens = estimate_tokens(remaining)
    if other_tokens > 0:
        buckets["other"] = other_tokens

    return dict(buckets), total


def _extract_first_user_query(text: str) -> str | None:
    """Pull the <user_query>...</user_query> block out of a user message, if present."""
    match = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _workspace_from_path(path: Path) -> str:
    """Decode a Cursor-encoded workspace path like ``Users-pratyushaduvvuri-Desktop-Galileo-api``."""
    # Cursor encodes / as -. We can't perfectly reverse this (paths can contain - too)
    # but for display purposes the encoded form is fine.
    parts = path.parts
    try:
        idx = parts.index("projects")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return path.parent.name


def parse_session_file(path: Path) -> SessionRecord | None:
    """Parse a single Cursor JSONL transcript. Returns None if the file is empty / unreadable."""
    if not path.exists() or path.stat().st_size == 0:
        return None

    turns: list[TurnRecord] = []
    bucket_totals: dict[str, int] = defaultdict(int)
    user_tokens_total = 0
    assistant_tokens_total = 0
    first_user_query: str | None = None

    with path.open("r", encoding="utf-8") as fh:
        for turn_index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role", "")
            text = "".join(
                str(item.get("text", ""))
                for item in obj.get("message", {}).get("content", [])
                if isinstance(item, dict) and item.get("type") == "text"
            )
            tokens = estimate_tokens(text)

            if role == "user":
                buckets, _ = _attribute_user_message(text)
                user_tokens_total += tokens
                for bucket, count in buckets.items():
                    bucket_totals[bucket] += count
                if first_user_query is None:
                    first_user_query = _extract_first_user_query(text)
                turns.append(
                    TurnRecord(
                        turn_index=turn_index,
                        role=role,
                        raw_text=text,
                        estimated_tokens=tokens,
                        bucket_tokens=buckets,
                    )
                )
            elif role == "assistant":
                assistant_tokens_total += tokens
                turns.append(
                    TurnRecord(
                        turn_index=turn_index,
                        role=role,
                        raw_text=text,
                        estimated_tokens=tokens,
                    )
                )

    user_turns = [t for t in turns if t.role == "user"]
    assistant_turns = [t for t in turns if t.role == "assistant"]

    session_id = path.stem
    is_subagent = "subagents" in path.parts

    return SessionRecord(
        session_id=session_id,
        source_file=path,
        workspace=_workspace_from_path(path),
        turn_count=len(turns),
        user_turn_count=len(user_turns),
        assistant_turn_count=len(assistant_turns),
        user_tokens_est=user_tokens_total,
        assistant_tokens_est=assistant_tokens_total,
        bucket_tokens=dict(bucket_totals),
        first_user_query=first_user_query,
        turns=turns,
        file_bytes=path.stat().st_size,
        is_subagent=is_subagent,
    )


def iter_transcript_paths(projects_dir: Path) -> list[Path]:
    """Yield every JSONL transcript path under ``<projects_dir>/*/agent-transcripts/``.

    Includes nested subagent transcripts under ``.../<session>/subagents/<subagent>.jsonl``.
    """
    if not projects_dir.exists():
        return []
    return sorted(projects_dir.glob("*/agent-transcripts/**/*.jsonl"))


@dataclass
class CursorScanResult:
    """Aggregate result of scanning all Cursor transcripts."""

    sessions: list[SessionRecord]
    total_files_seen: int

    @property
    def parent_sessions(self) -> list[SessionRecord]:
        return [s for s in self.sessions if not s.is_subagent]

    @property
    def subagent_sessions(self) -> list[SessionRecord]:
        return [s for s in self.sessions if s.is_subagent]


def scan_cursor_projects(projects_dir: Path) -> CursorScanResult:
    """Walk every Cursor transcript and parse it. Returns aggregate scan result."""
    paths = iter_transcript_paths(projects_dir)
    sessions: list[SessionRecord] = []
    for path in paths:
        session = parse_session_file(path)
        if session is not None:
            sessions.append(session)
    return CursorScanResult(sessions=sessions, total_files_seen=len(paths))
