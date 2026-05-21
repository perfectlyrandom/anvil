"""Parser for OpenAI Codex CLI session rollouts.

Codex CLI writes one JSONL per session under::

    ~/.codex/sessions/<yyyy>/<mm>/<dd>/rollout-<iso>-<uuid>.jsonl

Each line is a structured event with::

    {"timestamp": "...", "type": "session_meta|event_msg|response_item|turn_context", "payload": {...}}

The fields we care about, by event type:

* ``session_meta.payload`` - ``id``, ``timestamp``, ``cwd``, ``originator`` ("Codex CLI" /
  "Codex Desktop"), ``model_provider`` ("openai" / "azure" / ...).
* ``turn_context.payload.model`` - the model used for that turn (e.g. ``gpt-5.5``).
* ``event_msg.payload.type == "token_count"`` - carries
  ``payload.info.total_token_usage`` (cumulative across the session) and
  ``payload.info.last_token_usage`` (this turn). Fields:
  ``input_tokens``, ``cached_input_tokens``, ``output_tokens``,
  ``reasoning_output_tokens``, ``total_tokens``.

The last ``token_count`` event in a file is authoritative for session totals.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodexTokenUsage:
    """Codex's token bucketing - slightly different from Anthropic's.

    Note: ``cached_input_tokens`` is a SUBSET of ``input_tokens`` in OpenAI's accounting
    (cached tokens are still input, just billed cheaper). We track them separately so
    cost calculators can apply the discounted rate to the cached portion.
    """

    input_tokens: int = 0  # total input, including cached
    cached_input_tokens: int = 0  # subset of input that hit the cache
    output_tokens: int = 0
    reasoning_output_tokens: int = 0  # subset of output that was hidden reasoning
    total_tokens: int = 0


@dataclass
class CodexSession:
    """One Codex CLI session, parsed from a single rollout JSONL."""

    session_id: str
    source_file: Path
    cwd: str | None
    started_at: datetime | None
    model: str | None  # the dominant model across the session's turns
    models_seen: dict[str, int]  # turn counts per model (for mixed-model sessions)
    originator: str | None  # "Codex CLI" / "Codex Desktop"
    model_provider: str | None  # "openai" / "azure" / ...
    turn_count: int
    total_usage: CodexTokenUsage
    first_user_text: str | None


@dataclass
class CodexScanResult:
    """Result of scanning the Codex sessions root."""

    sessions: list[CodexSession]
    errors: list[str] = field(default_factory=list)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_first_user_text(text: str, max_chars: int = 200) -> str:
    """Trim a user message for preview. Codex stores raw text in ``payload.message``."""
    cleaned = " ".join(text.split())
    return cleaned[:max_chars]


def parse_codex_session(path: Path) -> CodexSession | None:
    """Parse one Codex rollout JSONL file. Returns None if the file is corrupt or empty."""
    session_id: str | None = None
    cwd: str | None = None
    started_at: datetime | None = None
    originator: str | None = None
    provider: str | None = None
    models_seen: Counter[str] = Counter()
    turn_count = 0
    total_usage = CodexTokenUsage()
    first_user_text: str | None = None

    try:
        with path.open() as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                t = record.get("type")
                payload = record.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                if t == "session_meta":
                    session_id = payload.get("id") or session_id
                    cwd = payload.get("cwd") or cwd
                    started_at = _parse_timestamp(payload.get("timestamp")) or started_at
                    originator = payload.get("originator") or originator
                    provider = payload.get("model_provider") or provider
                elif t == "turn_context":
                    m = payload.get("model")
                    if isinstance(m, str):
                        models_seen[m] += 1
                elif t == "event_msg":
                    et = payload.get("type")
                    if et == "user_message" and first_user_text is None:
                        msg = payload.get("message")
                        if isinstance(msg, str) and msg.strip():
                            first_user_text = _extract_first_user_text(msg)
                    elif et == "task_complete":
                        turn_count += 1
                    elif et == "token_count":
                        # ``total_token_usage`` is cumulative across the session; always
                        # overwrite so the last event wins.
                        info = payload.get("info") or {}
                        tu = info.get("total_token_usage") or {}
                        if isinstance(tu, dict):
                            total_usage = CodexTokenUsage(
                                input_tokens=int(tu.get("input_tokens") or 0),
                                cached_input_tokens=int(tu.get("cached_input_tokens") or 0),
                                output_tokens=int(tu.get("output_tokens") or 0),
                                reasoning_output_tokens=int(tu.get("reasoning_output_tokens") or 0),
                                total_tokens=int(tu.get("total_tokens") or 0),
                            )
    except OSError as exc:
        logger.warning("codex parser: skipping %s (%s)", path, exc)
        return None

    if session_id is None:
        # File didn't have a session_meta line; skip it.
        return None

    dominant_model = models_seen.most_common(1)[0][0] if models_seen else None
    return CodexSession(
        session_id=session_id,
        source_file=path,
        cwd=cwd,
        started_at=started_at,
        model=dominant_model,
        models_seen=dict(models_seen),
        originator=originator,
        model_provider=provider,
        turn_count=turn_count,
        total_usage=total_usage,
        first_user_text=first_user_text,
    )


def scan_codex_sessions(sessions_root: Path) -> CodexScanResult:
    """Walk ``~/.codex/sessions`` and parse every rollout JSONL we find."""
    if not sessions_root.exists() or not sessions_root.is_dir():
        return CodexScanResult(sessions=[])
    sessions: list[CodexSession] = []
    errors: list[str] = []
    for jsonl in sorted(sessions_root.rglob("rollout-*.jsonl")):
        parsed = parse_codex_session(jsonl)
        if parsed is None:
            errors.append(f"unparseable: {jsonl}")
            continue
        sessions.append(parsed)
    return CodexScanResult(sessions=sessions, errors=errors)
