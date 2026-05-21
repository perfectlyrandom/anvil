"""Read per-session token counts from Cursor's main global state DB.

Cursor stores each chat turn (a "bubble") as a JSON blob in the ``cursorDiskKV``
table of ``state.vscdb``, keyed ``bubbleId:<sessionId>:<bubbleId>``. Each blob
contains a ``tokenCount`` field with ``inputTokens`` and ``outputTokens``.

This is the ground-truth count for sessions that used Cursor's billed models.
For BYOK sessions (own API key), Cursor doesn't populate it - see
https://cursor.com/help/models-and-usage/api-keys - so those bubbles record
zeros and we drop them here, letting the transcript-based estimator handle
those sessions instead.

We aggregate per ``session_id`` so the cost layer can join against
``SessionRecord.session_id`` from ``cursor_transcripts``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BubbleObservation:
    """Real measured token totals for one Cursor session."""

    session_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    bubble_count: int = 0


@dataclass
class CursorBubbleResult:
    """Aggregate result of scanning state.vscdb for bubble token counts."""

    observations: dict[str, BubbleObservation] = field(default_factory=dict)

    def tokens_for(self, session_id: str) -> BubbleObservation | None:
        return self.observations.get(session_id)

    @property
    def session_count(self) -> int:
        return len(self.observations)


def scan_cursor_bubbles(state_db: Path) -> CursorBubbleResult:
    """Walk ``bubbleId:*`` rows and aggregate non-zero ``tokenCount`` per session.

    Returns an empty result if the DB doesn't exist or can't be opened.
    Bubbles with zero tokens (BYOK or user-only turns) are skipped, so the
    observation set covers only sessions where Cursor recorded real usage.
    """
    if not state_db.exists():
        return CursorBubbleResult()

    try:
        con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return CursorBubbleResult()

    obs: dict[str, BubbleObservation] = {}
    try:
        cursor = con.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'")
        for key, value in cursor:
            parts = key.split(":", 2)
            if len(parts) < 3:
                continue
            sid = parts[1]
            try:
                blob = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
            tc = blob.get("tokenCount")
            if not isinstance(tc, dict):
                continue
            inp = int(tc.get("inputTokens") or 0)
            out = int(tc.get("outputTokens") or 0)
            if inp + out == 0:
                continue
            entry = obs.get(sid)
            if entry is None:
                entry = BubbleObservation(session_id=sid)
                obs[sid] = entry
            entry.input_tokens += inp
            entry.output_tokens += out
            entry.bubble_count += 1
    except sqlite3.DatabaseError:
        return CursorBubbleResult(observations=obs)
    finally:
        con.close()

    return CursorBubbleResult(observations=obs)
