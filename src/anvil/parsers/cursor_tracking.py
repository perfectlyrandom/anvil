"""Read Cursor's local AI-tracking SQLite DB to recover model attribution.

Cursor ships a tiny SQLite database at ``~/.cursor/ai-tracking/ai-code-tracking.db`` that
records every accepted AI code chunk. Each row has the conversationId, the model used
for that turn (e.g. ``claude-opus-4-7-thinking-xhigh``), and a timestamp.

This is the only on-disk signal we have for which Cursor session used which model.
Coverage is partial: only conversations that wrote code into the buffer leave rows,
so read-only chats (Q&A, exploration) won't show up. But for the ones we can match,
we get a real model name to price against.

The conversationId in the DB matches the basename of Cursor's transcript JSONL files
(verified empirically), which lets us join in :mod:`anvil.analysis.cost_breakdown`.

Cursor uses non-canonical model identifiers like ``claude-opus-4-7-thinking-xhigh`` -
we normalize those to the keys used in :mod:`anvil.analysis.pricing` before pricing.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Cursor's model identifiers are messy and aspirational ("claude-opus-4-7" doesn't exist
# yet at Anthropic's published pricing, "high-thinking" and "xhigh" are Cursor-side effort
# tiers). We canonicalize to the closest billable model whose pricing we know. When in
# doubt, prefer the *more expensive* canonical so we don't under-report spend.
_MODEL_NORMALIZE: list[tuple[str, str]] = [
    # Anthropic Opus family - every Cursor opus variant priced at current Opus rate
    ("claude-opus-4-7", "claude-opus-4-5"),
    ("claude-opus-4-6", "claude-opus-4-5"),
    ("claude-opus-4-5", "claude-opus-4-5"),
    ("claude-4.6-opus", "claude-opus-4-5"),
    ("claude-4.7-opus", "claude-opus-4-5"),
    # Anthropic Sonnet family
    ("claude-sonnet-4-7", "claude-sonnet-4-5"),
    ("claude-sonnet-4-6", "claude-sonnet-4-5"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5"),
    ("claude-4.6-sonnet", "claude-sonnet-4-5"),
    ("claude-4.7-sonnet", "claude-sonnet-4-5"),
    # Anthropic Haiku
    ("claude-haiku-4-5", "claude-haiku-4-5"),
    ("claude-4.6-haiku", "claude-haiku-4-5"),
    # OpenAI GPT-5 family (Cursor's "gpt-5.5-medium" / "gpt-5.5-high" all bill same)
    ("gpt-5.5", "gpt-5.5"),
    ("gpt-5-mini", "gpt-5-mini"),
    ("gpt-5", "gpt-5"),
]


def normalize_cursor_model(raw: str | None) -> str | None:
    """Map Cursor's noisy model identifier to a key our pricing table understands.

    Returns None for ``default`` or unrecognized values so callers can fall back to the
    user-configured ``default_pricing_model``.
    """
    if not raw or raw == "default":
        return None
    lowered = raw.lower()
    for prefix, canonical in _MODEL_NORMALIZE:
        if lowered.startswith(prefix):
            return canonical
    return None  # unknown - let caller fall back to default


@dataclass
class CursorModelObservation:
    """Per-conversation model attribution, derived from the tracking DB."""

    conversation_id: str
    dominant_model: str | None  # normalized, ready to feed into pricing
    raw_model: str | None  # what Cursor actually wrote, for debug / UI
    chunk_count: int  # how many code chunks we saw for this conversation
    earliest_ts_ms: int
    latest_ts_ms: int


@dataclass
class CursorTrackingResult:
    """All the conversations whose model we could resolve from the DB."""

    observations: dict[str, CursorModelObservation] = field(default_factory=dict)
    # Distribution of canonical models across all chunks, weighted by chunk count.
    # Used as a calibrated "best guess" for sessions where we have no DB entry.
    model_distribution: Counter[str] = field(default_factory=Counter)

    def model_for(self, conversation_id: str) -> str | None:
        """Convenience lookup for the join in cost_breakdown."""
        obs = self.observations.get(conversation_id)
        return obs.dominant_model if obs else None


def scan_cursor_tracking(db_path: Path) -> CursorTrackingResult:
    """Read the Cursor AI-tracking SQLite and roll up to per-conversation model attribution.

    Returns an empty result (no exception) if the DB doesn't exist or the schema changed.
    Cursor isn't a public API; we want to fail soft.
    """
    if not db_path.exists():
        return CursorTrackingResult()
    result = CursorTrackingResult()
    try:
        # Read-only sqlite open - Cursor itself may have the DB open for writes.
        uri = f"file:{db_path}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=1.0)
        cur = con.cursor()
        rows = cur.execute(
            "SELECT conversationId, model, COUNT(*), MIN(timestamp), MAX(timestamp) "
            "FROM ai_code_hashes "
            "WHERE source = 'composer' AND conversationId IS NOT NULL "
            "GROUP BY conversationId, model"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("cursor tracking DB unreadable: %s", exc)
        return CursorTrackingResult()

    # Two-pass: first collect per-(conv, model) chunk counts, then pick the dominant
    # model per conv. This handles conversations that switched models mid-session.
    per_conv_counts: dict[str, Counter[tuple[str, str | None]]] = {}
    earliest: dict[str, int] = {}
    latest: dict[str, int] = {}
    for conv_id, raw_model, count, ts_min, ts_max in rows:
        per_conv_counts.setdefault(conv_id, Counter())[(raw_model or "", raw_model)] += count
        earliest[conv_id] = min(earliest.get(conv_id, ts_min), ts_min)
        latest[conv_id] = max(latest.get(conv_id, ts_max), ts_max)
        canonical = normalize_cursor_model(raw_model)
        if canonical:
            result.model_distribution[canonical] += count

    for conv_id, counter in per_conv_counts.items():
        (_key, raw_model), _count = counter.most_common(1)[0]
        canonical = normalize_cursor_model(raw_model)
        result.observations[conv_id] = CursorModelObservation(
            conversation_id=conv_id,
            dominant_model=canonical,
            raw_model=raw_model,
            chunk_count=sum(counter.values()),
            earliest_ts_ms=earliest[conv_id],
            latest_ts_ms=latest[conv_id],
        )
    return result
