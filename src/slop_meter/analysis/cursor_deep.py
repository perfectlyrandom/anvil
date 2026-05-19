"""Deeper Cursor analysis: mid-session sampling and cross-session bloat detection.

``cursor_stats.py`` handles first-pass aggregates. This module adds:

- **Mid-session sampling**: pick representative turns from inside long sessions, not
  just the opening prompt. This catches issues like long-conversation drift, late
  attached-file dumps, and repetitive corrective prompts.
- **Cross-session bloat detection**: identify buckets that consistently eat tokens
  across many sessions (the persistent leak) vs. one-off heavy users (a single
  session's worth of attached files).
- **Repeated-prompt detection**: surface near-duplicate opening prompts across
  sessions - the "David Mallon" pattern the LLM analyzer flagged.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median

from slop_meter.parsers.cursor_transcripts import CursorScanResult, SessionRecord, TurnRecord


@dataclass
class MidSessionSample:
    """A turn sampled from the middle of a long session."""

    session_id: str
    workspace: str
    turn_index: int
    role: str
    estimated_tokens: int
    text_preview: str  # first ~300 chars, single-line
    duplicate_count: int = 1  # how many sessions had this same content


@dataclass
class BloatBucket:
    """A bucket that's expensive across many sessions (the consistent-leak pattern)."""

    name: str
    median_tokens_per_session: int
    p95_tokens_per_session: int
    appears_in_sessions: int
    total_tokens: int


@dataclass
class RepeatedPromptCluster:
    """A cluster of sessions whose opening prompts are near-duplicates."""

    canonical_first_query: str
    session_ids: list[str]
    total_user_tokens_burned: int


@dataclass
class CursorDeepReport:
    mid_session_samples: list[MidSessionSample]
    bloat_buckets: list[BloatBucket]
    repeated_prompt_clusters: list[RepeatedPromptCluster]


def _normalize_prompt(text: str) -> str:
    """Cheap normalization for dedup: lowercase, collapse whitespace, strip URLs/UUIDs."""
    text = text.lower()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<uuid>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _prompt_signature(text: str) -> str:
    """Hash of the first ~200 chars of normalized text. Catches near-duplicate openings."""
    normalized = _normalize_prompt(text)[:200]
    return hashlib.sha1(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    k = int(round((len(sorted_values) - 1) * pct))
    return sorted_values[k]


def sample_mid_session_turns(
    sessions: list[SessionRecord], *, samples_per_session: int = 2, min_turn_tokens: int = 200
) -> list[MidSessionSample]:
    """Pull 1-2 high-token turns from the middle 60% of each long session.

    Deduplicates near-identical pasted blobs (same opening 200-char signature) so a
    base64 audio dump that appears in 8 sessions shows up once with ``duplicate_count=8``
    instead of polluting the output. "Middle 60%" because opening prompts are already
    captured by ``first_user_query`` and trailing turns are often short wrap-ups.
    """
    by_signature: dict[str, MidSessionSample] = {}
    for session in sessions:
        if session.turn_count < 10:
            continue
        start = int(session.turn_count * 0.2)
        end = int(session.turn_count * 0.8)
        eligible: list[TurnRecord] = [
            t for t in session.turns[start:end] if t.role == "user" and t.estimated_tokens >= min_turn_tokens
        ]
        if not eligible:
            continue
        eligible.sort(key=lambda t: t.estimated_tokens, reverse=True)
        for turn in eligible[:samples_per_session]:
            preview = re.sub(r"\s+", " ", turn.raw_text)[:300]
            signature = _prompt_signature(turn.raw_text)
            existing = by_signature.get(signature)
            if existing is None:
                by_signature[signature] = MidSessionSample(
                    session_id=session.session_id,
                    workspace=session.workspace,
                    turn_index=turn.turn_index,
                    role=turn.role,
                    estimated_tokens=turn.estimated_tokens,
                    text_preview=preview,
                )
            else:
                existing.duplicate_count += 1
                if turn.estimated_tokens > existing.estimated_tokens:
                    existing.estimated_tokens = turn.estimated_tokens
    samples = list(by_signature.values())
    samples.sort(key=lambda s: s.estimated_tokens * s.duplicate_count, reverse=True)
    return samples


def detect_bloat_buckets(sessions: list[SessionRecord], *, min_sessions: int = 5) -> list[BloatBucket]:
    """Identify buckets that consistently consume tokens across many sessions."""
    per_bucket_session_tokens: dict[str, list[int]] = defaultdict(list)
    for session in sessions:
        for bucket, tokens in session.bucket_tokens.items():
            if tokens > 0:
                per_bucket_session_tokens[bucket].append(tokens)

    bloat: list[BloatBucket] = []
    for bucket, tokens_per_session in per_bucket_session_tokens.items():
        if len(tokens_per_session) < min_sessions:
            continue
        bloat.append(
            BloatBucket(
                name=bucket,
                median_tokens_per_session=int(median(tokens_per_session)),
                p95_tokens_per_session=_percentile(tokens_per_session, 0.95),
                appears_in_sessions=len(tokens_per_session),
                total_tokens=sum(tokens_per_session),
            )
        )
    bloat.sort(key=lambda b: b.median_tokens_per_session * b.appears_in_sessions, reverse=True)
    return bloat


def find_repeated_prompts(sessions: list[SessionRecord], *, min_cluster_size: int = 3) -> list[RepeatedPromptCluster]:
    """Cluster sessions by near-identical opening prompts."""
    by_signature: dict[str, list[SessionRecord]] = defaultdict(list)
    for session in sessions:
        if not session.first_user_query:
            continue
        by_signature[_prompt_signature(session.first_user_query)].append(session)

    clusters: list[RepeatedPromptCluster] = []
    for cluster in by_signature.values():
        if len(cluster) < min_cluster_size:
            continue
        canonical = max(cluster, key=lambda s: len(s.first_user_query or ""))
        clusters.append(
            RepeatedPromptCluster(
                canonical_first_query=(canonical.first_user_query or "")[:200],
                session_ids=[s.session_id for s in cluster],
                total_user_tokens_burned=sum(s.user_tokens_est for s in cluster),
            )
        )
    clusters.sort(key=lambda c: c.total_user_tokens_burned, reverse=True)
    return clusters


def deep_analyze(scan: CursorScanResult, *, mid_sample_top_n: int = 15) -> CursorDeepReport:
    """Run the full deep analysis pipeline."""
    parent_sessions = [s for s in scan.sessions if not s.is_subagent]
    return CursorDeepReport(
        mid_session_samples=sample_mid_session_turns(parent_sessions)[:mid_sample_top_n],
        bloat_buckets=detect_bloat_buckets(parent_sessions),
        repeated_prompt_clusters=find_repeated_prompts(parent_sessions),
    )


# Silence unused-import warnings when importing aggregates in tests.
_ = Counter
