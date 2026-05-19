"""Aggregate statistics over a set of parsed Cursor sessions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from anvil.parsers.cursor_transcripts import CursorScanResult, SessionRecord


@dataclass
class WorkspaceStats:
    workspace: str
    session_count: int
    total_turns: int
    total_user_tokens: int
    total_assistant_tokens: int


@dataclass
class BucketStat:
    name: str
    total_tokens: int
    share_of_user_tokens: float
    appears_in_sessions: int


@dataclass
class CursorAggregate:
    """Aggregated cursor stats across all scanned sessions."""

    total_sessions: int
    parent_sessions: int
    subagent_sessions: int
    total_turns: int
    total_user_tokens: int
    total_assistant_tokens: int
    avg_turns_per_session: float
    median_turns_per_session: int
    workspaces: list[WorkspaceStats]
    buckets: list[BucketStat]
    longest_sessions: list[SessionRecord]


def _median(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) // 2
    return sorted_values[mid]


def aggregate(scan: CursorScanResult, *, top_sessions: int = 10) -> CursorAggregate:
    """Build a CursorAggregate from a scan result."""
    sessions = scan.sessions
    total_user = sum(s.user_tokens_est for s in sessions)
    total_assistant = sum(s.assistant_tokens_est for s in sessions)
    total_turns = sum(s.turn_count for s in sessions)

    # Workspace rollup
    by_ws: dict[str, list[SessionRecord]] = defaultdict(list)
    for s in sessions:
        by_ws[s.workspace].append(s)
    workspaces = [
        WorkspaceStats(
            workspace=ws,
            session_count=len(items),
            total_turns=sum(i.turn_count for i in items),
            total_user_tokens=sum(i.user_tokens_est for i in items),
            total_assistant_tokens=sum(i.assistant_tokens_est for i in items),
        )
        for ws, items in by_ws.items()
    ]
    workspaces.sort(key=lambda w: w.total_user_tokens + w.total_assistant_tokens, reverse=True)

    # Bucket rollup
    bucket_totals: dict[str, int] = defaultdict(int)
    bucket_appearances: dict[str, int] = defaultdict(int)
    for s in sessions:
        for bucket, tokens in s.bucket_tokens.items():
            bucket_totals[bucket] += tokens
            if tokens > 0:
                bucket_appearances[bucket] += 1
    buckets = [
        BucketStat(
            name=name,
            total_tokens=tokens,
            share_of_user_tokens=(tokens / total_user) if total_user else 0.0,
            appears_in_sessions=bucket_appearances[name],
        )
        for name, tokens in bucket_totals.items()
    ]
    buckets.sort(key=lambda b: b.total_tokens, reverse=True)

    turn_counts = [s.turn_count for s in sessions]
    longest = sorted(
        sessions,
        key=lambda s: s.user_tokens_est + s.assistant_tokens_est,
        reverse=True,
    )[:top_sessions]

    return CursorAggregate(
        total_sessions=len(sessions),
        parent_sessions=len(scan.parent_sessions),
        subagent_sessions=len(scan.subagent_sessions),
        total_turns=total_turns,
        total_user_tokens=total_user,
        total_assistant_tokens=total_assistant,
        avg_turns_per_session=(total_turns / len(sessions)) if sessions else 0.0,
        median_turns_per_session=_median(turn_counts),
        workspaces=workspaces,
        buckets=buckets,
        longest_sessions=longest,
    )
