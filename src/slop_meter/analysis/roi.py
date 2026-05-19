"""ROI correlation: AI cost ↔ shipping signal.

This is intentionally NOT a single ROI number. The METR study (2025) showed AI tools
can make experienced engineers feel faster while measurably slowing them down -
collapsing this into one metric encourages exactly the kind of misuse we want to
avoid. Instead we bucket by week, surface per-week cost vs PRs-merged, and let the
user calibrate against their own felt productivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from slop_meter.connectors.github import PullRequestRecord
from slop_meter.parsers.claude_code import ClaudeCodeScanResult


def _iso_week_key(when: datetime) -> str:
    """ISO week key like '2026-W19'. Always treats input as UTC for stable bucketing."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    iso_year, iso_week, _ = when.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _week_start(when: datetime) -> datetime:
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    weekday = when.weekday()  # Monday=0
    return (when - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass
class WeekBucket:
    iso_week: str
    week_start: datetime
    claude_code_sessions: int
    claude_code_input_tokens: int
    claude_code_output_tokens: int
    claude_code_estimated_cost_usd: float
    prs_merged: int
    prs_opened: int
    lines_changed_merged: int
    cost_per_merged_pr: float | None
    cost_per_loc_merged: float | None


@dataclass
class RoiReport:
    """Per-week rollup of AI cost vs PR shipping."""

    weeks: list[WeekBucket]
    earliest_week: str | None
    latest_week: str | None
    total_estimated_cost_usd: float
    total_merged_prs: int


def correlate(
    claude_scan: ClaudeCodeScanResult | None,
    prs: list[PullRequestRecord],
    *,
    cost_per_session: dict[str, float] | None = None,
) -> RoiReport:
    """Cross-correlate Claude Code sessions and merged PRs by ISO week.

    ``cost_per_session`` maps session_id → dollar cost (typically from
    ``analysis/claude_cost.py``). If absent, cost columns are zero but token / PR
    columns still work.
    """
    cost_per_session = cost_per_session or {}

    buckets: dict[str, WeekBucket] = {}

    def _ensure(when: datetime) -> WeekBucket:
        key = _iso_week_key(when)
        if key not in buckets:
            buckets[key] = WeekBucket(
                iso_week=key,
                week_start=_week_start(when),
                claude_code_sessions=0,
                claude_code_input_tokens=0,
                claude_code_output_tokens=0,
                claude_code_estimated_cost_usd=0.0,
                prs_merged=0,
                prs_opened=0,
                lines_changed_merged=0,
                cost_per_merged_pr=None,
                cost_per_loc_merged=None,
            )
        return buckets[key]

    if claude_scan is not None:
        for session in claude_scan.sessions:
            ts = session.started_at or session.ended_at
            if ts is None:
                continue
            bucket = _ensure(ts)
            bucket.claude_code_sessions += 1
            bucket.claude_code_input_tokens += session.total_usage.input_tokens
            bucket.claude_code_output_tokens += session.total_usage.output_tokens
            bucket.claude_code_estimated_cost_usd += cost_per_session.get(session.session_id, 0.0)

    for pr in prs:
        opened = _ensure(pr.created_at)
        opened.prs_opened += 1
        if pr.merged_at is not None:
            merged = _ensure(pr.merged_at)
            merged.prs_merged += 1
            merged.lines_changed_merged += pr.total_lines_changed

    for bucket in buckets.values():
        if bucket.prs_merged > 0:
            bucket.cost_per_merged_pr = bucket.claude_code_estimated_cost_usd / bucket.prs_merged
        if bucket.lines_changed_merged > 0:
            bucket.cost_per_loc_merged = bucket.claude_code_estimated_cost_usd / bucket.lines_changed_merged

    ordered = sorted(buckets.values(), key=lambda b: b.week_start)
    return RoiReport(
        weeks=ordered,
        earliest_week=ordered[0].iso_week if ordered else None,
        latest_week=ordered[-1].iso_week if ordered else None,
        total_estimated_cost_usd=sum(b.claude_code_estimated_cost_usd for b in ordered),
        total_merged_prs=sum(b.prs_merged for b in ordered),
    )
