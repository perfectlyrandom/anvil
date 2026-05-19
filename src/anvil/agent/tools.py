"""Read-only tools the AI agent can call to inspect the user's local data.

Each tool returns a small, JSON-serializable result so the model can reason over
it. Tools intentionally cap result sizes so the agent doesn't blow context.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anvil.analysis.cost_breakdown import build_cost_breakdown
from anvil.analysis.cursor_deep import deep_analyze
from anvil.analysis.cursor_stats import aggregate
from anvil.analysis.pricing import estimate_cost
from anvil.analysis.roi import correlate
from anvil.connectors.github import (
    GitHubConnectorError,
    default_since,
    fetch_authored_prs,
)
from anvil.parsers.claude_code import scan_claude_code_projects
from anvil.parsers.codex import CodexScanResult, scan_codex_sessions
from anvil.parsers.cursor_tracking import CursorTrackingResult, scan_cursor_tracking
from anvil.parsers.cursor_transcripts import (
    CursorScanResult,
    SessionRecord,
    scan_cursor_projects,
)


@dataclass
class ToolContext:
    """Shared, cached state the agent's tools read from.

    Scans are slow (4M+ tokens of JSONL) - we scan once at agent creation time and
    reuse for the lifetime of the chat. The frontend can call ``invalidate()`` to
    refresh after long-running work.
    """

    cursor_projects_dir: Path
    claude_projects_dir: Path
    codex_sessions_dir: Path | None = None
    cursor_tracking_db: Path | None = None
    default_pricing_model: str = "claude-opus-4-5"
    github_login: str | None = None
    _cursor_scan: CursorScanResult | None = None
    _codex_scan: CodexScanResult | None = None
    _cursor_tracking: CursorTrackingResult | None = None

    def cursor_scan(self) -> CursorScanResult:
        if self._cursor_scan is None:
            self._cursor_scan = scan_cursor_projects(self.cursor_projects_dir)
        return self._cursor_scan

    def codex_scan(self) -> CodexScanResult:
        if self._codex_scan is None:
            if self.codex_sessions_dir is None:
                self._codex_scan = CodexScanResult(sessions=[])
            else:
                self._codex_scan = scan_codex_sessions(self.codex_sessions_dir)
        return self._codex_scan

    def cursor_tracking(self) -> CursorTrackingResult:
        if self._cursor_tracking is None:
            if self.cursor_tracking_db is None:
                self._cursor_tracking = CursorTrackingResult()
            else:
                self._cursor_tracking = scan_cursor_tracking(self.cursor_tracking_db)
        return self._cursor_tracking

    def invalidate(self) -> None:
        self._cursor_scan = None
        self._codex_scan = None
        self._cursor_tracking = None


def tool_summarize_cursor(ctx: ToolContext) -> dict[str, Any]:
    """Return the same overview the CLI shows: session counts, bucket attribution, top workspaces."""
    agg = aggregate(ctx.cursor_scan())
    return {
        "total_sessions": agg.total_sessions,
        "parent_sessions": agg.parent_sessions,
        "subagent_sessions": agg.subagent_sessions,
        "total_turns": agg.total_turns,
        "avg_turns_per_session": round(agg.avg_turns_per_session, 1),
        "median_turns_per_session": agg.median_turns_per_session,
        "total_user_tokens_est": agg.total_user_tokens,
        "total_assistant_tokens_est": agg.total_assistant_tokens,
        "buckets": [
            {
                "name": b.name,
                "total_tokens": b.total_tokens,
                "share_of_user_input": round(b.share_of_user_tokens, 4),
                "appears_in_sessions": b.appears_in_sessions,
            }
            for b in agg.buckets
        ],
        "top_workspaces": [
            {
                "workspace": w.workspace,
                "session_count": w.session_count,
                "total_user_tokens": w.total_user_tokens,
                "total_assistant_tokens": w.total_assistant_tokens,
            }
            for w in agg.workspaces[:5]
        ],
    }


def tool_repeated_prompts(ctx: ToolContext, min_cluster_size: int = 3) -> dict[str, Any]:
    """Return clusters of sessions whose opening prompts are near-duplicates."""
    deep = deep_analyze(ctx.cursor_scan())
    clusters = [c for c in deep.repeated_prompt_clusters if len(c.session_ids) >= min_cluster_size]
    return {
        "cluster_count": len(clusters),
        "clusters": [
            {
                "session_count": len(c.session_ids),
                "user_tokens_burned_estimate": c.total_user_tokens_burned,
                "canonical_prompt": c.canonical_first_query,
                "session_ids": c.session_ids[:10],
            }
            for c in clusters[:15]
        ],
    }


def tool_top_sessions(ctx: ToolContext, n: int = 10, sort_by: str = "tokens") -> dict[str, Any]:
    """List the largest Cursor sessions.

    ``sort_by`` is ``tokens`` (user + assistant token estimate) or ``turns``.
    """
    sessions = ctx.cursor_scan().sessions
    if sort_by == "turns":
        sorted_sessions = sorted(sessions, key=lambda s: s.turn_count, reverse=True)
    else:
        sorted_sessions = sorted(sessions, key=lambda s: s.user_tokens_est + s.assistant_tokens_est, reverse=True)
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "workspace": s.workspace,
                "turn_count": s.turn_count,
                "user_tokens_est": s.user_tokens_est,
                "assistant_tokens_est": s.assistant_tokens_est,
                "first_query": (s.first_user_query or "")[:300],
                "is_subagent": s.is_subagent,
            }
            for s in sorted_sessions[:n]
        ]
    }


def tool_search_sessions(ctx: ToolContext, keyword: str, limit: int = 20) -> dict[str, Any]:
    """Find Cursor sessions whose opening prompt contains ``keyword`` (case-insensitive)."""
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches: list[SessionRecord] = [
        s for s in ctx.cursor_scan().sessions if s.first_user_query and pattern.search(s.first_user_query)
    ]
    matches.sort(key=lambda s: s.user_tokens_est, reverse=True)
    return {
        "keyword": keyword,
        "match_count": len(matches),
        "matches": [
            {
                "session_id": s.session_id,
                "workspace": s.workspace,
                "user_tokens_est": s.user_tokens_est,
                "first_query": (s.first_user_query or "")[:300],
            }
            for s in matches[:limit]
        ],
    }


def tool_bucket_bloat(ctx: ToolContext) -> dict[str, Any]:
    """Median + p95 tokens per session for each bucket. Surfaces persistent context leaks."""
    deep = deep_analyze(ctx.cursor_scan())
    return {
        "buckets": [
            {
                "name": b.name,
                "median_tokens_per_session": b.median_tokens_per_session,
                "p95_tokens_per_session": b.p95_tokens_per_session,
                "appears_in_sessions": b.appears_in_sessions,
                "total_tokens": b.total_tokens,
            }
            for b in deep.bloat_buckets
        ]
    }


def tool_github_prs(ctx: ToolContext, days: int = 90, limit: int = 25) -> dict[str, Any]:
    """Fetch the user's recent GitHub PRs via the gh CLI."""
    try:
        prs = fetch_authored_prs(ctx.github_login, since=default_since(days))
    except GitHubConnectorError as exc:
        return {"error": str(exc), "prs": []}
    merged = [p for p in prs if p.shipped]
    return {
        "days_window": days,
        "total_found": len(prs),
        "merged_count": len(merged),
        "open_count": sum(1 for p in prs if p.state == "OPEN"),
        "total_lines_changed_merged": sum(p.total_lines_changed for p in merged),
        "recent_prs": [
            {
                "repo": p.repo,
                "number": p.number,
                "title": p.title,
                "state": p.state,
                "merged_at": p.merged_at.isoformat() if p.merged_at else None,
                "additions": p.additions,
                "deletions": p.deletions,
                "url": p.url,
            }
            for p in sorted(prs, key=lambda x: x.merged_at or x.created_at, reverse=True)[:limit]
        ],
    }


def tool_roi(ctx: ToolContext, days: int = 90) -> dict[str, Any]:
    """Per-ISO-week correlation of Claude Code cost vs merged GitHub PRs."""
    claude_scan = scan_claude_code_projects(ctx.claude_projects_dir)
    try:
        prs = fetch_authored_prs(ctx.github_login, since=default_since(days))
    except GitHubConnectorError as exc:
        return {"error": str(exc), "weeks": []}

    cost_per_session: dict[str, float] = {}
    for s in claude_scan.sessions:
        cost_per_session[s.session_id] = sum(estimate_cost(usage, model) for model, usage in s.usage_by_model.items())
    report = correlate(claude_scan, prs, cost_per_session=cost_per_session)
    return {
        "total_estimated_cost_usd": round(report.total_estimated_cost_usd, 2),
        "total_merged_prs": report.total_merged_prs,
        "active_weeks": len(report.weeks),
        "weeks": [
            {
                "iso_week": w.iso_week,
                "sessions": w.claude_code_sessions,
                "input_tokens": w.claude_code_input_tokens,
                "output_tokens": w.claude_code_output_tokens,
                "estimated_cost_usd": round(w.claude_code_estimated_cost_usd, 2),
                "prs_opened": w.prs_opened,
                "prs_merged": w.prs_merged,
                "lines_changed_merged": w.lines_changed_merged,
                "cost_per_merged_pr": (round(w.cost_per_merged_pr, 2) if w.cost_per_merged_pr is not None else None),
            }
            for w in report.weeks
        ],
    }


def tool_cost_breakdown(ctx: ToolContext) -> dict[str, Any]:
    """Unified cost across Cursor + Claude Code + Codex CLI, segmented by source/model/project.

    Cursor is volume-only (no model in logs); Claude Code and Codex are priced from their
    token logs. Includes cache utilization stats and model-swap recommendations.
    """
    cursor = ctx.cursor_scan()
    claude = scan_claude_code_projects(ctx.claude_projects_dir)
    codex = ctx.codex_scan()
    tracking = ctx.cursor_tracking()
    b = build_cost_breakdown(
        cursor,
        claude,
        codex,
        cursor_tracking=tracking,
        fallback_model=ctx.default_pricing_model,
    )
    return {
        "grand_total_usd": round(b.grand_total_usd, 4),
        "totals_by_source": {k: round(v, 4) for k, v in b.totals_by_source.items()},
        "totals_by_provider": {k: round(v, 4) for k, v in b.totals_by_provider.items()},
        "totals_by_model": {k: round(v, 4) for k, v in b.totals_by_model.items()},
        "cursor_sessions_unpriced": b.cursor_sessions_unpriced,
        "cache": {
            "hit_rate": round(b.cache.hit_rate, 4),
            "cacheable_input_tokens": b.cache.cacheable_input_tokens,
            "cached_input_tokens": b.cache.cached_input_tokens,
            "saved_usd": round(b.cache.estimated_cache_savings_usd, 4),
            "left_on_table_usd": round(b.cache.estimated_cache_left_on_table_usd, 4),
        },
        "top_rows": [
            {
                "source": r.source,
                "provider": r.provider,
                "model": r.model_label,
                "project": r.project,
                "sessions": r.sessions,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_hit_rate": round(r.cache_hit_rate, 4),
                "estimated_usd": round(r.estimated_usd, 4),
            }
            for r in b.rows[:15]
        ],
        "switch_recommendations": [
            {
                "project": r.project,
                "from_model": r.model_label,
                "to_model": r.cheaper_alternative_model,
                "paid_usd": round(r.estimated_usd, 4),
                "would_pay_usd": round(r.cheaper_alternative_usd or 0, 4),
                "saved_usd": round(r.potential_model_savings_usd, 4),
            }
            for r in b.switch_recommendations
        ],
    }


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    impl: Callable[..., dict[str, Any]]


def build_tool_specs() -> list[ToolSpec]:
    """Return the schema + implementation for every agent tool."""
    return [
        ToolSpec(
            name="summarize_cursor",
            description=(
                "Returns a high-level summary of the user's Cursor usage: session counts, total "
                "estimated tokens, per-bucket attribution (user_query vs attached_files vs "
                "external_links vs workspace_rule etc.), and top workspaces. Use this for "
                "general 'what does my usage look like' questions."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            impl=lambda ctx: tool_summarize_cursor(ctx),
        ),
        ToolSpec(
            name="repeated_prompts",
            description=(
                "Returns clusters of Cursor sessions whose opening prompts are near-duplicates - "
                "the 'I keep typing the same thing' pattern. Each cluster shows how many sessions "
                "share that opening and the total user tokens burned. Use this when the user asks "
                "about repetition, waste, or 'what am I doing wrong'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "min_cluster_size": {
                        "type": "integer",
                        "description": "Minimum number of sessions in a cluster (default 3)",
                    }
                },
                "required": [],
            },
            impl=lambda ctx, min_cluster_size=3: tool_repeated_prompts(ctx, min_cluster_size),
        ),
        ToolSpec(
            name="top_sessions",
            description=(
                "List the largest Cursor sessions by token estimate or turn count. Use when the "
                "user asks about specific big sessions or wants to drill into outliers."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many sessions to return (default 10)"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["tokens", "turns"],
                        "description": "Sort key (default 'tokens')",
                    },
                },
                "required": [],
            },
            impl=lambda ctx, n=10, sort_by="tokens": tool_top_sessions(ctx, n, sort_by),
        ),
        ToolSpec(
            name="search_sessions",
            description=(
                "Case-insensitive substring search over Cursor session opening prompts. Useful "
                "for finding 'all sessions where I asked about X'. Returns up to 20 matches."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Substring to search for"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["keyword"],
            },
            impl=lambda ctx, keyword, limit=20: tool_search_sessions(ctx, keyword, limit),
        ),
        ToolSpec(
            name="bucket_bloat",
            description=(
                "Per-bucket median and p95 tokens-per-session. Surfaces buckets that consistently "
                "eat tokens (the persistent context leak) vs. ones with a single huge outlier."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            impl=lambda ctx: tool_bucket_bloat(ctx),
        ),
        ToolSpec(
            name="github_prs",
            description=(
                "Fetches the user's recent GitHub PRs (authored by them) via the gh CLI. Use for "
                "'how much have I shipped' or 'what did I work on lately' questions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback window in days (default 90)"},
                    "limit": {"type": "integer", "description": "Max recent PRs to list (default 25)"},
                },
                "required": [],
            },
            impl=lambda ctx, days=90, limit=25: tool_github_prs(ctx, days, limit),
        ),
        ToolSpec(
            name="roi_by_week",
            description=(
                "Per-ISO-week correlation of Claude Code cost vs merged GitHub PRs. Use for "
                "'is my AI spend worth it' or 'show me cost vs shipping' questions. "
                "NOTE: presents per-week buckets, never a single ROI number - cost-per-PR is a "
                "starting point for calibration, not a leaderboard metric."
            ),
            input_schema={
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "Lookback window in days (default 90)"}},
                "required": [],
            },
            impl=lambda ctx, days=90: tool_roi(ctx, days),
        ),
        ToolSpec(
            name="cost_breakdown",
            description=(
                "Unified cost breakdown across every AI tool we can see locally: Cursor "
                "(volume only - no model in logs), Claude Code (priced), Codex CLI (priced). "
                "Includes cache utilization stats and 'cheaper model would have done it' "
                "recommendations. Use this for 'how much have I spent', 'where is my money "
                "going', 'should I switch models', 'how is my cache utilization'."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            impl=lambda ctx: tool_cost_breakdown(ctx),
        ),
    ]
