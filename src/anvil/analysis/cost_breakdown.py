"""Unified cost breakdown across every AI tool we can see locally.

Each tool exposes a different amount of usage data:

* **Cursor** — no model, no token counts, no API key, no cost. We can ESTIMATE prompt
  tokens via tiktoken but we can't price them honestly because we don't know the model.
  So Cursor shows up here only as a "tokens, but unpriced" line item.
* **Claude Code** — full per-turn usage with model. Anthropic pricing applies.
* **Codex CLI** — per-session cumulative usage with model and provider. OpenAI pricing.

This module rolls everything up into a ``CostBreakdownReport`` with rows that can be
filtered/segmented by source, provider, model, and project (cwd) on the dashboard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from anvil.analysis.pricing import (
    UNKNOWN_MODEL_FALLBACK,
    ModelPrice,
    humanize_model,
    price_for_model,
)
from anvil.parsers.claude_code import ClaudeCodeScanResult
from anvil.parsers.codex import CodexScanResult, CodexSession
from anvil.parsers.cursor_tracking import CursorTrackingResult
from anvil.parsers.cursor_transcripts import CursorScanResult


@dataclass
class CostRow:
    """One source x model x project cost line. The smallest unit the UI groups on."""

    source: str  # "cursor" | "claude_code" | "codex"
    provider: str  # "anthropic" | "openai" | "unknown"
    model: str  # raw model identifier ("claude-opus-4-5", "gpt-5.5", ...)
    model_label: str  # human-friendly display ("Opus 4.5")
    project: str  # cwd or workspace label, "(unknown)" if missing
    sessions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # subset of input
    cache_creation_tokens: int = 0  # Anthropic-only (write-to-cache cost)
    estimated_usd: float = 0.0
    is_estimated: bool = False  # True when we don't know the actual model and used a fallback
    # If a cheaper model exists in the same family, what would this row cost there?
    cheaper_alternative_model: str | None = None
    cheaper_alternative_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Cached share of input. None-equivalent (0) when no input recorded."""
        if self.input_tokens == 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens

    @property
    def potential_model_savings_usd(self) -> float:
        if self.cheaper_alternative_usd is None:
            return 0.0
        return max(0.0, self.estimated_usd - self.cheaper_alternative_usd)


@dataclass
class CacheStats:
    """Cross-source cache utilization rollup."""

    cacheable_input_tokens: int = 0  # input from sources that support caching
    cached_input_tokens: int = 0  # subset that hit cache
    estimated_cache_savings_usd: float = 0.0  # what we saved vs paying full input rate
    estimated_cache_left_on_table_usd: float = 0.0  # what we'd save if hit rate were 80%

    @property
    def hit_rate(self) -> float:
        if self.cacheable_input_tokens == 0:
            return 0.0
        return self.cached_input_tokens / self.cacheable_input_tokens


@dataclass
class CostBreakdownReport:
    """The aggregate the dashboard renders."""

    rows: list[CostRow]
    totals_by_source: dict[str, float] = field(default_factory=dict)
    totals_by_provider: dict[str, float] = field(default_factory=dict)
    totals_by_model: dict[str, float] = field(default_factory=dict)
    grand_total_usd: float = 0.0
    grand_total_input_tokens: int = 0
    grand_total_output_tokens: int = 0
    cursor_sessions_unpriced: int = 0  # legacy name; now = sessions estimated against fallback model
    cursor_sessions_priced: int = 0  # Cursor sessions priced against a known model from the DB
    cache: CacheStats = field(default_factory=CacheStats)
    # Top model-switch recommendations: rows sorted by potential savings descending.
    switch_recommendations: list[CostRow] = field(default_factory=list)


# ---- helpers ----------------------------------------------------------------------------


def _project_label(cwd: str | Path | None) -> str:
    """Squash absolute paths to the last 2 segments so the UI stays readable."""
    if cwd is None:
        return "(unknown)"
    p = Path(str(cwd))
    parts = p.parts
    if len(parts) <= 2:
        return str(p)
    return "/".join(parts[-2:])


def _cheaper_in_family(model: str) -> str | None:
    """Return a strictly-cheaper-but-still-capable peer model, or None.

    The list is judgmental, not algorithmic — we only suggest swaps where the cheaper
    model is plausibly "good enough" for routine engineering work. A model recommends
    against Opus → Haiku because that's a real quality drop; Opus → Sonnet is fine.
    """
    swap_map = {
        # Claude
        "claude-opus-4-5": "claude-sonnet-4-5",
        "claude-opus-4": "claude-sonnet-4",
        "claude-3-opus": "claude-3-5-sonnet",
        # OpenAI
        "gpt-5.5": "gpt-5-mini",
        "gpt-5": "gpt-5-mini",
        "gpt-4o": "gpt-4o-mini",
        "gpt-4.1": "gpt-4.1-mini",
    }
    for prefix, target in swap_map.items():
        if model.startswith(prefix):
            return target
    return None


def _cost_anthropic(
    input_tokens: int, output_tokens: int, cache_creation: int, cache_read: int, price: ModelPrice
) -> float:
    """Anthropic billing: input + output + cache_write + cache_read all priced separately."""
    return (
        input_tokens * price.input_per_mtok / 1_000_000
        + output_tokens * price.output_per_mtok / 1_000_000
        + cache_creation * price.cache_write_per_mtok / 1_000_000
        + cache_read * price.cache_read_per_mtok / 1_000_000
    )


def _cost_openai(input_tokens: int, output_tokens: int, cached_input: int, price: ModelPrice) -> float:
    """OpenAI billing: cached_input is a discount on input, output is separate."""
    uncached_input = max(0, input_tokens - cached_input)
    return (
        uncached_input * price.input_per_mtok / 1_000_000
        + cached_input * price.cache_read_per_mtok / 1_000_000
        + output_tokens * price.output_per_mtok / 1_000_000
    )


def _row_from_codex(session: CodexSession) -> CostRow:
    """Build one CostRow per Codex session. Cheaper-model price is computed inline."""
    model = session.model or "(unknown)"
    price = price_for_model(model) or UNKNOWN_MODEL_FALLBACK
    cost = _cost_openai(
        session.total_usage.input_tokens,
        session.total_usage.output_tokens,
        session.total_usage.cached_input_tokens,
        price,
    )
    cheaper = _cheaper_in_family(model)
    cheaper_cost = None
    if cheaper:
        cheaper_price = price_for_model(cheaper)
        if cheaper_price is not None:
            cheaper_cost = _cost_openai(
                session.total_usage.input_tokens,
                session.total_usage.output_tokens,
                session.total_usage.cached_input_tokens,
                cheaper_price,
            )
    return CostRow(
        source="codex",
        provider=session.model_provider or "openai",
        model=model,
        model_label=humanize_model(model),
        project=_project_label(session.cwd),
        sessions=1,
        input_tokens=session.total_usage.input_tokens,
        output_tokens=session.total_usage.output_tokens,
        cached_input_tokens=session.total_usage.cached_input_tokens,
        estimated_usd=cost,
        cheaper_alternative_model=cheaper,
        cheaper_alternative_usd=cheaper_cost,
    )


def _rows_from_claude_code(scan: ClaudeCodeScanResult) -> list[CostRow]:
    """Aggregate Claude Code sessions into (model, project) rows. One row per (model, project).

    Claude Code can split a single session across multiple models, so we explode each
    session into its ``usage_by_model`` entries first, then re-group by (model, project).
    """
    grouped: dict[tuple[str, str], CostRow] = {}
    for session in scan.sessions:
        project = _project_label(session.cwd)
        for model, usage in session.usage_by_model.items():
            if usage.input_tokens == 0 and usage.output_tokens == 0:
                continue
            key = (model, project)
            row = grouped.get(key)
            if row is None:
                row = CostRow(
                    source="claude_code",
                    provider="anthropic",
                    model=model,
                    model_label=humanize_model(model),
                    project=project,
                    cheaper_alternative_model=_cheaper_in_family(model),
                )
                grouped[key] = row
            row.sessions += 1
            row.input_tokens += usage.input_tokens
            row.output_tokens += usage.output_tokens
            row.cached_input_tokens += usage.cache_read_input_tokens
            row.cache_creation_tokens += usage.cache_creation_input_tokens
    # Cost pass — done once per row at the end so we don't repeatedly look up pricing.
    for row in grouped.values():
        price = price_for_model(row.model)
        if price is None:
            continue
        row.estimated_usd = _cost_anthropic(
            row.input_tokens,
            row.output_tokens,
            row.cache_creation_tokens,
            row.cached_input_tokens,
            price,
        )
        cheaper = row.cheaper_alternative_model
        if cheaper:
            cheaper_price = price_for_model(cheaper)
            if cheaper_price is not None:
                row.cheaper_alternative_usd = _cost_anthropic(
                    row.input_tokens,
                    row.output_tokens,
                    row.cache_creation_tokens,
                    row.cached_input_tokens,
                    cheaper_price,
                )
    return list(grouped.values())


def _rows_from_cursor(
    scan: CursorScanResult,
    tracking: CursorTrackingResult,
    fallback_model: str,
) -> list[CostRow]:
    """Price Cursor sessions using the AI-tracking DB where available, fallback otherwise.

    The tracking DB only stores AI-generated code chunks, so it covers maybe 5% of
    sessions (the ones that wrote code into the buffer). For the remaining 95% we use
    ``fallback_model`` as a calibrated guess - the user's ``default_pricing_model``,
    which can be overridden by the observed distribution from the DB if that exists.

    We bucket by (model, project) so a workspace that flipped between Opus and Sonnet
    shows up as two rows. Sessions priced from real DB attribution are NOT marked
    estimated; fallback sessions ARE.
    """
    if not scan.sessions:
        return []
    # If we observed a clear modal model from the DB, use it as the fallback instead of
    # the static default - more honest than always assuming Opus for everyone.
    if tracking.model_distribution:
        observed_modal = tracking.model_distribution.most_common(1)[0][0]
        fallback_model = observed_modal

    grouped: dict[tuple[str, str, bool], CostRow] = {}
    for session in scan.sessions:
        project = _project_label(session.workspace)
        model = tracking.model_for(session.session_id)
        is_estimated = model is None
        if model is None:
            model = fallback_model
        key = (model, project, is_estimated)
        row = grouped.get(key)
        if row is None:
            row = CostRow(
                source="cursor",
                provider="anthropic" if model.startswith("claude-") else "openai",
                model=model,
                model_label=humanize_model(model) + (" (est.)" if is_estimated else ""),
                project=project,
                is_estimated=is_estimated,
                cheaper_alternative_model=_cheaper_in_family(model),
            )
            grouped[key] = row
        row.sessions += 1
        row.input_tokens += session.user_tokens_est
        row.output_tokens += session.assistant_tokens_est

    # Cost pass — Cursor's transcripts have no cache attribution, so we treat all input
    # as uncached (worst-case pricing). Use the Anthropic formula for Claude rows and
    # the OpenAI formula for GPT rows.
    for row in grouped.values():
        price = price_for_model(row.model)
        if price is None:
            continue
        if row.model.startswith("claude-"):
            row.estimated_usd = _cost_anthropic(row.input_tokens, row.output_tokens, 0, 0, price)
        else:
            row.estimated_usd = _cost_openai(row.input_tokens, row.output_tokens, 0, price)
        cheaper = row.cheaper_alternative_model
        if cheaper:
            cheaper_price = price_for_model(cheaper)
            if cheaper_price is not None:
                if row.model.startswith("claude-"):
                    row.cheaper_alternative_usd = _cost_anthropic(
                        row.input_tokens, row.output_tokens, 0, 0, cheaper_price
                    )
                else:
                    row.cheaper_alternative_usd = _cost_openai(row.input_tokens, row.output_tokens, 0, cheaper_price)
    return list(grouped.values())


def _compute_cache_stats(rows: list[CostRow]) -> CacheStats:
    """Roll cache hit rate + savings up across all sources that report cache data."""
    stats = CacheStats()
    for row in rows:
        # Cursor doesn't report cache, so it doesn't contribute to the rate.
        if row.source == "cursor":
            continue
        stats.cacheable_input_tokens += row.input_tokens
        stats.cached_input_tokens += row.cached_input_tokens
        price = price_for_model(row.model)
        if price is None:
            continue
        # "saved" = cached tokens times (full-input-rate - cache-read-rate)
        delta = price.input_per_mtok - price.cache_read_per_mtok
        stats.estimated_cache_savings_usd += row.cached_input_tokens * delta / 1_000_000
        # "left on table" = uncached input times delta, scaled so hitting 80% hit rate would save.
        uncached = max(0, row.input_tokens - row.cached_input_tokens)
        target_hit_rate = 0.80
        already_hit = row.cached_input_tokens / row.input_tokens if row.input_tokens else 0
        room_to_grow = max(0.0, target_hit_rate - already_hit)
        recoverable_tokens = uncached * (room_to_grow / (1 - already_hit)) if already_hit < 1 else 0
        stats.estimated_cache_left_on_table_usd += recoverable_tokens * delta / 1_000_000
    return stats


def build_cost_breakdown(
    cursor_scan: CursorScanResult | None,
    claude_scan: ClaudeCodeScanResult | None,
    codex_scan: CodexScanResult | None,
    cursor_tracking: CursorTrackingResult | None = None,
    fallback_model: str = "claude-opus-4-5",
) -> CostBreakdownReport:
    """Combine every source's scan into one segmentable report.

    ``cursor_tracking`` supplies model attribution for Cursor conversations we can
    identify in the AI-tracking DB. Sessions not in the DB are priced against
    ``fallback_model`` and marked ``is_estimated=True``.
    """
    rows: list[CostRow] = []
    cursor_estimated_sessions = 0
    cursor_priced_sessions = 0
    if cursor_scan is not None:
        cursor_rows = _rows_from_cursor(
            cursor_scan,
            cursor_tracking or CursorTrackingResult(),
            fallback_model=fallback_model,
        )
        for r in cursor_rows:
            if r.is_estimated:
                cursor_estimated_sessions += r.sessions
            else:
                cursor_priced_sessions += r.sessions
        rows.extend(cursor_rows)
    if claude_scan is not None:
        rows.extend(_rows_from_claude_code(claude_scan))
    if codex_scan is not None:
        rows.extend(_row_from_codex(s) for s in codex_scan.sessions if s.total_usage.input_tokens > 0)
    rows.sort(key=lambda r: r.estimated_usd, reverse=True)

    # Aggregate rollups.
    totals_by_source: dict[str, float] = defaultdict(float)
    totals_by_provider: dict[str, float] = defaultdict(float)
    totals_by_model: dict[str, float] = defaultdict(float)
    grand = 0.0
    grand_in = 0
    grand_out = 0
    for row in rows:
        totals_by_source[row.source] += row.estimated_usd
        totals_by_provider[row.provider] += row.estimated_usd
        totals_by_model[row.model_label] += row.estimated_usd
        grand += row.estimated_usd
        grand_in += row.input_tokens
        grand_out += row.output_tokens

    cache = _compute_cache_stats(rows)
    switch_recs = sorted(
        (r for r in rows if r.potential_model_savings_usd > 0.01),
        key=lambda r: r.potential_model_savings_usd,
        reverse=True,
    )[:10]

    return CostBreakdownReport(
        rows=rows,
        totals_by_source=dict(totals_by_source),
        totals_by_provider=dict(totals_by_provider),
        totals_by_model=dict(totals_by_model),
        grand_total_usd=grand,
        grand_total_input_tokens=grand_in,
        grand_total_output_tokens=grand_out,
        cursor_sessions_unpriced=cursor_estimated_sessions,
        cursor_sessions_priced=cursor_priced_sessions,
        cache=cache,
        switch_recommendations=switch_recs,
    )
