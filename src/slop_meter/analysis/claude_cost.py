"""Cost analysis over Claude Code sessions.

Uses Anthropic pricing (``analysis/pricing.py``) to convert real token counts into
USD estimates, broken down by model, project (cwd), and cache hit rate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from slop_meter.analysis.pricing import (
    UNKNOWN_MODEL_FALLBACK,
    cache_hit_rate,
    estimate_cost,
    price_for_model,
)
from slop_meter.parsers.claude_code import ClaudeCodeScanResult, ClaudeCodeSession, TurnUsage


@dataclass
class ModelCost:
    model: str
    sessions_using: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost_usd: float
    cache_hit_rate: float | None
    is_priced: bool


@dataclass
class ProjectCost:
    cwd: str
    sessions: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class ClaudeCodeCostReport:
    total_sessions: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    estimated_total_cost_usd: float
    overall_cache_hit_rate: float | None
    by_model: list[ModelCost]
    by_project: list[ProjectCost]
    most_expensive_sessions: list[tuple[ClaudeCodeSession, float]]
    unpriced_models: list[str]


def _agg_usage_by_model(scan: ClaudeCodeScanResult) -> dict[str, TurnUsage]:
    out: dict[str, TurnUsage] = defaultdict(TurnUsage)
    sessions_using: dict[str, set[str]] = defaultdict(set)
    for session in scan.sessions:
        for model, usage in session.usage_by_model.items():
            existing = out[model]
            out[model] = TurnUsage(
                input_tokens=existing.input_tokens + usage.input_tokens,
                output_tokens=existing.output_tokens + usage.output_tokens,
                cache_creation_input_tokens=existing.cache_creation_input_tokens + usage.cache_creation_input_tokens,
                cache_read_input_tokens=existing.cache_read_input_tokens + usage.cache_read_input_tokens,
                web_search_requests=existing.web_search_requests + usage.web_search_requests,
                web_fetch_requests=existing.web_fetch_requests + usage.web_fetch_requests,
            )
            sessions_using[model].add(session.session_id)
    # Stash session-count via a hidden side channel since TurnUsage doesn't carry it.
    # Returning a separate mapping is cleaner.
    return out


def _sessions_per_model(scan: ClaudeCodeScanResult) -> dict[str, int]:
    counts: dict[str, set[str]] = defaultdict(set)
    for session in scan.sessions:
        for model in session.usage_by_model:
            counts[model].add(session.session_id)
    return {m: len(ids) for m, ids in counts.items()}


def build_cost_report(scan: ClaudeCodeScanResult, *, top_sessions: int = 10) -> ClaudeCodeCostReport:
    usage_by_model = _agg_usage_by_model(scan)
    sessions_per_model = _sessions_per_model(scan)

    by_model: list[ModelCost] = []
    unpriced: list[str] = []
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_cost = 0.0

    for model, usage in sorted(usage_by_model.items(), key=lambda item: item[0]):
        price = price_for_model(model)
        is_priced = price is not None
        if not is_priced:
            unpriced.append(model)
            cost = estimate_cost(usage, model="claude-sonnet-4-5")  # fallback estimate
        else:
            cost = estimate_cost(usage, model)
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        total_cache_read += usage.cache_read_input_tokens
        total_cache_write += usage.cache_creation_input_tokens
        total_cost += cost if is_priced else 0.0
        by_model.append(
            ModelCost(
                model=model,
                sessions_using=sessions_per_model.get(model, 0),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens,
                cache_write_tokens=usage.cache_creation_input_tokens,
                estimated_cost_usd=cost,
                cache_hit_rate=cache_hit_rate(usage),
                is_priced=is_priced,
            )
        )
    by_model.sort(key=lambda m: m.estimated_cost_usd, reverse=True)

    # Project (cwd) rollup
    project_tokens: dict[str, int] = defaultdict(int)
    project_sessions: dict[str, int] = defaultdict(int)
    project_cost: dict[str, float] = defaultdict(float)
    session_costs: list[tuple[ClaudeCodeSession, float]] = []

    for session in scan.sessions:
        cost = 0.0
        tokens = 0
        for model, usage in session.usage_by_model.items():
            cost += estimate_cost(usage, model)
            tokens += usage.input_tokens + usage.output_tokens
        project_tokens[session.cwd] += tokens
        project_sessions[session.cwd] += 1
        project_cost[session.cwd] += cost
        session_costs.append((session, cost))

    by_project = [
        ProjectCost(
            cwd=cwd,
            sessions=project_sessions[cwd],
            total_tokens=project_tokens[cwd],
            estimated_cost_usd=project_cost[cwd],
        )
        for cwd in project_cost
    ]
    by_project.sort(key=lambda p: p.estimated_cost_usd, reverse=True)
    session_costs.sort(key=lambda pair: pair[1], reverse=True)

    overall_hit = None
    if total_cache_read + total_input > 0:
        overall_hit = total_cache_read / (total_cache_read + total_input)
    fallback_price = price_for_model("claude-sonnet-4-5") or UNKNOWN_MODEL_FALLBACK
    del fallback_price  # silence unused-warning; reserved for future fallback display

    return ClaudeCodeCostReport(
        total_sessions=len(scan.sessions),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cache_read_tokens=total_cache_read,
        total_cache_write_tokens=total_cache_write,
        estimated_total_cost_usd=total_cost,
        overall_cache_hit_rate=overall_hit,
        by_model=by_model,
        by_project=by_project,
        most_expensive_sessions=session_costs[:top_sessions],
        unpriced_models=unpriced,
    )
