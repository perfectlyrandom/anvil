"""Tips engine: turn deep-analysis findings into prioritized, actionable recommendations.

The deep analyzer surfaces *facts* (you repeated prompt X 8 times, bucket Y has a p95 of 50K).
This module turns those into *advice* (save prompt X as a Cursor rule; switch bucket Y to scoped @
attachments) with rough token & dollar savings so the user can decide what's worth their time.

Estimates are intentionally conservative - we're trying to be a calibration tool, not a salesperson.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from anvil.analysis.claude_cost import ClaudeCodeCostReport
from anvil.analysis.cursor_deep import CursorDeepReport
from anvil.analysis.cursor_stats import CursorAggregate
from anvil.analysis.pricing import humanize_model, input_price_per_token
from anvil.analysis.skills import SkillsAuditReport

Severity = Literal["info", "warn", "high"]

# Default Cursor-pricing model. Cursor's JSONL doesn't record the model, so we let
# the caller pass one in. Opus is the more common high-pain default for daily drivers.
_DEFAULT_PRICING_MODEL = "claude-opus-4-5"


@dataclass
class Tip:
    """A single actionable recommendation."""

    id: str
    severity: Severity
    category: str  # "repetition" | "bloat" | "session_hygiene" | "cache" | "duplication"
    title: str  # one-line headline, with concrete numbers
    detail: str  # 1-2 sentence explanation of WHY this matters
    action: str  # what to do about it ("save as a Cursor rule", "use @file ...", etc.)
    estimated_savings_tokens: int | None = None
    estimated_savings_usd: float | None = None
    priced_against_model: str | None = None  # human label like "Opus 4.5", for transparency
    sample: dict[str, Any] = field(default_factory=dict)  # optional payload for richer rendering

    @property
    def severity_rank(self) -> int:
        return {"high": 0, "warn": 1, "info": 2}[self.severity]


def _usd(tokens: int, model: str) -> float:
    return round(tokens * input_price_per_token(model), 2)


def _tips_from_repeated_prompts(report: CursorDeepReport, pricing_model: str) -> list[Tip]:
    out: list[Tip] = []
    model_label = humanize_model(pricing_model)
    for i, cluster in enumerate(report.repeated_prompt_clusters[:5]):
        n = len(cluster.session_ids)
        burned = cluster.total_user_tokens_burned
        # Savings if you replaced all but the first occurrence with a tiny @rule reference (~50 tokens).
        savings_tokens = max(0, burned - (burned // n) - 50 * (n - 1))
        # Severity: anything ≥5 sessions OR ≥100K tokens is HIGH; anything ≥3 sessions is WARN.
        if n >= 5 or burned >= 100_000:
            sev: Severity = "high"
        elif n >= 3:
            sev = "warn"
        else:
            continue
        out.append(
            Tip(
                id=f"repeat-{i}",
                severity=sev,
                category="repetition",
                title=f"You opened {n} sessions with the same prompt - {burned:,} tokens",
                detail=(
                    f'"{cluster.canonical_first_query[:100]}…" - this exact opening shows up in '
                    f"{n} different Cursor sessions. The model isn't getting smarter each time."
                ),
                action=(
                    "Save it as a Cursor rule (`.cursor/rules/*.mdc` with `alwaysApply: true`) "
                    "or as a saved prompt template. Then your real first message is the unique part."
                ),
                estimated_savings_tokens=savings_tokens,
                estimated_savings_usd=_usd(savings_tokens, pricing_model),
                priced_against_model=model_label,
                sample={
                    "prompt_preview": cluster.canonical_first_query[:200],
                    "session_count": n,
                    "tokens_burned": burned,
                },
            )
        )
    return out


def _tips_from_bucket_bloat(report: CursorDeepReport, pricing_model: str) -> list[Tip]:
    out: list[Tip] = []
    model_label = humanize_model(pricing_model)
    # Bucket-specific advice. Order matters: most actionable first.
    advice = {
        "attached_files": (
            "Cursor is dumping whole files into context. Switch to scoped @symbol or @function "
            "references when you can, and use `.cursorignore` to keep noisy paths out."
        ),
        "agent_skill": (
            "Skill files are getting heavy. Audit which skills are auto-attached, trim verbose "
            "examples, and split rarely-used skills into description-only entries."
        ),
        "agent_transcripts": (
            "Sub-agent transcripts add up. Limit how deep you delegate, or use single-shot tools "
            "instead of full conversations when the answer is small."
        ),
        "git_status": (
            "Cursor is sending big git status snapshots. Commit or stash work-in-progress before "
            "long Cursor sessions, and consider `.cursorignore` for build outputs."
        ),
        "external_links": ("Linked-page content adds up fast. Paste the *relevant* part instead of the whole URL."),
        "mcp": (
            "MCP tools are returning verbose payloads. Look at which MCP responses are biggest and "
            "consider tighter queries or disabling chatty servers."
        ),
        "image_files": (
            "Image attachments are dense. Strip metadata, downscale, or only attach when the image "
            "actually matters to the question."
        ),
    }
    for bucket in report.bloat_buckets:
        # Severity rules: p95 ≥ 50K tokens AND appearing in ≥10 sessions = HIGH; ≥20K = WARN.
        if bucket.p95_tokens_per_session >= 50_000 and bucket.appears_in_sessions >= 10:
            sev: Severity = "high"
        elif bucket.p95_tokens_per_session >= 20_000 and bucket.appears_in_sessions >= 10:
            sev = "warn"
        else:
            continue
        # Conservative savings: if you cut p95 sessions back to the median, what would you save?
        per_session_savings = max(0, bucket.p95_tokens_per_session - bucket.median_tokens_per_session)
        # Apply across top-decile sessions (estimate: 10% of sessions).
        savings_tokens = per_session_savings * max(1, bucket.appears_in_sessions // 10)
        out.append(
            Tip(
                id=f"bloat-{bucket.name}",
                severity=sev,
                category="bloat",
                title=f"`{bucket.name}` bucket p95 is {bucket.p95_tokens_per_session:,} tokens / session",
                detail=(
                    f"Across {bucket.appears_in_sessions} sessions, the 95th-percentile session "
                    f"spends {bucket.p95_tokens_per_session:,} tokens just on `{bucket.name}` "
                    f"(median is {bucket.median_tokens_per_session:,})."
                ),
                action=advice.get(
                    bucket.name,
                    f"Investigate what's making `{bucket.name}` so heavy in your worst sessions.",
                ),
                estimated_savings_tokens=savings_tokens,
                estimated_savings_usd=_usd(savings_tokens, pricing_model),
                priced_against_model=model_label,
                sample={
                    "bucket": bucket.name,
                    "median": bucket.median_tokens_per_session,
                    "p95": bucket.p95_tokens_per_session,
                    "sessions": bucket.appears_in_sessions,
                },
            )
        )
    return out


def _tips_from_forked_sessions(report: CursorDeepReport, pricing_model: str) -> list[Tip]:
    """Surface sessions that look like duplicates/forks (same workspace, same leading turns)."""
    out: list[Tip] = []
    model_label = humanize_model(pricing_model)
    for i, cluster in enumerate(report.forked_session_clusters[:5]):
        n = len(cluster.session_ids)
        wasted = cluster.duplicated_token_cost
        # A depth ≥5 match is very high confidence; depth 3-4 is "looks like a fork".
        if cluster.identical_leading_turns >= 5 and wasted >= 10_000:
            sev: Severity = "high"
        elif wasted >= 5_000:
            sev = "warn"
        else:
            continue
        out.append(
            Tip(
                id=f"fork-{i}",
                severity=sev,
                category="duplication",
                title=(
                    f"{n} sessions in {cluster.workspace.split('/')[-1] or cluster.workspace} "
                    f"share their first {cluster.identical_leading_turns} turns - looks duplicated"
                ),
                detail=(
                    f"The first {cluster.identical_leading_turns} user+assistant turns are byte-identical "
                    f"across {n} different Cursor sessions starting with "
                    f'"{cluster.canonical_first_query[:80]}…". Each fork re-paid the model for the same '
                    f"leading exchange - about {wasted:,} tokens of pure rework."
                ),
                action=(
                    "If this was on purpose (parallel agents/worktrees) ignore it. Otherwise, "
                    "continue inside the original session via the agent transcript browser, or use "
                    "Cursor's branch-from-chat feature so the model keeps the cached prefix."
                ),
                estimated_savings_tokens=wasted,
                estimated_savings_usd=_usd(wasted, pricing_model),
                priced_against_model=model_label,
                sample={
                    "workspace": cluster.workspace,
                    "first_query": cluster.canonical_first_query[:200],
                    "session_count": n,
                    "depth": cluster.identical_leading_turns,
                },
            )
        )
    return out


def _tips_from_repeated_pastes(report: CursorDeepReport, pricing_model: str) -> list[Tip]:
    out: list[Tip] = []
    model_label = humanize_model(pricing_model)
    for i, sample in enumerate(report.mid_session_samples):
        if sample.duplicate_count < 3:
            continue
        # Per-reuse savings minus a tiny ref cost.
        per_reuse_savings = max(0, sample.estimated_tokens - 100)
        savings_tokens = per_reuse_savings * (sample.duplicate_count - 1)
        sev: Severity = "high" if sample.duplicate_count >= 5 else "warn"
        out.append(
            Tip(
                id=f"paste-{i}",
                severity=sev,
                category="repetition",
                title=(f"Same {sample.estimated_tokens:,}-token paste appeared in {sample.duplicate_count} sessions"),
                detail=(
                    f'This content ("{sample.text_preview[:80]}…") was pasted mid-session '
                    f"{sample.duplicate_count} separate times. You're paying for the same context repeatedly."
                ),
                action=(
                    "Save it as a project-scoped `.cursor/rules/*.mdc` with `alwaysApply: true`, or "
                    "stash it in a snippet and reference by `@file`."
                ),
                estimated_savings_tokens=savings_tokens,
                estimated_savings_usd=_usd(savings_tokens, pricing_model),
                priced_against_model=model_label,
                sample={
                    "preview": sample.text_preview[:200],
                    "tokens": sample.estimated_tokens,
                    "reuse_count": sample.duplicate_count,
                },
            )
        )
    return out[:5]


def _tips_from_session_hygiene(agg: CursorAggregate | None) -> list[Tip]:
    out: list[Tip] = []
    if agg is None:
        return out
    # Heuristic: median turns > 40 means sessions are running long. Marginal value of late turns is low.
    if agg.median_turns_per_session > 40:
        out.append(
            Tip(
                id="hygiene-long-sessions",
                severity="warn",
                category="session_hygiene",
                title=f"Median session is {agg.median_turns_per_session} turns long",
                detail=(
                    "Long sessions accumulate stale context, drift, and re-explanations. Most "
                    "research shows quality degrades past 20-30 turns for code tasks."
                ),
                action="Start fresh sessions for new sub-tasks. Use the agent to summarize state when you do switch.",
            )
        )
    elif agg.avg_turns_per_session > 25:
        out.append(
            Tip(
                id="hygiene-medium-sessions",
                severity="info",
                category="session_hygiene",
                title=f"Average session is {agg.avg_turns_per_session:.1f} turns long",
                detail="Sessions in the 20-30 turn range are usually fine, but watch for drift on the long tail.",
                action="No action needed yet. Glance at the heaviest-session list to spot outliers.",
            )
        )
    return out


def _tips_from_skills(report: SkillsAuditReport | None, pricing_model: str) -> list[Tip]:
    """Two skill-level tips: stale skills (no visible consultation) and high cumulative overhead."""
    if report is None or not report.skills:
        return []
    out: list[Tip] = []
    model_label = humanize_model(pricing_model)

    # 1) Stale skills - exposed many times, no assistant-side path mention. Caveat: silent
    # following and Cursor's context summarization can both hide real usage, so this is the
    # "no visible evidence" set, not "definitely unused".
    stale = [s for s in report.skills if s.is_stale]
    if stale:
        wasted = sum(s.description_tokens * s.exposed_in_sessions for s in stale)
        # Take the top-3 worst offenders by waste to name in the headline.
        stale_sorted = sorted(stale, key=lambda s: s.description_tokens * s.exposed_in_sessions, reverse=True)
        names = ", ".join(s.name for s in stale_sorted[:3])
        more = f" + {len(stale) - 3} more" if len(stale) > 3 else ""
        out.append(
            Tip(
                id="skills-stale",
                severity="high" if wasted >= 50_000 else "warn",
                category="skills",
                title=f"{len(stale)} skill{'s' if len(stale) != 1 else ''} look stale",
                detail=(
                    f"{names}{more} - exposed in dozens of sessions but no assistant response "
                    f"has mentioned their path. Could mean the description doesn't trigger, the "
                    f"skill is redundant, or the model is following it silently. ~{wasted:,} tokens "
                    f"of prelude so far either way."
                ),
                action=(
                    "Open each one and either (a) rewrite the description so it actually describes "
                    "the user-asks it should match, (b) merge it into a sibling skill that's already "
                    "working, or (c) delete it if you're confident it's unused."
                ),
                estimated_savings_tokens=wasted,
                estimated_savings_usd=_usd(wasted, pricing_model),
                priced_against_model=model_label,
                sample={
                    "stale_skill_names": [s.name for s in stale_sorted[:10]],
                    "stale_count": len(stale),
                },
            )
        )

    # 2) High per-session overhead - every prompt pays for the full list.
    overhead = report.per_session_description_overhead_tokens
    if overhead >= 5_000 and report.total_sessions_scanned >= 20:
        out.append(
            Tip(
                id="skills-overhead",
                severity="warn" if overhead < 10_000 else "high",
                category="skills",
                title=f"~{overhead:,} tokens of skill descriptions injected into every Cursor session",
                detail=(
                    f"You have {report.total_skills} skills installed. Cursor lists every one in the "
                    f"<available_skills> block of every prompt - even the stale ones. Over "
                    f"{report.total_sessions_scanned:,} sessions, that's "
                    f"~{report.total_lifetime_description_tokens:,} lifetime tokens of overhead."
                ),
                action=(
                    "Audit the Skills tab. Anything not consulted in the last 30+ sessions is a "
                    "candidate to delete or move to a project-scoped .cursor/skills/ folder so it "
                    "only loads where relevant."
                ),
                estimated_savings_tokens=report.total_lifetime_description_tokens // 4,
                estimated_savings_usd=_usd(report.total_lifetime_description_tokens // 4, pricing_model),
                priced_against_model=model_label,
                sample={
                    "overhead_per_session": overhead,
                    "skill_count": report.total_skills,
                },
            )
        )
    return out


def _tips_from_cache(cost_report: ClaudeCodeCostReport | None) -> list[Tip]:
    if cost_report is None or cost_report.total_input_tokens < 500_000:
        return []
    rate = cost_report.overall_cache_hit_rate or 0.0
    if rate < 0.30:
        # Estimate: if you got to 60% cache hit on the cold input portion, what could you save?
        # Sonnet cache reads are ~10% of normal input price, so going from 30% -> 60% cuts ~27% of input cost.
        savings_usd = (cost_report.estimated_total_cost_usd or 0) * 0.27
        return [
            Tip(
                id="cache-cold",
                severity="high",
                category="cache",
                title=f"Claude Code cache hit rate is only {rate * 100:.0f}%",
                detail=(
                    f"You've sent {cost_report.total_input_tokens:,} input tokens with very low cache reuse. "
                    "Each cold prompt costs full-price even when most of it is identical to a prior call."
                ),
                action=(
                    "Keep your system prompt + skill files stable across calls (don't reshuffle order). "
                    "Use Claude's prompt-caching: stable prefix, then the variable tail."
                ),
                estimated_savings_usd=round(savings_usd, 2),
            )
        ]
    return []


def build_tips(
    deep_report: CursorDeepReport | None,
    agg: CursorAggregate | None = None,
    cost_report: ClaudeCodeCostReport | None = None,
    skills_report: SkillsAuditReport | None = None,
    *,
    pricing_model: str = _DEFAULT_PRICING_MODEL,
) -> list[Tip]:
    """Combine all heuristics and return tips sorted by severity then by est. savings.

    ``pricing_model`` is the model family used to price Cursor-side savings. The Claude
    Code cache tip already knows its own model.
    """
    tips: list[Tip] = []
    if deep_report is not None:
        tips.extend(_tips_from_forked_sessions(deep_report, pricing_model))
        tips.extend(_tips_from_repeated_prompts(deep_report, pricing_model))
        tips.extend(_tips_from_bucket_bloat(deep_report, pricing_model))
        tips.extend(_tips_from_repeated_pastes(deep_report, pricing_model))
    tips.extend(_tips_from_session_hygiene(agg))
    tips.extend(_tips_from_cache(cost_report))
    tips.extend(_tips_from_skills(skills_report, pricing_model))
    tips.sort(
        key=lambda t: (t.severity_rank, -(t.estimated_savings_tokens or 0)),
    )
    return tips


def summary_savings(tips: list[Tip]) -> tuple[int, float]:
    """Sum total estimated savings across all tips (tokens, usd)."""
    total_tokens = sum(t.estimated_savings_tokens or 0 for t in tips)
    total_usd = sum(t.estimated_savings_usd or 0.0 for t in tips)
    return total_tokens, round(total_usd, 2)
