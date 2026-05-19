"""Harsh expert critic that audits your AI setup and tells you what's broken.

This is deliberately opinionated. The dashboard already shows you numbers; the coach
turns those numbers into a roast plus a concrete fix. Each issue has:

* ``severity`` — ``critical`` / ``high`` / ``medium`` / ``low``. Ordering matters; the UI
  shows them in this order and treats critical/high differently.
* ``title`` — one line, no equivocation. "You're burning Opus dollars on Sonnet work."
* ``body`` — 1-3 sentences with the actual numbers driving the verdict.
* ``action`` — what to do next, in imperative voice.
* ``evidence`` — optional dict the UI can render as a small table (key→value pairs).

We're explicit about NOT calling an LLM here — these checks need to be fast, deterministic,
and run on every dashboard refresh. The LLM agent is available separately if the user
wants to chat about a specific issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from anvil.analysis.cost_breakdown import CostBreakdownReport
from anvil.analysis.cursor_deep import CursorDeepReport
from anvil.analysis.cursor_stats import CursorAggregate
from anvil.analysis.pricing import humanize_model
from anvil.analysis.shipped import ShippedReport, SignificanceTier
from anvil.analysis.skills import SkillsAuditReport

Severity = Literal["critical", "high", "medium", "low"]


@dataclass
class CoachIssue:
    """One verdict from the coach. Order in the UI follows ``severity_rank``."""

    id: str  # stable for deduping / a/b
    severity: Severity
    title: str
    body: str
    action: str
    evidence: dict[str, str] = field(default_factory=dict)
    estimated_savings_usd: float | None = None

    @property
    def severity_rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3}[self.severity]


@dataclass
class CoachReport:
    issues: list[CoachIssue]
    grade: str  # 'A' / 'B' / 'C' / 'D' / 'F' — visual at-a-glance
    one_line_summary: str  # the headline above the grade

    @property
    def total_estimated_savings_usd(self) -> float:
        return sum(i.estimated_savings_usd or 0.0 for i in self.issues)


# ---- individual checks. each returns ``CoachIssue`` or None. ------------------------------


def _check_overpaying_for_model(cost: CostBreakdownReport) -> CoachIssue | None:
    """Are you on the most-expensive tier when a cheaper one would do?"""
    if not cost.switch_recommendations:
        return None
    top = cost.switch_recommendations[0]
    save = top.potential_model_savings_usd
    if save < 5.0:
        return None
    severity: Severity = "critical" if save > 50 else ("high" if save > 20 else "medium")
    # Strip "(est.)" leak from the label; coach voice shouldn't quote our internal markers.
    expensive_label = top.model_label.replace(" (est.)", "").strip()
    cheaper_label = (
        humanize_model(top.cheaper_alternative_model) if top.cheaper_alternative_model else "a cheaper model"
    )
    return CoachIssue(
        id="overpay_model",
        severity=severity,
        title=f"You're paying {expensive_label} prices for {cheaper_label}-level work.",
        body=(
            f"Your biggest spend row is {expensive_label} on {top.project} at ${top.estimated_usd:.2f}. "
            f"The same input/output mix on {cheaper_label} would have cost "
            f"${top.cheaper_alternative_usd or 0:.2f} — a {(save / top.estimated_usd * 100):.0f}% cut."
        ),
        action=(
            f"For routine engineering work, default to {cheaper_label}. "
            f"Reserve {expensive_label} for the hard reasoning steps you can actually point to."
        ),
        evidence={
            "current model": expensive_label,
            "suggested model": cheaper_label,
            "paid": f"${top.estimated_usd:.2f}",
            "would have paid": f"${top.cheaper_alternative_usd or 0:.2f}",
        },
        estimated_savings_usd=save,
    )


def _check_cache_neglect(cost: CostBreakdownReport) -> CoachIssue | None:
    """Cache hit rate under 30% on real volume is leaving money on the table."""
    if cost.cache.cacheable_input_tokens < 100_000:
        return None
    if cost.cache.hit_rate >= 0.30:
        return None
    severity: Severity = "high" if cost.cache.estimated_cache_left_on_table_usd > 5 else "medium"
    return CoachIssue(
        id="cache_neglect",
        severity=severity,
        title=f"Cache hit rate is {cost.cache.hit_rate * 100:.0f}% — that's a leak.",
        body=(
            f"You moved {cost.cache.cacheable_input_tokens:,} tokens of cacheable input but only "
            f"{cost.cache.cached_input_tokens:,} actually hit cache. The rest paid full input rate."
        ),
        action=(
            "Stop starting fresh chats for tweaks to the same code. Continue in the existing "
            "session so cache reads stay warm. For Claude Code, use the 1h ephemeral cache for "
            "long-running planning sessions."
        ),
        evidence={
            "hit rate": f"{cost.cache.hit_rate * 100:.1f}%",
            "saved by cache": f"${cost.cache.estimated_cache_savings_usd:.2f}",
            "left on table @ 80%": f"${cost.cache.estimated_cache_left_on_table_usd:.2f}",
        },
        estimated_savings_usd=cost.cache.estimated_cache_left_on_table_usd,
    )


def _check_forked_sessions(deep: CursorDeepReport | None) -> CoachIssue | None:
    """Duplicating a session and continuing in the dupe is a wallet-burner."""
    if deep is None or not deep.forked_session_clusters:
        return None
    clusters = deep.forked_session_clusters
    biggest = max(clusters, key=lambda c: c.duplicated_token_cost)
    severity: Severity = "high" if len(clusters) > 3 else "medium"
    return CoachIssue(
        id="forked_sessions",
        severity=severity,
        title=f"{len(clusters)} clusters of duplicated sessions detected.",
        body=(
            f"The biggest cluster has {len(biggest.session_ids)} byte-identical openings in "
            f"workspace '{biggest.workspace}' burning {biggest.duplicated_token_cost:,} tokens "
            "in the duplicated prefix alone. You're paying for the same context to be re-built every time."
        ),
        action=(
            "Continue in the original session instead of duplicating. If you need a branch point, "
            "use Cursor's 'Branch from chat' so the cache state is reused."
        ),
        evidence={
            "clusters": str(len(clusters)),
            "biggest cluster size": str(len(biggest.session_ids)),
            "biggest workspace": biggest.workspace,
        },
    )


def _check_stale_skills(skills: SkillsAuditReport | None) -> CoachIssue | None:
    """Skills that get loaded but show no evidence of consultation are likely overhead.

    "Stale" not "dead" - we may miss silent following or summarized turns. See the Skills
    tab methodology note for the honest caveats.
    """
    if skills is None or skills.stale_skills_count < 3:
        return None
    severity: Severity = "high" if skills.per_session_description_overhead_tokens > 1500 else "medium"
    return CoachIssue(
        id="stale_skills",
        severity=severity,
        title=f"{skills.stale_skills_count} skills look stale - exposed but never visibly consulted.",
        body=(
            f"They get loaded into the description block ({skills.per_session_description_overhead_tokens:,} "
            f"tokens per session) and no session in Cursor, Codex, or Claude Code has mentioned "
            f"their <name>/SKILL.md path. Across {skills.total_sessions_scanned} Cursor sessions "
            f"that's {skills.total_lifetime_description_tokens:,} tokens of prelude with no visible payoff. "
            f"Caveat: in Cursor the model can follow a skill silently without quoting its path, so this "
            f"is the no-visible-evidence set, not definitively unused."
        ),
        action=(
            "Audit the Skills tab. For each stale entry, either rewrite the description so it "
            "triggers when you actually need the skill, move it out of always-applied scope, "
            "or delete it."
        ),
        evidence={
            "stale": str(skills.stale_skills_count),
            "overhead per session": f"{skills.per_session_description_overhead_tokens:,} tokens",
            "lifetime overhead": f"{skills.total_lifetime_description_tokens:,} tokens",
        },
    )


def _check_attached_files_bloat(agg: CursorAggregate) -> CoachIssue | None:
    """A bucket eating > 30% of total tokens is a context shape problem."""
    if not agg.buckets or agg.total_user_tokens == 0:
        return None
    # Find the heaviest non-user_query bucket - the user's own queries are supposed to be there.
    contextual_buckets = [b for b in agg.buckets if b.name != "user_query"]
    if not contextual_buckets:
        return None
    heaviest = max(contextual_buckets, key=lambda b: b.share_of_user_tokens)
    if heaviest.share_of_user_tokens < 0.30:
        return None
    severity: Severity = "high" if heaviest.share_of_user_tokens > 0.50 else "medium"
    return CoachIssue(
        id=f"bucket_bloat_{heaviest.name}",
        severity=severity,
        title=f"'{heaviest.name}' is {heaviest.share_of_user_tokens * 100:.0f}% of your prompt budget.",
        body=(
            f"Across {heaviest.appears_in_sessions} sessions, this bucket ate "
            f"{heaviest.total_tokens:,} tokens. The model never asked for that much; you handed it over."
        ),
        action={
            "attached_files": "Attach by section, not whole file. Use @file:lines syntax. Trust the model to read in if it needs more.",
            "workspace_rule": "Audit your .cursor/rules — most always-apply rules are read once and ignored. Use agent-requested rules.",
            "external_links": "Pasting URLs costs tokens for the fetch. Quote the 2-3 lines you care about instead.",
            "available_skills": "See Skills tab — kill any skill that's exposed but never consulted.",
        }.get(heaviest.name, "Reduce what you hand over by default; let the model pull on demand."),
        evidence={
            "bucket": heaviest.name,
            "share of input": f"{heaviest.share_of_user_tokens * 100:.1f}%",
            "tokens": f"{heaviest.total_tokens:,}",
            "sessions": str(heaviest.appears_in_sessions),
        },
    )


def _check_roi_mismatch(cost: CostBreakdownReport, shipped: ShippedReport | None) -> CoachIssue | None:
    """If you've burned real money and shipped little, that's the conversation."""
    if shipped is None:
        return None
    if cost.grand_total_usd < 20:
        return None
    moved_needle = shipped.by_tier.get(SignificanceTier.moved_needle.value, 0)
    real_work = shipped.by_tier.get(SignificanceTier.real_work.value, 0)
    impactful = moved_needle + real_work
    if impactful == 0:
        cost_per = float("inf")
    else:
        cost_per = cost.grand_total_usd / impactful
    # Threshold: > $40 per impactful PR is suspicious, > $100 is real.
    if cost_per < 40:
        return None
    severity: Severity = "high" if cost_per > 100 or impactful == 0 else "medium"
    return CoachIssue(
        id="roi_mismatch",
        severity=severity,
        title=(f"${cost.grand_total_usd:.0f} burned · {impactful} meaningful PRs in " f"{shipped.window_days}d."),
        body=(
            f"That's ${cost_per:.0f} per impactful PR (or no PRs at all). Either the AI isn't "
            "earning its keep on this work or your PR titles are too modest to score above 'routine'."
            if impactful > 0
            else "You spent real money and shipped nothing scored above 'routine'. Either the AI "
            "isn't earning its keep, or your PR titles + descriptions undersell the work."
        ),
        action=(
            "Write better PR descriptions — the significance score reads them for customer/perf/"
            "architecture signals. If the work genuinely is mostly routine, look at the "
            "cheaper-model wins on the Cost tab."
        ),
        evidence={
            "spend": f"${cost.grand_total_usd:.2f}",
            "meaningful PRs": str(impactful),
            "$ per impactful PR": f"${cost_per:.0f}" if impactful else "n/a",
        },
    )


def _check_missing_signals(
    cost: CostBreakdownReport,
    shipped: ShippedReport | None,
) -> CoachIssue | None:
    """If half the inputs are blank, the verdict is unreliable."""
    blanks: list[str] = []
    if cost.cursor_sessions_unpriced > cost.cursor_sessions_priced * 5:
        blanks.append("Cursor model attribution (most sessions estimated)")
    if shipped is None:
        blanks.append("GitHub PR data (gh CLI not authed?)")
    if not cost.totals_by_source.get("claude_code") and not cost.totals_by_source.get("codex"):
        blanks.append("priced source data (no Claude Code or Codex usage)")
    if not blanks:
        return None
    return CoachIssue(
        id="missing_signals",
        severity="low",
        title="Some inputs are missing — verdict is partial.",
        body=(
            "The coach can only roast you on data it can see. The following streams aren't "
            "fully populated, so dollar estimates and ROI checks are best-effort."
        ),
        action="Run `gh auth login` if needed; consider switching some workflows to Claude Code or Codex CLI for real token-level telemetry.",
        evidence={"blanks": ", ".join(blanks)},
    )


# ---- assembly ------------------------------------------------------------------------------


def _assign_grade(issues: list[CoachIssue]) -> tuple[str, str]:
    """Translate the worst severity present into a single visual grade + headline."""
    crit = sum(1 for i in issues if i.severity == "critical")
    high = sum(1 for i in issues if i.severity == "high")
    med = sum(1 for i in issues if i.severity == "medium")
    if crit > 0:
        return "F", f"{crit} critical {'issue' if crit == 1 else 'issues'} — fix these first."
    if high >= 3:
        return "D", "Multiple high-severity leaks. You're not poor, you're paying the lazy tax."
    if high >= 1:
        return "C", "One or more high-severity leaks. Address them and you'll feel it on the bill."
    if med >= 2:
        return "B", "Mostly fine. A couple of medium things to clean up."
    if med == 1:
        return "B", "Solid setup. One thing to look at."
    return "A", "Honestly? You're running clean. Keep it that way."


def run_coach(
    cost: CostBreakdownReport,
    *,
    deep: CursorDeepReport | None = None,
    skills: SkillsAuditReport | None = None,
    cursor_stats: CursorAggregate | None = None,
    shipped: ShippedReport | None = None,
) -> CoachReport:
    """Run every check and assemble the verdict."""
    raw: list[CoachIssue | None] = [
        _check_overpaying_for_model(cost),
        _check_cache_neglect(cost),
        _check_forked_sessions(deep),
        _check_stale_skills(skills),
        _check_attached_files_bloat(cursor_stats) if cursor_stats else None,
        _check_roi_mismatch(cost, shipped),
        _check_missing_signals(cost, shipped),
    ]
    issues = [i for i in raw if i is not None]
    issues.sort(key=lambda i: (i.severity_rank, -(i.estimated_savings_usd or 0.0)))
    grade, summary = _assign_grade(issues)
    return CoachReport(issues=issues, grade=grade, one_line_summary=summary)
