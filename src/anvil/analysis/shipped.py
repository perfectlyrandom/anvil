"""Classify GitHub PRs by significance — a tiered score, not a binary.

Raw PR counts are misleading — version bumps, lockfile updates, automated changelog
PRs, and reverts all inflate the number without representing real product work. A
≥50-LoC bar is also too coarse: a 60-LoC rename is not equivalent to a 60-LoC
performance fix that moves customer experience.

So we score each PR on a 0-100 scale and bucket into four tiers:

* **moved-the-needle** (≥70) — perf/feat/architecture/customer signal, sizeable, reviewed.
* **real work** (40-69) — solid fix/feat/refactor with at least some signal.
* **routine** (15-39) — small, narrow, but legitimate engineering work.
* **noise** (<15 or noise-pattern match) — bumps, reverts, lint, typo, bot, etc.

Inputs to the score:

* Conventional-commits type (``feat`` / ``perf`` / ``fix`` / ``refactor`` / …).
* PR body keyword signals — "customer", "performance"/"latency", "architecture"/
  "migration", "regression"/"incident", "tech debt".
* Negative title signals — "rename", "typo", "comment-only", "lint".
* Size (logarithmic so 10K-LoC PRs don't dominate).
* Files touched (cross-cutting > single-file work).
* Review count as a soft significance signal (un-reviewed merges weigh less).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from anvil.connectors.github import PullRequestRecord


class PrCategory(str, Enum):
    feature = "feature"
    fix = "fix"
    refactor = "refactor"
    perf = "perf"
    docs = "docs"
    test = "test"
    chore = "chore"
    style = "style"
    ci = "ci"
    build = "build"
    revert = "revert"
    other = "other"


# Conventional-commits prefix → category. "feat" maps to "feature" for readability.
_CONVENTIONAL_TYPE_MAP: dict[str, PrCategory] = {
    "feat": PrCategory.feature,
    "feature": PrCategory.feature,
    "fix": PrCategory.fix,
    "bugfix": PrCategory.fix,
    "refactor": PrCategory.refactor,
    "perf": PrCategory.perf,
    "docs": PrCategory.docs,
    "doc": PrCategory.docs,
    "test": PrCategory.test,
    "tests": PrCategory.test,
    "chore": PrCategory.chore,
    "style": PrCategory.style,
    "ci": PrCategory.ci,
    "build": PrCategory.build,
    "revert": PrCategory.revert,
}

_CONVENTIONAL_RE = re.compile(
    r"^\s*(?P<type>[a-z]+)"
    r"(?:\([^)]*\))?"  # optional scope
    r"!?"  # optional breaking-change bang
    r"\s*:",
    re.IGNORECASE,
)

# Title patterns that mean "not real work shipped." Conservative — only match clearly trivial.
_NOISE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*revert\b",
        r"\bversion\s*bump\b",
        r"\bbump\s+(?:version|deps|dependencies|to\s+v?\d)",
        r"^\s*merge\s+(?:branch|pull|remote-tracking)",
        r"\[bot\]",
        r"\bdependabot\b",
        r"^\s*chore\(deps(?:-dev)?\)",
        r"^\s*chore\(release\)",
        r"^\s*release\s+v?\d",
        r"\bauto[- ]?generated\b",
        r"^\s*\(?wip\)?\s*:",
        r"^\s*draft\s*:",
    )
]


# Size buckets are [lo, hi) on total LoC changed (additions + deletions).
_SIZE_BUCKETS: list[tuple[str, int, int]] = [
    ("XS", 0, 10),
    ("S", 10, 50),
    ("M", 50, 300),
    ("L", 300, 1000),
    ("XL", 1000, 10**9),
]

# Base points per conventional-commits type. Perf > feature > fix > refactor by design:
# perf wins are unambiguously good; feature ships value; fix repairs broken value; refactor
# tightens internals but rarely moves a user metric. Docs/test/chore/style/ci start at 0.
_CATEGORY_BASE_SCORE: dict[PrCategory, int] = {
    PrCategory.perf: 35,
    PrCategory.feature: 30,
    PrCategory.fix: 22,
    PrCategory.refactor: 12,
    PrCategory.docs: 0,
    PrCategory.test: 0,
    PrCategory.chore: 0,
    PrCategory.style: 0,
    PrCategory.ci: 0,
    PrCategory.build: 0,
    PrCategory.revert: -50,
    PrCategory.other: 5,  # an untagged PR isn't zero — it's just unlabeled
}


# Body keyword groups → boost. We look at title+body together. The phrasing here is
# intentionally generous (e.g. matches "customer-facing" and "for the customer") so
# senior-engineer PRs that mention real product impact get credit.
_BODY_SIGNAL_GROUPS: list[tuple[str, int, list[str]]] = [
    (
        "customer_impact",
        20,
        ["customer", "user-facing", "user facing", "ux", "product impact", "blocker"],
    ),
    (
        "performance",
        18,
        [
            "performance",
            "perf ",
            "latency",
            "throughput",
            "p95",
            "p99",
            "slow",
            "speedup",
            "speed up",
            "optimization",
            "optimisation",
            "memory leak",
            "n+1",
            "scalab",
        ],
    ),
    (
        "architecture",
        18,
        [
            "architecture",
            "redesign",
            "rewrite",
            "migration",
            "refactor at scale",
            "decouple",
            "split out",
            "extract service",
        ],
    ),
    (
        "reliability",
        15,
        [
            "regression",
            "incident",
            "outage",
            "hotfix",
            "data loss",
            "race condition",
            "deadlock",
            "crash",
            "panic",
        ],
    ),
    (
        "tech_debt",
        10,
        [
            "tech debt",
            "technical debt",
            "cleanup at scale",
            "legacy",
            "deprecat",
        ],
    ),
    (
        "security",
        18,
        ["security", "cve", "vulnerab", "auth bypass", "leaked", "injection"],
    ),
]


# Title-only penalties — these don't override conventional-commits type, they pull it down.
# Calibrated so a "feat: rename ..." PR scores < 40 (the meaningful threshold) even at
# medium size with normal reviewer count. Renames, typos, lint, and formatting are exactly
# the category of "looks like feat but isn't meaningful product work" the user called out.
_TITLE_PENALTIES: list[tuple[str, int]] = [
    (r"\b(typo|spelling|comment)\b", -20),
    (r"\blint(er|ing)?\b", -14),
    (r"\brename\b", -14),
    (r"\bformat(ting)?\b", -14),
    (r"\bremove\s+unused\b", -12),
    (r"\bwhitespace\b", -14),
    (r"^\s*nit\b", -14),
]
_TITLE_PENALTY_RES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(p, re.IGNORECASE), score) for p, score in _TITLE_PENALTIES
]


# Cross-cutting bonus: PRs that touch many files signal something architectural / wide-impact.
def _files_touched_score(changed_files: int) -> int:
    if changed_files >= 25:
        return 12
    if changed_files >= 10:
        return 6
    if changed_files >= 4:
        return 3
    return 0


# Size bonus: logarithmic so a 10K-LoC PR doesn't dwarf a 200-LoC perf fix.
def _size_score(lines_changed: int) -> int:
    if lines_changed <= 5:
        return 0
    # log10 gives: 10 LoC → 1, 100 LoC → 2, 1000 LoC → 3, 10K LoC → 4.
    # Scale to a 0-20 range so it contributes but doesn't dominate.
    return min(20, int(math.log10(lines_changed) * 6))


def _review_score(review_count: int) -> int:
    """A non-trivial PR with zero reviews is suspicious. Treat 1+ reviews as a soft positive."""
    if review_count == 0:
        return -5
    if review_count >= 2:
        return 3
    return 1


# Thresholds chosen so most legit "feat: do X" PRs land ≥40, perf wins ≥70, and pure
# noise stays <15 even when it survives the regex filter.
_SCORE_MOVED_THE_NEEDLE = 70
_SCORE_REAL_WORK = 40
_SCORE_ROUTINE = 15


class SignificanceTier(str, Enum):
    moved_needle = "moved-the-needle"
    real_work = "real-work"
    routine = "routine"
    noise = "noise"


def categorize_title(title: str) -> PrCategory:
    """Parse the conventional-commits prefix and map to a category."""
    match = _CONVENTIONAL_RE.match(title or "")
    if not match:
        return PrCategory.other
    prefix = match.group("type").lower()
    return _CONVENTIONAL_TYPE_MAP.get(prefix, PrCategory.other)


def size_bucket(lines_changed: int) -> str:
    for label, lo, hi in _SIZE_BUCKETS:
        if lo <= lines_changed < hi:
            return label
    return "XL"


def is_noise_title(title: str) -> bool:
    return any(p.search(title or "") for p in _NOISE_PATTERNS)


def _signal_hits(text: str) -> list[str]:
    """Which body-signal groups did this PR's text trigger? Returns group names."""
    lowered = text.lower() if text else ""
    if not lowered:
        return []
    hits: list[str] = []
    for group_name, _boost, keywords in _BODY_SIGNAL_GROUPS:
        if any(kw in lowered for kw in keywords):
            hits.append(group_name)
    return hits


def score_pr(pr: PullRequestRecord) -> tuple[int, SignificanceTier, list[str]]:
    """Score a PR's significance.

    Returns ``(score, tier, signals)`` where:
      * ``score`` is 0-100 (clamped),
      * ``tier`` is one of ``SignificanceTier``,
      * ``signals`` is the list of body-keyword groups that matched, for tooltip/debug.
    """
    if not pr.shipped:
        return 0, SignificanceTier.noise, []
    if is_noise_title(pr.title):
        return 0, SignificanceTier.noise, ["noise_title"]
    cat = categorize_title(pr.title)
    raw = _CATEGORY_BASE_SCORE.get(cat, 0)

    # Title penalties — these can pull a feat: rename below the meaningful line, which
    # is the whole point of having them.
    for pattern, penalty in _TITLE_PENALTY_RES:
        if pattern.search(pr.title or ""):
            raw += penalty

    # Body keyword signals.
    signals = _signal_hits(f"{pr.title}\n{pr.body}")
    signal_set = set(signals)
    for group_name, boost, _kws in _BODY_SIGNAL_GROUPS:
        if group_name in signal_set:
            raw += boost

    raw += _size_score(pr.total_lines_changed)
    raw += _files_touched_score(pr.changed_files)
    raw += _review_score(pr.review_count)

    score = max(0, min(100, raw))

    if score >= _SCORE_MOVED_THE_NEEDLE:
        tier = SignificanceTier.moved_needle
    elif score >= _SCORE_REAL_WORK:
        tier = SignificanceTier.real_work
    elif score >= _SCORE_ROUTINE:
        tier = SignificanceTier.routine
    else:
        tier = SignificanceTier.noise
    return score, tier, signals


def is_meaningful(pr: PullRequestRecord) -> bool:
    """Did this PR actually ship something users would notice?

    Backward-compat alias: now defined as score ≥ ``_SCORE_REAL_WORK``. Tests that
    expect the old behavior still pass because the rule-based filters still drop the
    same noise patterns first.
    """
    _, tier, _ = score_pr(pr)
    return tier in (SignificanceTier.moved_needle, SignificanceTier.real_work)


@dataclass
class TriagedPr:
    pr: PullRequestRecord
    category: PrCategory
    size_bucket: str
    is_meaningful: bool
    score: int
    tier: SignificanceTier
    signals: list[str]


@dataclass
class RepoShipped:
    repo: str
    total_prs: int = 0
    meaningful: int = 0
    features: int = 0
    fixes: int = 0
    refactors: int = 0
    other_meaningful: int = 0
    loc_meaningful: int = 0


@dataclass
class ShippedReport:
    triaged: list[TriagedPr]
    total_prs: int
    total_meaningful: int
    by_category: dict[str, int]  # PrCategory.value -> count (str keys for jinja friendliness)
    by_size: dict[str, int]
    by_tier: dict[str, int]  # SignificanceTier.value -> count
    by_repo: list[RepoShipped]
    top_meaningful: list[TriagedPr]
    moved_the_needle: list[TriagedPr]  # subset of top_meaningful, score ≥ 70
    window_days: int

    @property
    def meaningful_share(self) -> float:
        if self.total_prs == 0:
            return 0.0
        return self.total_meaningful / self.total_prs


def triage(prs: list[PullRequestRecord]) -> list[TriagedPr]:
    """Classify every PR; pure function, no aggregation."""
    result: list[TriagedPr] = []
    for pr in prs:
        score, tier, signals = score_pr(pr)
        result.append(
            TriagedPr(
                pr=pr,
                category=categorize_title(pr.title),
                size_bucket=size_bucket(pr.total_lines_changed),
                is_meaningful=tier in (SignificanceTier.moved_needle, SignificanceTier.real_work),
                score=score,
                tier=tier,
                signals=signals,
            )
        )
    return result


def build_shipped_report(prs: list[PullRequestRecord], *, window_days: int = 90) -> ShippedReport:
    """Compute the full 'what did I ship' report."""
    triaged = triage(prs)
    meaningful_only = [t for t in triaged if t.is_meaningful]

    by_category: dict[str, int] = {}
    for t in meaningful_only:
        by_category[t.category.value] = by_category.get(t.category.value, 0) + 1

    by_size: dict[str, int] = {}
    for t in meaningful_only:
        by_size[t.size_bucket] = by_size.get(t.size_bucket, 0) + 1

    by_tier: dict[str, int] = {tier.value: 0 for tier in SignificanceTier}
    for t in triaged:
        by_tier[t.tier.value] = by_tier.get(t.tier.value, 0) + 1

    by_repo_map: dict[str, RepoShipped] = {}
    for t in triaged:
        s = by_repo_map.setdefault(t.pr.repo, RepoShipped(repo=t.pr.repo))
        s.total_prs += 1
        if t.is_meaningful:
            s.meaningful += 1
            s.loc_meaningful += t.pr.total_lines_changed
            if t.category == PrCategory.feature:
                s.features += 1
            elif t.category == PrCategory.fix:
                s.fixes += 1
            elif t.category in (PrCategory.refactor, PrCategory.perf):
                s.refactors += 1
            else:
                s.other_meaningful += 1
    by_repo = sorted(by_repo_map.values(), key=lambda r: (r.meaningful, r.loc_meaningful), reverse=True)

    # Showcase: highest-significance PRs first, with recency as tiebreak. This is the
    # list that drives the "top shipped" carousel — significance > raw LoC.
    top_meaningful = sorted(
        meaningful_only,
        key=lambda t: (
            t.score,
            t.pr.merged_at.timestamp() if t.pr.merged_at else 0.0,
        ),
        reverse=True,
    )[:15]

    moved_needle = [t for t in top_meaningful if t.tier == SignificanceTier.moved_needle]

    return ShippedReport(
        triaged=triaged,
        total_prs=len(triaged),
        total_meaningful=len(meaningful_only),
        by_category=by_category,
        by_size=by_size,
        by_tier=by_tier,
        by_repo=by_repo,
        top_meaningful=top_meaningful,
        moved_the_needle=moved_needle,
        window_days=window_days,
    )
