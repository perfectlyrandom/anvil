"""Tests for anvil.analysis.shipped — meaningful-PRs classification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from anvil.analysis.shipped import (
    PrCategory,
    SignificanceTier,
    build_shipped_report,
    categorize_title,
    is_meaningful,
    is_noise_title,
    score_pr,
    size_bucket,
    triage,
)
from anvil.connectors.github import PullRequestRecord


def _pr(
    *,
    title: str,
    additions: int = 100,
    deletions: int = 20,
    merged: bool = True,
    repo: str = "acme/api",
    number: int = 1,
    body: str = "",
    review_count: int = 2,  # a default 'meaningful' PR has reviewers; un-reviewed merges are penalized
    changed_files: int = 3,
) -> PullRequestRecord:
    now = datetime(2026, 5, 1, tzinfo=UTC)
    return PullRequestRecord(
        number=number,
        repo=repo,
        title=title,
        state="MERGED" if merged else "OPEN",
        created_at=now - timedelta(days=1),
        merged_at=now if merged else None,
        closed_at=None,
        additions=additions,
        deletions=deletions,
        changed_files=changed_files,
        is_draft=False,
        url=f"https://github.com/{repo}/pull/{number}",
        body=body,
        review_count=review_count,
    )


class TestCategorizeTitle:
    @pytest.mark.parametrize(
        "title, expected",
        [
            ("feat: add foo", PrCategory.feature),
            ("feat(api): add foo", PrCategory.feature),
            ("feat!: breaking change", PrCategory.feature),
            ("fix(scope): null check", PrCategory.fix),
            ("fix: regression", PrCategory.fix),
            ("refactor: split module", PrCategory.refactor),
            ("perf: cache scan", PrCategory.perf),
            ("docs: README", PrCategory.docs),
            ("test: add coverage", PrCategory.test),
            ("chore: cleanup", PrCategory.chore),
            ("revert: oops", PrCategory.revert),
            ("Add new endpoint", PrCategory.other),  # no conventional prefix
            ("", PrCategory.other),
        ],
    )
    def test_conventional_prefixes(self, title: str, expected: PrCategory) -> None:
        # Given/When/Then: parsing the conventional-commits prefix returns the right category.
        assert categorize_title(title) == expected


class TestSizeBucket:
    @pytest.mark.parametrize(
        "loc, expected",
        [
            (0, "XS"),
            (5, "XS"),
            (9, "XS"),
            (10, "S"),
            (49, "S"),
            (50, "M"),
            (299, "M"),
            (300, "L"),
            (999, "L"),
            (1000, "XL"),
            (50_000, "XL"),
        ],
    )
    def test_buckets(self, loc: int, expected: str) -> None:
        assert size_bucket(loc) == expected


class TestNoiseDetection:
    @pytest.mark.parametrize(
        "title",
        [
            'Revert "some change"',
            "chore(deps): bump axios from 1.0 to 1.1",
            "chore(deps-dev): bump pytest",
            "chore(release): 1.2.3",
            "version bump",
            "Bump version to 0.5.0",
            "release v1.2.3",
            "Merge branch 'main' into feature",
            "Merge pull request #123",
            "[bot] auto-update",
            "dependabot: bump foo",
            "WIP: working on something",
            "draft: not done",
        ],
    )
    def test_clearly_noise(self, title: str) -> None:
        assert is_noise_title(title) is True

    @pytest.mark.parametrize(
        "title",
        [
            "feat: add caching",
            "fix: handle empty list",
            "Add new endpoint for traces",
            "refactor: split parser into modules",
            "perf: cache scan results",
        ],
    )
    def test_clearly_not_noise(self, title: str) -> None:
        assert is_noise_title(title) is False


class TestIsMeaningful:
    def test_feature_with_reviews_is_meaningful(self) -> None:
        # Given/When/Then: a reviewed feat PR with normal size crosses the meaningful bar.
        assert is_meaningful(_pr(title="feat: add a thing", merged=True)) is True

    def test_fix_with_body_signal_is_meaningful(self) -> None:
        # Given/When/Then: a fix that mentions customer impact lands as meaningful.
        pr = _pr(title="fix: null check", body="customer was hitting a 500 here", merged=True)
        assert is_meaningful(pr) is True

    def test_perf_is_almost_always_meaningful(self) -> None:
        # Given/When/Then: perf wins are unambiguously meaningful even when small.
        assert is_meaningful(_pr(title="perf: cache scan", additions=30, deletions=5)) is True

    def test_unmerged_is_not_meaningful(self) -> None:
        # Given/When/Then: unmerged PRs never count as shipped.
        assert is_meaningful(_pr(title="feat: add a thing", merged=False)) is False

    def test_chore_is_not_meaningful(self) -> None:
        # Given/When/Then: bare chores have zero base score.
        assert is_meaningful(_pr(title="chore: cleanup", merged=True)) is False

    def test_docs_is_not_meaningful(self) -> None:
        # Given/When/Then: docs are intentionally not "shipped product work".
        assert is_meaningful(_pr(title="docs: README", merged=True)) is False

    def test_revert_is_not_meaningful(self) -> None:
        # Given/When/Then: revert via title pattern hits the noise filter immediately.
        assert is_meaningful(_pr(title="Revert old change", merged=True)) is False

    def test_version_bump_is_not_meaningful(self) -> None:
        # Given/When/Then: release-train PRs hit the noise filter.
        assert is_meaningful(_pr(title="chore(release): 1.2.3", merged=True)) is False

    def test_feat_rename_is_routine_not_meaningful(self) -> None:
        # Given/When/Then: a "feat: rename X to Y" PR is intentionally penalized below the bar.
        pr = _pr(title="feat: rename foo to bar", additions=200, deletions=200)
        assert is_meaningful(pr) is False

    def test_feat_typo_is_routine_not_meaningful(self) -> None:
        # Given/When/Then: typo / comment-only PRs do not qualify even if they're labeled "feat:".
        pr = _pr(title="feat: fix typo in comment", additions=2, deletions=2)
        assert is_meaningful(pr) is False

    def test_large_untagged_with_no_review_is_not_meaningful(self) -> None:
        # Given/When/Then: a self-merged 800-LoC PR with no body signals is suspicious.
        pr = _pr(
            title="Reorganize the things",
            additions=500,
            deletions=300,
            review_count=0,
            body="",
        )
        assert is_meaningful(pr) is False

    def test_large_perf_pr_is_moved_the_needle(self) -> None:
        # Given/When/Then: a reviewed perf PR with customer + latency signals scores into the top tier.
        pr = _pr(
            title="perf: cut p99 latency by 40%",
            additions=300,
            deletions=100,
            body="Customer was complaining about latency on the ingest path. p99 down from 4s to 2.4s.",
            review_count=3,
            changed_files=8,
        )
        score, tier, signals = score_pr(pr)
        assert tier == SignificanceTier.moved_needle
        assert score >= 70
        assert "performance" in signals
        assert "customer_impact" in signals


class TestBuildShippedReport:
    def test_aggregates_correctly_by_category(self) -> None:
        # Given: a mix of meaningful and trivial PRs, with default review_count=2.
        prs = [
            _pr(title="feat: a", number=1),
            _pr(title="feat: b", number=2),
            _pr(title="fix: c", number=3, body="customer impact"),
            _pr(title="refactor: d", number=4, body="architecture redesign"),
            _pr(title="chore: e", number=5),
            _pr(title="docs: f", number=6),
            _pr(title="Revert oops", number=7),
        ]

        # When: building the report.
        report = build_shipped_report(prs)

        # Then: categories and meaningful counts are correct.
        assert report.total_prs == 7
        assert report.total_meaningful == 4  # 2 feat + 1 fix + 1 refactor
        assert report.by_category == {"feature": 2, "fix": 1, "refactor": 1}

    def test_groups_by_repo(self) -> None:
        # Given: PRs across two repos with default body signals turned on.
        prs = [
            _pr(title="feat: a", repo="acme/api", number=1),
            _pr(title="fix: b", repo="acme/api", number=2, body="regression hotfix"),
            _pr(title="feat: c", repo="acme/ui", number=3),
            _pr(title="chore: d", repo="acme/api", number=4),
        ]

        # When: building the report.
        report = build_shipped_report(prs)

        # Then: by_repo is sorted by meaningful count descending.
        assert len(report.by_repo) == 2
        assert report.by_repo[0].repo == "acme/api"
        assert report.by_repo[0].meaningful == 2
        assert report.by_repo[0].total_prs == 3  # includes the chore
        assert report.by_repo[1].repo == "acme/ui"
        assert report.by_repo[1].meaningful == 1

    def test_top_meaningful_sorted_by_significance_not_size(self) -> None:
        # Given: a small perf PR, a small bare feat, and a huge bare feat — significance > LoC.
        prs = [
            _pr(title="feat: small", additions=10, deletions=5, number=1),
            _pr(
                title="perf: tiny win on hot path",
                additions=15,
                deletions=2,
                number=2,
                body="cuts p99 latency by 30% on the ingest endpoint, customer-visible",
            ),
            _pr(title="feat: medium", additions=80, deletions=20, number=3),
        ]

        # When: building the report.
        report = build_shipped_report(prs)

        # Then: the perf+customer PR ranks highest despite being tiny.
        assert [t.pr.number for t in report.top_meaningful[:1]] == [2]

    def test_meaningful_share(self) -> None:
        # Given: 4 of 10 PRs cross the threshold, the rest are bare chores.
        prs = [_pr(title="feat: thing " + str(i), number=i, body="customer-facing change") for i in range(1, 5)] + [
            _pr(title="chore: b", number=i) for i in range(5, 11)
        ]

        # When: building the report.
        report = build_shipped_report(prs)

        # Then: meaningful_share is 0.4.
        assert report.meaningful_share == pytest.approx(0.4)


class TestTriage:
    def test_returns_one_entry_per_pr_with_score(self) -> None:
        # Given: a meaningful feat PR and a bare chore PR.
        prs = [_pr(title="feat: a", number=1), _pr(title="chore: b", number=2)]

        # When: triaging.
        out = triage(prs)

        # Then: every PR has category, size_bucket, score, tier, signals.
        assert len(out) == 2
        assert out[0].category == PrCategory.feature
        assert out[0].is_meaningful is True
        assert out[0].tier in (SignificanceTier.moved_needle, SignificanceTier.real_work)
        assert out[0].score >= 40
        assert out[1].category == PrCategory.chore
        assert out[1].is_meaningful is False
        assert out[1].tier in (SignificanceTier.routine, SignificanceTier.noise)
