"""Tests for ROI correlation and deep Cursor analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from anvil.analysis.cursor_deep import deep_analyze
from anvil.analysis.roi import correlate
from anvil.connectors.github import PullRequestRecord
from anvil.parsers.claude_code import scan_claude_code_projects
from anvil.parsers.cursor_transcripts import scan_cursor_projects


def _write_jsonl(path: Path, lines: list[dict]) -> None:  # type: ignore[type-arg]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def _pr(
    number: int,
    *,
    created: str,
    merged: str | None = None,
    additions: int = 100,
    deletions: int = 50,
) -> PullRequestRecord:
    return PullRequestRecord(
        number=number,
        repo="me/proj",
        title=f"PR {number}",
        state="MERGED" if merged else "OPEN",
        created_at=datetime.fromisoformat(created),
        merged_at=datetime.fromisoformat(merged) if merged else None,
        closed_at=None,
        additions=additions,
        deletions=deletions,
        changed_files=3,
        is_draft=False,
        url=f"https://github.com/me/proj/pull/{number}",
    )


def test_correlate_assigns_costs_to_weeks(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "projects" / "p" / "s1.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": "ok",
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
                },
                "timestamp": "2026-04-06T12:00:00Z",  # Monday of ISO week 2026-W15
                "cwd": "/p",
                "sessionId": "s1",
            }
        ],
    )
    scan = scan_claude_code_projects(tmp_path / "projects")
    prs = [
        _pr(1, created="2026-04-07T12:00:00+00:00", merged="2026-04-08T12:00:00+00:00"),
        _pr(2, created="2026-04-14T12:00:00+00:00", merged="2026-04-15T12:00:00+00:00"),
    ]
    report = correlate(scan, prs, cost_per_session={"s1": 5.0})

    assert report.total_estimated_cost_usd == 5.0
    assert report.total_merged_prs == 2
    weeks_by_key = {w.iso_week: w for w in report.weeks}
    assert "2026-W15" in weeks_by_key
    assert weeks_by_key["2026-W15"].claude_code_estimated_cost_usd == 5.0
    assert weeks_by_key["2026-W15"].prs_merged == 1


def test_deep_analysis_catches_repeated_prompts(tmp_path: Path) -> None:
    ws_dir = tmp_path / "projects" / "ws-1" / "agent-transcripts"
    for sess_id in ("a", "b", "c"):
        session_dir = ws_dir / sess_id
        session_dir.mkdir(parents=True)
        (session_dir / f"{sess_id}.jsonl").write_text(
            json.dumps(
                {
                    "role": "user",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "<user_query>read my DMs and create a ticket</user_query>",
                            }
                        ]
                    },
                }
            )
            + "\n"
        )

    scan = scan_cursor_projects(tmp_path / "projects")
    deep = deep_analyze(scan)
    assert len(deep.repeated_prompt_clusters) >= 1
    cluster = deep.repeated_prompt_clusters[0]
    assert len(cluster.session_ids) == 3
    assert "create a ticket" in cluster.canonical_first_query


def test_deep_analysis_detects_forked_sessions(tmp_path: Path) -> None:
    # Given: two sessions in the same workspace with byte-identical first 3 turns
    ws_dir = tmp_path / "projects" / "ws-fork" / "agent-transcripts"
    leading_turns = [
        {
            "role": "user",
            "message": {"content": [{"type": "text", "text": "<user_query>build a feature X</user_query>"}]},
        },
        {
            "role": "assistant",
            "message": {"content": [{"type": "text", "text": "Sure, I'll start with the data layer."}]},
        },
        {
            "role": "user",
            "message": {"content": [{"type": "text", "text": "<user_query>ok go ahead</user_query>"}]},
        },
    ]
    for sess_id, divergent in (("forkA", "do the API"), ("forkB", "do the UI")):
        session_dir = ws_dir / sess_id
        session_dir.mkdir(parents=True)
        # Each session has the same first 3 turns then diverges on turn 4.
        all_turns = [
            *leading_turns,
            {
                "role": "assistant",
                "message": {"content": [{"type": "text", "text": divergent}]},
            },
        ]
        (session_dir / f"{sess_id}.jsonl").write_text("\n".join(json.dumps(t) for t in all_turns) + "\n")

    # When: deep analysis runs
    scan = scan_cursor_projects(tmp_path / "projects")
    deep = deep_analyze(scan)

    # Then: the fork detector flags both sessions as one duplicate cluster
    assert len(deep.forked_session_clusters) == 1
    cluster = deep.forked_session_clusters[0]
    assert sorted(cluster.session_ids) == ["forkA", "forkB"]
    assert cluster.identical_leading_turns >= 3
    assert cluster.duplicated_token_cost > 0
