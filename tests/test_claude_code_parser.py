"""Smoke tests for the Claude Code transcript parser and cost computation."""

from __future__ import annotations

import json
from pathlib import Path

from anvil.analysis.claude_cost import build_cost_report
from anvil.analysis.pricing import cache_hit_rate, estimate_cost, price_for_model
from anvil.parsers.claude_code import (
    TurnUsage,
    parse_session_file,
    scan_claude_code_projects,
)


def _write_session(path: Path, lines: list[dict]) -> Path:  # type: ignore[type-arg]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")
    return path


def test_parse_real_world_shape(tmp_path: Path) -> None:
    """Mimic the actual Claude Code JSONL shape we observed on disk."""
    session_file = tmp_path / "projects" / "-Users-me-proj" / "sess-1.jsonl"
    _write_session(
        session_file,
        [
            {"type": "file-history-snapshot", "messageId": "x"},
            {
                "type": "user",
                "isMeta": True,
                "message": {"role": "user", "content": "<local-command-caveat>noise</local-command-caveat>"},
                "timestamp": "2026-04-01T22:05:35.508Z",
                "cwd": "/Users/me/proj",
                "gitBranch": "main",
                "version": "2.1.89",
                "entrypoint": "cli",
                "sessionId": "sess-1",
            },
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                "timestamp": "2026-04-01T22:05:36.000Z",
                "cwd": "/Users/me/proj",
                "sessionId": "sess-1",
                "isSidechain": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "hi there"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 200,
                        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
                    },
                },
                "timestamp": "2026-04-01T22:05:37.000Z",
                "sessionId": "sess-1",
            },
        ],
    )

    session = parse_session_file(session_file)
    assert session is not None
    assert session.session_id == "sess-1"
    assert session.cwd == "/Users/me/proj"
    assert session.git_branch == "main"
    assert session.entrypoint == "cli"
    assert session.first_user_text == "hello"
    assert session.total_usage.input_tokens == 100
    assert session.total_usage.output_tokens == 50
    assert session.total_usage.cache_read_input_tokens == 200
    assert session.total_usage.web_search_requests == 1
    assert "claude-sonnet-4-5-20250929" in session.usage_by_model


def test_scan_finds_all_projects(tmp_path: Path) -> None:
    _write_session(
        tmp_path / "projects" / "p1" / "s1.jsonl",
        [{"type": "user", "message": {"role": "user", "content": "x"}, "sessionId": "s1"}],
    )
    _write_session(
        tmp_path / "projects" / "p2" / "s2.jsonl",
        [{"type": "user", "message": {"role": "user", "content": "y"}, "sessionId": "s2"}],
    )
    scan = scan_claude_code_projects(tmp_path / "projects")
    assert scan.total_files_seen == 2
    assert len(scan.sessions) == 2


def test_price_lookup_matches_family_prefix() -> None:
    assert price_for_model("claude-sonnet-4-5-20250929") is not None
    assert price_for_model("claude-haiku-4-5") is not None
    assert price_for_model("<synthetic>") is None
    assert price_for_model(None) is None


def test_estimate_cost_zero_for_synthetic() -> None:
    usage = TurnUsage(input_tokens=1_000_000, output_tokens=500_000)
    assert estimate_cost(usage, "<synthetic>") == 0.0


def test_estimate_cost_for_known_model() -> None:
    # 1M input @ $3/MTok + 500k output @ $15/MTok = $3 + $7.50 = $10.50
    usage = TurnUsage(input_tokens=1_000_000, output_tokens=500_000)
    cost = estimate_cost(usage, "claude-sonnet-4-5-20250929")
    assert abs(cost - 10.50) < 0.001


def test_cache_hit_rate_basic() -> None:
    assert cache_hit_rate(TurnUsage()) is None
    rate = cache_hit_rate(TurnUsage(input_tokens=100, cache_read_input_tokens=900))
    assert rate is not None
    assert abs(rate - 0.9) < 1e-9


def test_cost_report_rolls_up_by_model_and_project(tmp_path: Path) -> None:
    _write_session(
        tmp_path / "projects" / "p1" / "s1.jsonl",
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": "ok",
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 500_000},
                },
                "timestamp": "2026-04-01T00:00:00Z",
                "cwd": "/proj/one",
                "sessionId": "s1",
            }
        ],
    )
    scan = scan_claude_code_projects(tmp_path / "projects")
    report = build_cost_report(scan)
    assert report.total_input_tokens == 1_000_000
    assert report.estimated_total_cost_usd > 0
    assert any(m.model.startswith("claude-sonnet-4-5") for m in report.by_model)
    assert any(p.cwd == "/proj/one" for p in report.by_project)
