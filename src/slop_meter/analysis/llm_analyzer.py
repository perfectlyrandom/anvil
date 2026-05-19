"""LLM-powered prompt analyzer.

Samples user prompts from a CursorScanResult, sends them to Claude Sonnet 4.5 with web
search enabled, and asks for concrete optimization advice grounded in current best
practices. Returns a markdown report.

Key design choices:

- We send only ``<user_query>`` content (not workspace rules / file context / system
  reminders) since the goal is to critique the user's prompting behavior, not Cursor's
  injected context.
- We sample by token estimate (largest first) so we focus on the prompts most likely to
  benefit from optimization, not 5-word followups.
- Web search is enabled so the model can ground recommendations in current 2025/2026
  prompt-engineering literature.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from slop_meter.parsers.cursor_transcripts import SessionRecord


@dataclass
class PromptSample:
    session_id: str
    workspace: str
    first_user_query: str
    user_tokens_est: int


@dataclass
class AnalyzerReport:
    """Result of a Claude analysis run."""

    sample_size: int
    model: str
    used_web_search: bool
    markdown_report: str
    raw_response_summary: dict[str, int]


def select_samples(sessions: list[SessionRecord], *, n: int = 30) -> list[PromptSample]:
    """Pick the top ``n`` sessions by user-token estimate that have a non-empty user_query."""
    candidates = [s for s in sessions if s.first_user_query and not s.is_subagent]
    candidates.sort(key=lambda s: s.user_tokens_est, reverse=True)
    selected = candidates[:n]
    return [
        PromptSample(
            session_id=s.session_id,
            workspace=s.workspace,
            first_user_query=s.first_user_query or "",
            user_tokens_est=s.user_tokens_est,
        )
        for s in selected
    ]


def _build_user_message(samples: list[PromptSample], *, force_search: bool) -> str:
    lines = [
        "I'm building a tool that analyzes developers' AI coding tool usage and gives them concrete",
        "advice on improving their prompts. Below are real first-user-query prompts from this developer's",
        f"recent Cursor sessions ({len(samples)} samples, sorted by total estimated user-side tokens",
        "for that session, largest first). Each prompt is the OPENING prompt of a session - the rest of",
        "the conversation isn't shown.",
        "",
        "Please:",
        "1. Identify 3-5 patterns you see across these prompts (e.g. typo density, vagueness, missing",
        "   context, request size, prompt style).",
        "2. For each pattern, give one concrete tip the developer could apply.",
        "3. Be honest. If a pattern is fine and doesn't need fixing, say so. Don't manufacture problems.",
        "4. Highlight any 'green flags' you see - things they're already doing well.",
    ]
    if force_search:
        lines.extend(
            [
                "5. **REQUIRED**: Use the web_search tool at least 2-3 times to find current (2025-2026)",
                "   research, papers, or blog posts on the patterns you identify. Topics likely worth",
                "   searching: 'agentic prompt engineering best practices 2026', 'context engineering",
                "   for code agents', 'Cursor prompt optimization', 'AI coding tool ROI studies METR'.",
                "   Cite every URL you use inline using markdown links. Findings that surprise the",
                "   developer (i.e. things they wouldn't already know) are especially valuable.",
            ]
        )
    else:
        lines.append("5. Optionally use web_search if a pattern requires up-to-date evidence; cite any URLs.")
    lines.extend(
        [
            "",
            "Output a single concise markdown report (target ~600-1000 words). No preamble.",
            "",
            "SAMPLES:",
            "",
        ]
    )
    for i, sample in enumerate(samples, start=1):
        lines.append(
            f"### Sample {i} — workspace `{sample.workspace}` — ~{sample.user_tokens_est:,} user tokens in session"
        )
        lines.append("")
        lines.append("```")
        text = sample.first_user_query.strip()
        # Cap each sample so total request stays reasonable
        if len(text) > 4000:
            text = text[:4000] + "\n[...truncated...]"
        lines.append(text)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def run_analysis(
    samples: list[PromptSample],
    *,
    api_key: str,
    model: str = "claude-sonnet-4-5-20250929",
    enable_web_search: bool = True,
    force_web_search: bool = False,
    max_tokens: int = 4096,
) -> AnalyzerReport:
    """Call Anthropic with the samples and return an AnalyzerReport.

    ``force_web_search`` adds explicit ``REQUIRED`` instructions for the model to
    invoke web_search 2-3 times so we can surface up-to-date research the developer
    wouldn't already know about.
    """
    # Lazy import so the rest of the package doesn't require anthropic at import-time.
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    user_message = _build_user_message(samples, force_search=enable_web_search and force_web_search)

    system_prompt = (
        "You are a senior dev tools researcher analyzing a developer's actual AI coding tool prompts. "
        "Your job is honest, evidence-based critique - not validation, not hype. "
        "Use web search to ground recommendations in current 2025/2026 prompt-engineering best "
        "practices when relevant, and cite the sources. Be specific, not generic. Call out things "
        "they're doing well alongside the issues."
    )

    if enable_web_search:
        # Bump max_uses when forcing so the model isn't constrained mid-search.
        max_uses = 8 if force_web_search else 5
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}],
        )
    else:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

    # Anthropic may run web_search internally and produce multiple content blocks
    # in the final response. Collect text blocks; ignore tool_use blocks (results already
    # used by the model in producing the final text).
    text_pieces: list[str] = []
    web_searches = 0
    for block in response.content:
        if block.type == "text":
            text_pieces.append(block.text)
        elif block.type == "server_tool_use" and getattr(block, "name", "") == "web_search":
            web_searches += 1
        elif block.type == "web_search_tool_result":
            # Already counted on the server_tool_use side.
            pass

    markdown_report = "\n".join(text_pieces).strip()

    summary = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "web_searches": web_searches,
    }

    return AnalyzerReport(
        sample_size=len(samples),
        model=model,
        used_web_search=enable_web_search,
        markdown_report=markdown_report,
        raw_response_summary=summary,
    )


def get_api_key_or_fail() -> str:
    """Pull the Anthropic key from the environment (loaded from .env via pydantic-settings)."""
    key = os.environ.get("SLOP_METER_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "no Anthropic API key found. Set SLOP_METER_ANTHROPIC_API_KEY in .env or ANTHROPIC_API_KEY in the environment."
        )
    return key
