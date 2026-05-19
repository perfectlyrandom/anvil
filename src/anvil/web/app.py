"""FastAPI app for the anvil dashboard.

Single-page dashboard with five sections (Overview / Cost / ROI / Deep / GitHub)
plus an "Ask" chat panel that talks to the Anthropic-backed agent.

Each section is rendered server-side and refreshed via htmx fragment requests, so
there's no JS build step. Templates live next to this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from anvil.agent.loop import Agent
from anvil.agent.tools import ToolContext
from anvil.analysis.claude_cost import build_cost_report
from anvil.analysis.coach import run_coach
from anvil.analysis.cost_breakdown import build_cost_breakdown
from anvil.analysis.cursor_deep import deep_analyze
from anvil.analysis.cursor_stats import aggregate
from anvil.analysis.pricing import estimate_cost
from anvil.analysis.roi import correlate
from anvil.analysis.shipped import build_shipped_report
from anvil.analysis.skills import build_skills_audit
from anvil.analysis.tips import build_tips, summary_savings
from anvil.config import Settings, load_settings
from anvil.connectors.github import GitHubConnectorError, current_login
from anvil.web.cache import ScanCache

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _format_int(value: int | float) -> str:
    return f"{int(value):,}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _format_usd(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


def _truncate(value: str | None, limit: int = 120) -> str:
    if not value:
        return ""
    cleaned = value.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "…"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. ``settings`` injection makes the app testable."""
    settings = settings or load_settings()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["intfmt"] = _format_int
    templates.env.filters["pct"] = _format_pct
    templates.env.filters["usd"] = _format_usd
    templates.env.filters["trunc"] = _truncate

    app = FastAPI(title="anvil", docs_url=None, redoc_url=None)
    cache = ScanCache()
    claude_projects_dir = Path.home() / ".claude" / "projects"

    # Lazy GitHub login resolution: failing to find gh shouldn't blow up the dashboard.
    try:
        gh_login = current_login()
    except GitHubConnectorError:
        gh_login = None

    tool_ctx = ToolContext(
        cursor_projects_dir=settings.cursor_projects_dir,
        claude_projects_dir=Path.home() / ".claude" / "projects",
        codex_sessions_dir=settings.codex_sessions_dir,
        cursor_tracking_db=settings.cursor_tracking_db,
        default_pricing_model=settings.default_pricing_model,
        github_login=gh_login,
    )
    # One agent per process; the chat is a single-user toy. Persist across requests.
    agent: Agent | None = None
    if settings.anthropic_api_key:
        agent = Agent(
            api_key=settings.anthropic_api_key,
            ctx=tool_ctx,
            model=settings.anthropic_model,
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "gh_login": gh_login,
                "agent_available": agent is not None,
                "model": settings.anthropic_model,
            },
        )

    @app.get("/fragments/overview", response_class=HTMLResponse)
    def fragment_overview(request: Request) -> HTMLResponse:
        scan = cache.cursor(settings.cursor_projects_dir)
        agg = aggregate(scan) if scan.sessions else None
        return templates.TemplateResponse(request, "_overview.html", {"scan": scan, "agg": agg})

    @app.get("/fragments/session/{session_id}", response_class=HTMLResponse)
    def fragment_session(request: Request, session_id: str) -> HTMLResponse:
        scan = cache.cursor(settings.cursor_projects_dir)
        session = next((s for s in scan.sessions if s.session_id == session_id), None)
        if session is None:
            return HTMLResponse(
                f"<div class='bg-elev border border-line p-6 rounded-sm'>"
                f"<p class='font-mono text-sm text-warn'>session {session_id} not found</p>"
                f"</div>"
            )
        # Per-turn ordering by tokens lets us highlight where the burn actually happened.
        heavy_turns = sorted(session.turns, key=lambda t: t.estimated_tokens, reverse=True)[:5]
        # Bucket breakdown sorted by tokens, only buckets that actually carry weight.
        total_user = session.user_tokens_est or 1
        bucket_rows = sorted(
            ((name, tokens, tokens / total_user) for name, tokens in session.bucket_tokens.items() if tokens > 0),
            key=lambda row: row[1],
            reverse=True,
        )
        return templates.TemplateResponse(
            request,
            "_session_detail.html",
            {
                "session": session,
                "bucket_rows": bucket_rows,
                "heavy_turns": heavy_turns,
            },
        )

    @app.get("/fragments/cost", response_class=HTMLResponse)
    def fragment_cost(request: Request) -> HTMLResponse:
        cursor_scan = cache.cursor(settings.cursor_projects_dir)
        claude_scan = cache.claude(claude_projects_dir)
        codex_scan = cache.codex(settings.codex_sessions_dir)
        tracking = cache.cursor_tracking(settings.cursor_tracking_db)
        breakdown = build_cost_breakdown(
            cursor_scan,
            claude_scan,
            codex_scan,
            cursor_tracking=tracking,
            fallback_model=settings.default_pricing_model,
        )
        return templates.TemplateResponse(request, "_cost.html", {"breakdown": breakdown})

    @app.get("/fragments/coach", response_class=HTMLResponse)
    def fragment_coach(request: Request) -> HTMLResponse:
        """Harsh expert critique of the user's AI setup. Refreshes every visit."""
        cursor_scan = cache.cursor(settings.cursor_projects_dir)
        claude_scan = cache.claude(claude_projects_dir)
        codex_scan = cache.codex(settings.codex_sessions_dir)
        tracking = cache.cursor_tracking(settings.cursor_tracking_db)
        breakdown = build_cost_breakdown(
            cursor_scan,
            claude_scan,
            codex_scan,
            cursor_tracking=tracking,
            fallback_model=settings.default_pricing_model,
        )
        deep = deep_analyze(cursor_scan) if cursor_scan.sessions else None
        agg = aggregate(cursor_scan) if cursor_scan.sessions else None
        skills_records = cache.skills(settings.cursor_skills_dir)
        skills_report = (
            build_skills_audit(
                skills_records,
                cursor_scan,
                extra_consulted_by_name=_extra_skill_consultations(skills_records),
            )
            if skills_records
            else None
        )
        shipped = None
        if gh_login is not None:
            try:
                prs = cache.prs(gh_login, 90)
                shipped = build_shipped_report(prs, window_days=90)
            except GitHubConnectorError:
                shipped = None
        report = run_coach(
            breakdown,
            deep=deep,
            skills=skills_report,
            cursor_stats=agg,
            shipped=shipped,
        )
        return templates.TemplateResponse(request, "_coach.html", {"report": report})

    @app.get("/fragments/roi", response_class=HTMLResponse)
    def fragment_roi(request: Request, days: int = 90) -> HTMLResponse:
        if gh_login is None:
            return templates.TemplateResponse(
                request,
                "_roi.html",
                {"report": None, "error": "GitHub CLI not authenticated. Run `gh auth login`."},
            )
        try:
            prs = cache.prs(gh_login, days)
        except GitHubConnectorError as exc:
            return templates.TemplateResponse(request, "_roi.html", {"report": None, "error": str(exc)})
        claude_scan = cache.claude(claude_projects_dir)
        cost_per_session = {
            s.session_id: sum(estimate_cost(usage, model) for model, usage in s.usage_by_model.items())
            for s in claude_scan.sessions
        }
        report = correlate(claude_scan, prs, cost_per_session=cost_per_session)
        return templates.TemplateResponse(
            request,
            "_roi.html",
            {"report": report, "days": days, "login": gh_login, "error": None},
        )

    @app.get("/fragments/deep", response_class=HTMLResponse)
    def fragment_deep(request: Request) -> HTMLResponse:
        scan = cache.cursor(settings.cursor_projects_dir)
        report = deep_analyze(scan) if scan.sessions else None
        return templates.TemplateResponse(request, "_deep.html", {"report": report})

    @app.get("/fragments/tips", response_class=HTMLResponse)
    def fragment_tips(request: Request) -> HTMLResponse:
        scan = cache.cursor(settings.cursor_projects_dir)
        deep = deep_analyze(scan) if scan.sessions else None
        agg = aggregate(scan) if scan.sessions else None
        # Cache report is best-effort: if no Claude Code data, no cache tip.
        claude_scan = cache.claude(claude_projects_dir)
        cost = build_cost_report(claude_scan) if claude_scan.sessions and claude_scan.has_any_real_usage else None
        skills = cache.skills(settings.cursor_skills_dir)
        skills_report = (
            build_skills_audit(skills, scan, extra_consulted_by_name=_extra_skill_consultations(skills))
            if skills
            else None
        )
        tips = build_tips(
            deep,
            agg,
            cost,
            skills_report,
            pricing_model=settings.default_pricing_model,
        )
        total_tokens, total_usd = summary_savings(tips)
        return templates.TemplateResponse(
            request,
            "_tips.html",
            {
                "tips": tips,
                "total_tokens": total_tokens,
                "total_usd": total_usd,
                "pricing_model": settings.default_pricing_model,
            },
        )

    def _extra_skill_consultations(skill_records: list[Any]) -> dict[str, int]:
        """Sum Codex + Claude Code session counts per skill name.

        Used to cross-reference Cursor's consultation signal with the other AI tools, so
        skills invoked outside Cursor (or followed silently in Cursor but mentioned
        elsewhere) stop showing as stale. We pass the installed skill names down so the
        bare-token matcher (``baz-pr-comments`` in prose) only fires on known skills.
        """
        known_names = frozenset(s.full_path.parent.name for s in skill_records)
        claude_mentions = cache.skill_mentions(claude_projects_dir, "**/*.jsonl", "claude", known_names)
        codex_mentions = cache.skill_mentions(settings.codex_sessions_dir, "**/rollout-*.jsonl", "codex", known_names)
        return {
            name: claude_mentions.get(name, 0) + codex_mentions.get(name, 0)
            for name in set(claude_mentions) | set(codex_mentions)
        }

    @app.get("/fragments/skills", response_class=HTMLResponse)
    def fragment_skills(request: Request) -> HTMLResponse:
        scan = cache.cursor(settings.cursor_projects_dir)
        skills = cache.skills(settings.cursor_skills_dir)
        report = build_skills_audit(skills, scan, extra_consulted_by_name=_extra_skill_consultations(skills))
        return templates.TemplateResponse(request, "_skills.html", {"report": report})

    @app.get("/fragments/shipped", response_class=HTMLResponse)
    def fragment_shipped(request: Request, days: int = 90) -> HTMLResponse:
        if gh_login is None:
            return templates.TemplateResponse(
                request,
                "_shipped.html",
                {"report": None, "error": "GitHub CLI not authenticated. Run `gh auth login`."},
            )
        try:
            prs = cache.prs(gh_login, days)
        except GitHubConnectorError as exc:
            return templates.TemplateResponse(request, "_shipped.html", {"report": None, "error": str(exc)})
        report = build_shipped_report(prs, window_days=days)
        return templates.TemplateResponse(
            request,
            "_shipped.html",
            {"report": report, "days": days, "login": gh_login, "error": None},
        )

    @app.post("/chat", response_class=HTMLResponse)
    def chat(request: Request, message: str = Form(...)) -> HTMLResponse:
        if agent is None:
            return templates.TemplateResponse(
                request,
                "_chat_turn.html",
                {
                    "user_text": message,
                    "reply_text": (
                        "Agent is unavailable: ANVIL_ANTHROPIC_API_KEY is not set. "
                        "Add it to your .env and restart the server."
                    ),
                    "tool_calls": [],
                    "usage": None,
                },
            )
        try:
            reply = agent.send(message)
        except Exception as exc:
            logger.exception("agent.send failed")
            raise HTTPException(status_code=500, detail=f"agent error: {exc}") from exc
        return templates.TemplateResponse(
            request,
            "_chat_turn.html",
            {
                "user_text": message,
                "reply_text": reply.text,
                "tool_calls": reply.tool_calls_summary,
                "usage": {
                    "input_tokens": reply.input_tokens,
                    "output_tokens": reply.output_tokens,
                    "iterations": reply.iterations,
                },
            },
        )

    @app.post("/chat/reset")
    def chat_reset() -> JSONResponse:
        if agent is not None:
            agent.reset()
        return JSONResponse({"ok": True})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "agent_available": agent is not None,
            "gh_login": gh_login,
            "cursor_projects_dir": str(settings.cursor_projects_dir),
        }

    return app
