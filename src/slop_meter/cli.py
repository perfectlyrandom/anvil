"""Typer CLI entry point.

All commands are stubs in commit 1 - they exit cleanly with a "not implemented" message
so the help surface, packaging, and pipx install all work end-to-end before any logic lands.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from slop_meter import __version__
from slop_meter.analysis.claude_cost import build_cost_report
from slop_meter.analysis.cursor_deep import deep_analyze
from slop_meter.analysis.cursor_stats import aggregate
from slop_meter.analysis.llm_analyzer import run_analysis, select_samples
from slop_meter.analysis.pricing import estimate_cost
from slop_meter.analysis.roi import correlate
from slop_meter.config import load_settings
from slop_meter.connectors.github import (
    GitHubConnectorError,
    current_login,
    default_since,
    fetch_authored_prs,
)
from slop_meter.db import create_engine_for_path, init_schema
from slop_meter.parsers.claude_code import scan_claude_code_projects
from slop_meter.parsers.cursor_transcripts import scan_cursor_projects

app = typer.Typer(
    name="slop_meter",
    help="How much of your AI workflow is shipping vs slop?",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def _not_implemented(command: str) -> None:
    console.print(f"[yellow]slop_meter {command}[/yellow] is not implemented yet (commit 1 - scaffolding only).")
    raise typer.Exit(code=0)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"slop_meter {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """slop_meter root."""
    del version


@app.command()
def init() -> None:
    """One-time setup. Creates ~/.slop_meter/, initializes SQLite schema, prompts for GitHub PAT."""
    settings = load_settings()
    engine = create_engine_for_path(settings.db_path)
    init_schema(engine)
    console.print(f"[green]initialized slop_meter at {settings.home_dir}[/green]")
    console.print(f"  database: {settings.db_path}")
    console.print("[dim]GitHub PAT prompt and keyring storage land in commit 3.[/dim]")


@app.command()
def watch(daemon: bool = typer.Option(False, "--daemon", help="Run in background.")) -> None:
    """Watch Cursor and Claude Code transcript directories for new turns."""
    del daemon
    _not_implemented("watch")


@app.command()
def web(
    host: str | None = typer.Option(None, "--host", help="Override bind host."),
    port: int | None = typer.Option(None, "--port", help="Override bind port."),
) -> None:
    """Launch the local FastAPI dashboard."""
    settings = load_settings()
    h = host or settings.web_host
    p = port or settings.web_port
    console.print(
        f"[yellow]slop_meter web[/yellow] dashboard scaffolding only - serving /healthz at http://{h}:{p}/healthz"
    )
    console.print("[dim]Full dashboard lands in commits 5-6.[/dim]")
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red]")
        raise typer.Exit(code=1) from None
    uvicorn.run("slop_meter.server:app", host=h, port=p, reload=False)


@app.command()
def top() -> None:
    """Live TUI view of recent activity (htop-style)."""
    _not_implemented("top")


@app.command()
def query(sql: str = typer.Argument(..., help="Raw SQL to run against the local database.")) -> None:
    """Datasette-style SQL escape hatch (read-only)."""
    del sql
    _not_implemented("query")


config_app = typer.Typer(help="View and edit slop_meter configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Print current effective configuration."""
    settings = load_settings()
    table = Table(title="slop_meter configuration", show_lines=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for key, value in settings.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


telemetry_app = typer.Typer(help="Manage opt-in telemetry.", no_args_is_help=True)
app.add_typer(telemetry_app, name="telemetry")


@telemetry_app.command("on")
def telemetry_on() -> None:
    """Enable opt-in anonymous telemetry."""
    _not_implemented("telemetry on")


@telemetry_app.command("off")
def telemetry_off() -> None:
    """Disable telemetry."""
    _not_implemented("telemetry off")


@telemetry_app.command("status")
def telemetry_status() -> None:
    """Show telemetry setting."""
    settings = load_settings()
    state = "on" if settings.telemetry_enabled else "off"
    console.print(f"telemetry: [cyan]{state}[/cyan]")


@app.command()
def doctor() -> None:
    """Check permissions, keychain access, schema version."""
    _not_implemented("doctor")


@app.command()
def export(out: str | None = typer.Option(None, "--out", help="Output path.")) -> None:
    """Dump the local database to a shareable file."""
    del out
    _not_implemented("export")


@app.command()
def analyze(
    source: str = typer.Option("cursor", "--source", help="Data source: 'cursor' or 'claude_code'."),
    with_llm: bool = typer.Option(False, "--with-llm", help="Run Claude analysis with web search on sampled prompts."),
    sample_size: int = typer.Option(30, "--samples", help="How many top sessions to send to the LLM analyzer."),
    no_web_search: bool = typer.Option(False, "--no-web-search", help="Disable web search in the LLM analyzer."),
    force_search: bool = typer.Option(
        False,
        "--force-search",
        help="Force the LLM to invoke web_search 2-3+ times for current research citations.",
    ),
    deep: bool = typer.Option(
        False,
        "--deep",
        help="Run extra analysis: mid-session sampling, cross-session bloat detection, repeated-prompt clusters.",
    ),
) -> None:
    """Scan local AI tool data, compute bucket stats, optionally get LLM critique of prompts."""
    if source == "cursor":
        _analyze_cursor(
            with_llm=with_llm,
            sample_size=sample_size,
            no_web_search=no_web_search,
            force_search=force_search,
            deep=deep,
        )
    elif source == "claude_code":
        _analyze_claude_code(deep=deep)
    else:
        console.print(f"[red]source '{source}' not supported. Use 'cursor' or 'claude_code'.[/red]")
        raise typer.Exit(code=1)


def _analyze_cursor(*, with_llm: bool, sample_size: int, no_web_search: bool, force_search: bool, deep: bool) -> None:
    settings = load_settings()
    console.print(f"[dim]scanning {settings.cursor_projects_dir}...[/dim]")
    scan = scan_cursor_projects(settings.cursor_projects_dir)
    if scan.total_files_seen == 0:
        console.print(f"[red]no transcripts found under {settings.cursor_projects_dir}[/red]")
        raise typer.Exit(code=1)

    agg = aggregate(scan)
    _print_overview(agg)
    _print_buckets(agg)
    _print_workspaces(agg)
    _print_longest_sessions(agg)

    if deep:
        deep_report = deep_analyze(scan)
        _print_deep_cursor(deep_report)

    if not with_llm:
        console.print()
        console.print(
            "[dim]tips:[/dim] [cyan]--with-llm[/cyan] for Claude critique, "
            "[cyan]--deep[/cyan] for mid-session bloat/repeat detection, "
            "[cyan]--force-search[/cyan] to force web_search citations."
        )
        return

    api_key = settings.anthropic_api_key
    if not api_key:
        console.print(
            "[red]no anthropic_api_key configured. Add SLOP_METER_ANTHROPIC_API_KEY to .env, "
            "or unset --with-llm.[/red]"
        )
        raise typer.Exit(code=1)

    samples = select_samples(scan.sessions, n=sample_size)
    if not samples:
        console.print("[yellow]no parseable user prompts found - nothing to analyze.[/yellow]")
        return

    web_status = "OFF" if no_web_search else ("FORCED" if force_search else "ON (model discretion)")
    console.print()
    console.print(
        f"[dim]running {settings.anthropic_model} on {len(samples)} prompt samples "
        f"(web search {web_status})...[/dim]"
    )
    report = run_analysis(
        samples,
        api_key=api_key,
        model=settings.anthropic_model,
        enable_web_search=not no_web_search,
        force_web_search=force_search,
    )

    console.print()
    console.rule("[bold magenta]LLM prompt critique[/bold magenta]")
    console.print(Markdown(report.markdown_report))
    console.rule()
    console.print(
        f"[dim]model: {report.model}  •  "
        f"input tokens: {report.raw_response_summary['input_tokens']:,}  •  "
        f"output tokens: {report.raw_response_summary['output_tokens']:,}  •  "
        f"web searches: {report.raw_response_summary['web_searches']}[/dim]"
    )


def _analyze_claude_code(*, deep: bool) -> None:
    del deep  # placeholder for future Claude-Code-specific deep analysis
    settings = load_settings()
    projects_dir = Path.home() / ".claude" / "projects"
    console.print(f"[dim]scanning {projects_dir}...[/dim]")
    scan = scan_claude_code_projects(projects_dir)
    if scan.total_files_seen == 0:
        console.print(f"[yellow]no Claude Code transcripts found under {projects_dir}[/yellow]")
        return
    if not scan.has_any_real_usage:
        console.print(
            "[yellow]found Claude Code transcripts, but no real usage tokens recorded "
            "(all sessions are synthetic / empty usage blocks). Cost data unavailable.[/yellow]"
        )
    report = build_cost_report(scan)
    _print_claude_code_overview(scan, report)
    _print_claude_code_models(report)
    _print_claude_code_projects(report)
    if report.unpriced_models:
        console.print(f"[dim]unpriced models (cost shown as $0.00): {', '.join(report.unpriced_models)}[/dim]")
    del settings


@app.command()
def cost(
    source: str = typer.Option(
        "claude_code", "--source", help="Data source: 'claude_code' (only real-token source today)."
    ),
) -> None:
    """Show estimated USD cost broken down by model and project."""
    if source != "claude_code":
        console.print(f"[red]source '{source}' has no real token counts. Try 'claude_code'.[/red]")
        raise typer.Exit(code=1)
    _analyze_claude_code(deep=False)


@app.command()
def roi(
    days: int = typer.Option(90, "--days", help="Lookback window in days for PRs."),
    login: str | None = typer.Option(None, "--login", help="GitHub login. Defaults to gh-auth user."),
) -> None:
    """Cross-correlate Claude Code cost with merged GitHub PRs by ISO week."""
    settings = load_settings()
    claude_dir = Path.home() / ".claude" / "projects"
    console.print(f"[dim]scanning Claude Code at {claude_dir}...[/dim]")
    claude_scan = scan_claude_code_projects(claude_dir)

    try:
        resolved_login = login or current_login()
    except GitHubConnectorError as exc:
        console.print(f"[red]GitHub: {exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[dim]fetching PRs authored by {resolved_login} (last {days} days)...[/dim]")
    try:
        prs = fetch_authored_prs(resolved_login, since=default_since(days))
    except GitHubConnectorError as exc:
        console.print(f"[red]GitHub: {exc}[/red]")
        raise typer.Exit(code=1) from None

    cost_per_session: dict[str, float] = {}
    for session in claude_scan.sessions:
        cost_per_session[session.session_id] = sum(
            estimate_cost(usage, model) for model, usage in session.usage_by_model.items()
        )

    report = correlate(claude_scan, prs, cost_per_session=cost_per_session)
    _print_roi_report(report, login=resolved_login, days=days)
    del settings


@app.command(name="sync")
def sync_cmd(
    days: int = typer.Option(90, "--days", help="Lookback window in days for PRs."),
    login: str | None = typer.Option(None, "--login", help="GitHub login. Defaults to gh-auth user."),
) -> None:
    """Fetch the user's recent GitHub PRs and print a summary (writes coming in v0.2)."""
    try:
        resolved_login = login or current_login()
        prs = fetch_authored_prs(resolved_login, since=default_since(days))
    except GitHubConnectorError as exc:
        console.print(f"[red]GitHub: {exc}[/red]")
        raise typer.Exit(code=1) from None

    merged = [p for p in prs if p.shipped]
    open_prs = [p for p in prs if p.state == "OPEN"]
    table = Table(title=f"GitHub PRs by {resolved_login} (last {days} days)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total PRs found", f"{len(prs):,}")
    table.add_row("Merged", f"{len(merged):,}")
    table.add_row("Open", f"{len(open_prs):,}")
    table.add_row("Total lines changed (merged)", f"{sum(p.total_lines_changed for p in merged):,}")
    if merged:
        avg_loc = sum(p.total_lines_changed for p in merged) / len(merged)
        table.add_row("Avg lines / merged PR", f"{avg_loc:,.0f}")
    console.print(table)

    if merged:
        recent = sorted(merged, key=lambda p: p.merged_at or p.created_at, reverse=True)[:10]
        recent_table = Table(title="Most recent merged PRs")
        recent_table.add_column("Repo", style="cyan", overflow="fold")
        recent_table.add_column("#")
        recent_table.add_column("Title", overflow="fold")
        recent_table.add_column("Merged at")
        recent_table.add_column("LoC", justify="right")
        for pr in recent:
            recent_table.add_row(
                pr.repo,
                str(pr.number),
                pr.title[:80],
                (pr.merged_at.date().isoformat() if pr.merged_at else ""),
                f"{pr.total_lines_changed:,}",
            )
        console.print(recent_table)


def _print_overview(agg) -> None:  # type: ignore[no-untyped-def]
    table = Table(title="Cursor scan overview", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")
    table.add_row("Sessions parsed", f"{agg.total_sessions:,}")
    table.add_row("  Parent sessions", f"{agg.parent_sessions:,}")
    table.add_row("  Subagent sessions", f"{agg.subagent_sessions:,}")
    table.add_row("Total turns", f"{agg.total_turns:,}")
    table.add_row("Avg turns / session", f"{agg.avg_turns_per_session:.1f}")
    table.add_row("Median turns / session", f"{agg.median_turns_per_session}")
    table.add_row("Est. user-side tokens", f"{agg.total_user_tokens:,}")
    table.add_row("Est. assistant-side tokens", f"{agg.total_assistant_tokens:,}")
    console.print(table)


def _print_buckets(agg) -> None:  # type: ignore[no-untyped-def]
    table = Table(title="User-side token attribution by bucket (estimated, cl100k_base)", show_lines=False)
    table.add_column("Bucket", style="cyan")
    table.add_column("Total tokens", justify="right")
    table.add_column("% of user input", justify="right")
    table.add_column("Sessions with > 0", justify="right")
    for bucket in agg.buckets:
        share = f"{bucket.share_of_user_tokens * 100:.1f}%"
        table.add_row(
            bucket.name,
            f"{bucket.total_tokens:,}",
            share,
            f"{bucket.appears_in_sessions:,}",
        )
    console.print(table)


def _print_workspaces(agg) -> None:  # type: ignore[no-untyped-def]
    table = Table(title="Top workspaces by total tokens (estimated)", show_lines=False)
    table.add_column("Workspace", style="cyan", overflow="fold")
    table.add_column("Sessions", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("User tokens", justify="right")
    table.add_column("Assistant tokens", justify="right")
    for ws in agg.workspaces[:8]:
        table.add_row(
            ws.workspace,
            f"{ws.session_count:,}",
            f"{ws.total_turns:,}",
            f"{ws.total_user_tokens:,}",
            f"{ws.total_assistant_tokens:,}",
        )
    console.print(table)


def _print_longest_sessions(agg) -> None:  # type: ignore[no-untyped-def]
    table = Table(title="Top 10 sessions by estimated total tokens", show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Workspace", style="cyan", overflow="fold")
    table.add_column("Turns", justify="right")
    table.add_column("User tokens", justify="right")
    table.add_column("Assistant tokens", justify="right")
    table.add_column("First query (truncated)", overflow="fold")
    for i, s in enumerate(agg.longest_sessions, 1):
        first = (s.first_user_query or "")[:80].replace("\n", " ")
        if s.first_user_query and len(s.first_user_query) > 80:
            first += "..."
        table.add_row(
            str(i),
            s.workspace,
            f"{s.turn_count:,}",
            f"{s.user_tokens_est:,}",
            f"{s.assistant_tokens_est:,}",
            first,
        )
    console.print(table)


def _print_deep_cursor(deep_report) -> None:  # type: ignore[no-untyped-def]
    if deep_report.repeated_prompt_clusters:
        table = Table(
            title=f"Repeated opening prompts ({len(deep_report.repeated_prompt_clusters)} clusters of ≥3 sessions)",
            show_lines=False,
        )
        table.add_column("# sessions", justify="right")
        table.add_column("User tokens burned", justify="right")
        table.add_column("Canonical prompt (truncated)", overflow="fold")
        for cluster in deep_report.repeated_prompt_clusters[:10]:
            table.add_row(
                str(len(cluster.session_ids)),
                f"{cluster.total_user_tokens_burned:,}",
                cluster.canonical_first_query.replace("\n", " "),
            )
        console.print(table)

    if deep_report.bloat_buckets:
        table = Table(title="Cross-session bucket bloat (per-session medians)", show_lines=False)
        table.add_column("Bucket", style="cyan")
        table.add_column("Median tokens / session", justify="right")
        table.add_column("p95 tokens / session", justify="right")
        table.add_column("Sessions affected", justify="right")
        table.add_column("Total tokens", justify="right")
        for bucket in deep_report.bloat_buckets[:10]:
            table.add_row(
                bucket.name,
                f"{bucket.median_tokens_per_session:,}",
                f"{bucket.p95_tokens_per_session:,}",
                f"{bucket.appears_in_sessions:,}",
                f"{bucket.total_tokens:,}",
            )
        console.print(table)

    if deep_report.mid_session_samples:
        table = Table(
            title=f"Heavy mid-session user turns (deduped, top {len(deep_report.mid_session_samples)})",
            show_lines=False,
        )
        table.add_column("Workspace", style="cyan", overflow="fold")
        table.add_column("Turn", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Reused in", justify="right")
        table.add_column("Preview", overflow="fold")
        for sample in deep_report.mid_session_samples:
            reused = f"{sample.duplicate_count} sessions" if sample.duplicate_count > 1 else "-"
            table.add_row(
                sample.workspace,
                str(sample.turn_index),
                f"{sample.estimated_tokens:,}",
                reused,
                sample.text_preview,
            )
        console.print(table)


def _print_claude_code_overview(scan, report) -> None:  # type: ignore[no-untyped-def]
    table = Table(title="Claude Code scan overview", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Sessions parsed", f"{scan.total_files_seen:,}")
    table.add_row("With usage data", f"{sum(1 for s in scan.sessions if s.total_usage.input_tokens > 0):,}")
    table.add_row("Total input tokens", f"{report.total_input_tokens:,}")
    table.add_row("Total output tokens", f"{report.total_output_tokens:,}")
    table.add_row("Total cache-read tokens", f"{report.total_cache_read_tokens:,}")
    table.add_row("Total cache-write tokens", f"{report.total_cache_write_tokens:,}")
    if report.overall_cache_hit_rate is not None:
        table.add_row("Overall cache hit rate", f"{report.overall_cache_hit_rate * 100:.1f}%")
    table.add_row("Estimated cost (USD)", f"${report.estimated_total_cost_usd:,.2f}")
    console.print(table)


def _print_claude_code_models(report) -> None:  # type: ignore[no-untyped-def]
    if not report.by_model:
        return
    table = Table(title="Claude Code cost by model", show_lines=False)
    table.add_column("Model", style="cyan", overflow="fold")
    table.add_column("Sessions", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Cache read", justify="right")
    table.add_column("Cache hit %", justify="right")
    table.add_column("USD", justify="right")
    for model in report.by_model:
        hit = f"{model.cache_hit_rate * 100:.1f}%" if model.cache_hit_rate is not None else "-"
        usd = f"${model.estimated_cost_usd:,.2f}" if model.is_priced else "[dim]unpriced[/dim]"
        table.add_row(
            model.model,
            f"{model.sessions_using:,}",
            f"{model.input_tokens:,}",
            f"{model.output_tokens:,}",
            f"{model.cache_read_tokens:,}",
            hit,
            usd,
        )
    console.print(table)


def _print_claude_code_projects(report) -> None:  # type: ignore[no-untyped-def]
    if not report.by_project:
        return
    table = Table(title="Claude Code cost by project (cwd)", show_lines=False)
    table.add_column("CWD", style="cyan", overflow="fold")
    table.add_column("Sessions", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("USD", justify="right")
    for project in report.by_project[:10]:
        table.add_row(
            project.cwd,
            f"{project.sessions:,}",
            f"{project.total_tokens:,}",
            f"${project.estimated_cost_usd:,.2f}",
        )
    console.print(table)


def _print_roi_report(report, *, login: str, days: int) -> None:  # type: ignore[no-untyped-def]
    if not report.weeks:
        console.print(f"[yellow]no data across the last {days} days for {login}.[/yellow]")
        return
    table = Table(
        title=f"AI cost vs shipping — {login}, weeks {report.earliest_week} to {report.latest_week}",
        show_lines=False,
    )
    table.add_column("ISO week", style="cyan")
    table.add_column("Sessions", justify="right")
    table.add_column("Input tok", justify="right")
    table.add_column("Output tok", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("PRs opened", justify="right")
    table.add_column("PRs merged", justify="right")
    table.add_column("LoC merged", justify="right")
    table.add_column("$/merged PR", justify="right")
    for week in report.weeks:
        cost_per = f"${week.cost_per_merged_pr:.2f}" if week.cost_per_merged_pr is not None else "-"
        table.add_row(
            week.iso_week,
            f"{week.claude_code_sessions:,}",
            f"{week.claude_code_input_tokens:,}",
            f"{week.claude_code_output_tokens:,}",
            f"${week.claude_code_estimated_cost_usd:,.2f}",
            f"{week.prs_opened:,}",
            f"{week.prs_merged:,}",
            f"{week.lines_changed_merged:,}",
            cost_per,
        )
    console.print(table)
    console.print()
    console.print(
        f"[dim]totals: ${report.total_estimated_cost_usd:,.2f} estimated Claude Code spend, "
        f"{report.total_merged_prs:,} merged PRs across {len(report.weeks)} active weeks.[/dim]"
    )
    console.print(
        "[dim]note: cost-per-PR is a starting point - calibrate against felt productivity, "
        "not a leaderboard. METR (2025) showed AI can make seniors feel faster while slowing them down.[/dim]"
    )


if __name__ == "__main__":
    app()
