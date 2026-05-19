"""Typer CLI entry point.

All commands are stubs in commit 1 - they exit cleanly with a "not implemented" message
so the help surface, packaging, and pipx install all work end-to-end before any logic lands.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from slop_meter import __version__
from slop_meter.analysis.cursor_stats import aggregate
from slop_meter.analysis.llm_analyzer import run_analysis, select_samples
from slop_meter.config import load_settings
from slop_meter.db import create_engine_for_path, init_schema
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
def sync(
    update_pricing: bool = typer.Option(False, "--update-pricing", help="Refresh model pricing JSON."),
) -> None:
    """One-shot pull from all configured data sources."""
    del update_pricing
    _not_implemented("sync")


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
    source: str = typer.Option("cursor", "--source", help="Data source. Only 'cursor' for now."),
    with_llm: bool = typer.Option(False, "--with-llm", help="Run Claude analysis with web search on sampled prompts."),
    sample_size: int = typer.Option(30, "--samples", help="How many top sessions to send to the LLM analyzer."),
    no_web_search: bool = typer.Option(False, "--no-web-search", help="Disable web search in the LLM analyzer."),
) -> None:
    """Scan local AI tool data, compute bucket stats, optionally get LLM critique of prompts."""
    if source != "cursor":
        console.print(f"[red]source '{source}' not supported yet. Use 'cursor'.[/red]")
        raise typer.Exit(code=1)

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

    if not with_llm:
        console.print()
        console.print(
            "[dim]tip:[/dim] pass [cyan]--with-llm[/cyan] to get a Claude-powered prompt critique with web search."
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

    console.print()
    console.print(
        f"[dim]running {settings.anthropic_model} on {len(samples)} prompt samples "
        f"({'web search ON' if not no_web_search else 'web search OFF'})...[/dim]"
    )
    report = run_analysis(
        samples,
        api_key=api_key,
        model=settings.anthropic_model,
        enable_web_search=not no_web_search,
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


if __name__ == "__main__":
    app()
