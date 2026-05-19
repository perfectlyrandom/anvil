"""Typer CLI entry point.

The dashboard is the product; the CLI exists to launch it. Earlier commits scaffolded a
dozen stub commands (``init``, ``watch``, ``top``, ``query``, ``analyze``, ``cost``, ``roi``,
etc.) that never grew real implementations because everything ended up living in the web UI.
Those were deleted to cut bloat; the only commands that survive are the ones with real
behavior the user actually runs.
"""

from __future__ import annotations

import typer
from rich.console import Console

from anvil import __version__
from anvil.config import load_settings

app = typer.Typer(
    name="anvil",
    help="How much is your AI spend actually buying you?",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"anvil {__version__}")
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
    """anvil root."""
    del version


@app.command()
def web(
    host: str | None = typer.Option(None, "--host", help="Override bind host."),
    port: int | None = typer.Option(None, "--port", help="Override bind port."),
) -> None:
    """Launch the local FastAPI dashboard with the chat agent."""
    settings = load_settings()
    h = host or settings.web_host
    p = port or settings.web_port
    if not settings.anthropic_api_key:
        console.print(
            "[yellow]warning:[/yellow] no ANVIL_ANTHROPIC_API_KEY in .env " "- the chat agent will be disabled."
        )
    console.print(f"[bold]anvil[/bold] dashboard at [cyan]http://{h}:{p}[/cyan]")
    console.print("[dim]ctrl-c to stop[/dim]")
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red]")
        raise typer.Exit(code=1) from None
    uvicorn.run("anvil.web.app:create_app", host=h, port=p, reload=False, factory=True)
