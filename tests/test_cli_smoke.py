"""Smoke tests for the Typer CLI - asserts every stub command shows up in --help."""

from __future__ import annotations

from typer.testing import CliRunner

from slop_meter.cli import app

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "slop_meter" in result.output


def test_all_commands_present_in_help() -> None:
    """Every command declared in commit 1 must show up in the root help text."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "init",
        "watch",
        "web",
        "top",
        "sync",
        "query",
        "config",
        "telemetry",
        "doctor",
        "export",
        "analyze",
    ):
        assert command in result.output, f"missing command in --help output: {command}"


def test_config_show_prints_table(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLOP_METER_HOME_DIR", str(tmp_path / ".slop_meter"))
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "home_dir" in result.output


def test_init_creates_database(tmp_path, monkeypatch) -> None:
    home = tmp_path / ".slop_meter"
    monkeypatch.setenv("SLOP_METER_HOME_DIR", str(home))
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (home / "data.db").exists()
