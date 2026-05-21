"""Configuration loaded from ~/.anvil/config.toml + env + CLI overrides."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_home() -> Path:
    return Path.home() / ".anvil"


class Settings(BaseSettings):
    """Runtime configuration for anvil."""

    model_config = SettingsConfigDict(
        env_prefix="ANVIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    home_dir: Path = Field(
        default_factory=_default_home,
        description="Where anvil caches scan state. Created on first run.",
    )
    web_host: str = Field(default="127.0.0.1", description="Dashboard bind host.")
    web_port: int = Field(default=7331, description="Dashboard bind port.")
    github_login: str | None = Field(default=None, description="GitHub username for PR/contribution lookups.")
    cursor_projects_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cursor" / "projects",
        description="Where Cursor stores per-project agent transcripts.",
    )
    cursor_skills_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cursor" / "skills",
        description="Where user-level Cursor skills live (SKILL.md files).",
    )
    codex_sessions_dir: Path = Field(
        default_factory=lambda: Path.home() / ".codex" / "sessions",
        description="Where Codex CLI writes session rollouts (token usage lives here).",
    )
    cursor_tracking_db: Path = Field(
        default_factory=lambda: Path.home() / ".cursor" / "ai-tracking" / "ai-code-tracking.db",
        description="Cursor's local AI-tracking SQLite. Recovers model attribution for Cursor sessions.",
    )
    cursor_state_db: Path = Field(
        default_factory=lambda: Path.home()
        / "Library"
        / "Application Support"
        / "Cursor"
        / "User"
        / "globalStorage"
        / "state.vscdb",
        description=(
            "Cursor's main global state SQLite. Holds per-bubble tokenCount, which gives real "
            "measured input/output tokens for sessions that used Cursor's billed models (BYOK is zero)."
        ),
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key for analysis features.")
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model used by the analyzer.",
    )
    default_pricing_model: str = Field(
        default="claude-opus-4-5",
        description=(
            "Model family used to price Cursor-derived savings (Cursor's JSONL doesn't "
            "record the model, so we can't infer it). Override with ANVIL_DEFAULT_PRICING_MODEL."
        ),
    )


def load_settings() -> Settings:
    """Return a Settings instance, ensuring home_dir exists."""
    settings = Settings()
    settings.home_dir.mkdir(parents=True, exist_ok=True)
    return settings
