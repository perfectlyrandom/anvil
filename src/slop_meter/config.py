"""Configuration loaded from ~/.slop_meter/config.toml + env + CLI overrides."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_home() -> Path:
    return Path.home() / ".slop_meter"


class Settings(BaseSettings):
    """Runtime configuration for slop_meter."""

    model_config = SettingsConfigDict(
        env_prefix="SLOP_METER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    home_dir: Path = Field(default_factory=_default_home, description="Where slop_meter stores its data.")
    db_filename: str = Field(default="data.db", description="SQLite database filename inside home_dir.")
    web_host: str = Field(default="127.0.0.1", description="Dashboard bind host.")
    web_port: int = Field(default=7331, description="Dashboard bind port.")
    telemetry_enabled: bool = Field(default=False, description="Opt-in anonymous telemetry. Off by default.")
    github_login: str | None = Field(default=None, description="GitHub username for PR/contribution lookups.")
    cursor_projects_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cursor" / "projects",
        description="Where Cursor stores per-project agent transcripts.",
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key for analysis features.")
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model used by the analyzer.",
    )

    @property
    def db_path(self) -> Path:
        return self.home_dir / self.db_filename


def load_settings() -> Settings:
    """Return a Settings instance, ensuring home_dir exists."""
    settings = Settings()
    settings.home_dir.mkdir(parents=True, exist_ok=True)
    return settings
