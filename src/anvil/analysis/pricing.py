"""Anthropic model pricing for cost estimation.

Prices are in USD per million tokens (MTok). Source: Anthropic pricing pages as of
the dates noted below. **These are estimates** - they go stale as Anthropic updates
prices. The ``anvil analyze --with-llm`` web-search analyzer can be asked to
verify current prices and propose updates.

The pricing table is keyed by **model family prefix** so we match newer dated variants
(e.g. ``claude-sonnet-4-5-20250929``) against the family entry ``claude-sonnet-4-5``.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvil.parsers.claude_code import TurnUsage


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for a single model family."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float  # 5-minute ephemeral cache by default
    cache_read_per_mtok: float


# Last refreshed manually: 2026-05-18. Verify with web search if it's been > 60 days.
# OpenAI cache_write rate is the same as input rate (no separate cache-write tier).
PRICING_TABLE: dict[str, ModelPrice] = {
    # Claude 4.5 family
    "claude-sonnet-4-5": ModelPrice(3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00, 1.25, 0.10),
    "claude-opus-4-5": ModelPrice(15.00, 75.00, 18.75, 1.50),
    # Claude 4 family
    "claude-opus-4": ModelPrice(15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-4": ModelPrice(3.00, 15.00, 3.75, 0.30),
    # Claude 3.7 family
    "claude-3-7-sonnet": ModelPrice(3.00, 15.00, 3.75, 0.30),
    # Claude 3.5 family
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00, 3.75, 0.30),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00, 1.00, 0.08),
    # Claude 3 family
    "claude-3-opus": ModelPrice(15.00, 75.00, 18.75, 1.50),
    "claude-3-sonnet": ModelPrice(3.00, 15.00, 3.75, 0.30),
    "claude-3-haiku": ModelPrice(0.25, 1.25, 0.30, 0.03),
    # GPT-5 family (Codex CLI default)
    "gpt-5.5": ModelPrice(1.25, 10.00, 1.25, 0.125),
    "gpt-5-mini": ModelPrice(0.25, 2.00, 0.25, 0.025),
    "gpt-5-nano": ModelPrice(0.05, 0.40, 0.05, 0.005),
    "gpt-5": ModelPrice(1.25, 10.00, 1.25, 0.125),
    # GPT-4o family (older Codex / Responses API workloads)
    "gpt-4o-mini": ModelPrice(0.15, 0.60, 0.15, 0.075),
    "gpt-4o": ModelPrice(2.50, 10.00, 2.50, 1.25),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.40, 0.10),
    "gpt-4.1": ModelPrice(2.00, 8.00, 2.00, 0.50),
}

UNKNOWN_MODEL_FALLBACK = ModelPrice(3.00, 15.00, 3.75, 0.30)


def price_for_model(model: str | None) -> ModelPrice | None:
    """Look up pricing by longest matching prefix. Returns None for synthetic/unknown."""
    if not model or model.startswith("<"):
        return None
    # Strip date suffix and try longest prefixes first.
    best: tuple[str, ModelPrice] | None = None
    for family, price in PRICING_TABLE.items():
        if model.startswith(family) and (best is None or len(family) > len(best[0])):
            best = (family, price)
    return best[1] if best else None


def estimate_cost(usage: TurnUsage, model: str | None) -> float:
    """Estimate USD cost for one TurnUsage block. Returns 0.0 if model is unknown."""
    price = price_for_model(model)
    if price is None:
        return 0.0
    return (
        usage.input_tokens * price.input_per_mtok / 1_000_000
        + usage.output_tokens * price.output_per_mtok / 1_000_000
        + usage.cache_creation_input_tokens * price.cache_write_per_mtok / 1_000_000
        + usage.cache_read_input_tokens * price.cache_read_per_mtok / 1_000_000
    )


def input_price_per_token(model: str | None) -> float:
    """Return the per-token input price in USD for a model, with a Sonnet 4.5 fallback."""
    price = price_for_model(model) or UNKNOWN_MODEL_FALLBACK
    return price.input_per_mtok / 1_000_000


def humanize_model(model: str | None) -> str:
    """Render a model family name for UI display ('claude-opus-4-5' → 'Opus 4.5')."""
    if not model:
        return "Sonnet 4.5"
    # Strip a trailing date suffix if present, then split into family + version pieces.
    base = model.split("-2", 1)[0]
    parts = base.split("-")
    # Drop the leading 'claude' prefix if present.
    if parts and parts[0] == "claude":
        parts = parts[1:]
    if not parts:
        return model
    family = parts[0].capitalize()
    version = ".".join(parts[1:]) if len(parts) > 1 else ""
    return f"{family} {version}".strip()


def cache_hit_rate(usage: TurnUsage) -> float | None:
    """Cache-read-tokens / (cache-read + input). Returns None if no input recorded."""
    total = usage.cache_read_input_tokens + usage.input_tokens
    if total == 0:
        return None
    return usage.cache_read_input_tokens / total
