"""Anthropic tool-use loop for the chat agent.

Conversation flow:

1. User sends a message.
2. Claude either responds directly or calls one or more tools.
3. We execute the tools locally (read-only queries against the user's data).
4. We send the tool results back to Claude.
5. Repeat until Claude returns ``end_turn`` or ``stop_reason != "tool_use"``.

We cap the loop at ``max_tool_iterations`` to prevent runaway tool spirals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

from anvil.agent.tools import ToolContext, ToolSpec, build_tool_specs

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are anvil's data agent. The user installed anvil
to understand their AI coding tool usage (Cursor + Claude Code + GitHub) and you help
them dig into the data.

Behavioral rules:
- Use the tools to ground every quantitative claim. Don't make up numbers.
- Prefer surface-level summaries first, then drill deeper if asked.
- When you report dollar figures, remind the user they're estimates (pricing table
  is in source, can drift) - one mention per conversation is enough.
- Never recommend metrics that incentivize bad behavior (lines-of-code, PR count
  alone). Frame insights as calibration, not evaluation.
- If the user asks about repeated prompts, attached_files bloat, or "where is my
  money going", you can be specific and direct - real findings are valuable.
- Keep responses concise. Tables and bullets over walls of text.
"""


@dataclass
class AgentMessage:
    """One message in the chat history. Role is 'user' or 'assistant'."""

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentReply:
    """Result of one agent.send() invocation."""

    text: str
    tool_calls_summary: list[str]
    input_tokens: int
    output_tokens: int
    iterations: int


class Agent:
    """Stateful chat agent. Keeps the Anthropic conversation history in memory.

    Single-user, single-session (no concurrency story yet). The web layer creates
    one Agent per browser session.
    """

    def __init__(
        self,
        *,
        api_key: str,
        ctx: ToolContext,
        model: str = "claude-sonnet-4-5-20250929",
        max_tool_iterations: int = 8,
        tool_specs: list[ToolSpec] | None = None,
    ) -> None:
        # Lazy import so the rest of the package doesn't depend on anthropic.
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._ctx = ctx
        self._model = model
        self._max_iter = max_tool_iterations
        self._tools = tool_specs or build_tool_specs()
        self._tools_by_name = {t.name: t for t in self._tools}
        self._history: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def history_turn_count(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history = []

    def _tool_specs_for_api(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in self._tools]

    def _exec_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools_by_name.get(name)
        if spec is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return spec.impl(self._ctx, **args)
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return {"error": f"tool {name} raised {type(exc).__name__}: {exc}"}

    def send(self, user_message: str) -> AgentReply:
        """Send one user message; run tool loop until the model is done."""
        self._history.append({"role": "user", "content": user_message})
        tool_calls_summary: list[str] = []
        total_input = 0
        total_output = 0

        for iteration in range(self._max_iter):
            # cast(Any, ...) because the Anthropic SDK wants very specific TypedDicts
            # for tools/messages and our dynamic dict-builders don't satisfy them at type
            # level, even though they're correct at runtime.
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                tools=cast(Any, self._tool_specs_for_api()),
                messages=cast(Any, self._history),
            )
            total_input += getattr(response.usage, "input_tokens", 0) or 0
            total_output += getattr(response.usage, "output_tokens", 0) or 0

            # Persist the assistant's full content block list (text + tool_use blocks)
            # since later tool_result blocks must reference the tool_use ids by position.
            assistant_blocks: list[dict[str, Any]] = []
            tool_uses: list[tuple[str, str, dict[str, Any]]] = []  # (id, name, input)
            text_pieces: list[str] = []
            for block in response.content:
                if block.type == "text":
                    assistant_blocks.append({"type": "text", "text": block.text})
                    text_pieces.append(block.text)
                elif block.type == "tool_use":
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    raw_input = block.input if isinstance(block.input, dict) else {}
                    tool_uses.append((block.id, block.name, raw_input))
            self._history.append({"role": "assistant", "content": assistant_blocks})

            if response.stop_reason != "tool_use":
                return AgentReply(
                    text="\n".join(text_pieces).strip(),
                    tool_calls_summary=tool_calls_summary,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    iterations=iteration + 1,
                )

            # Execute every tool call and ship results back in a single user message.
            tool_result_blocks: list[dict[str, Any]] = []
            for tool_id, tool_name, tool_input in tool_uses:
                result = self._exec_tool(tool_name, tool_input)
                summary = f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in tool_input.items())})"
                tool_calls_summary.append(summary)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(result, default=str),
                    }
                )
            self._history.append({"role": "user", "content": tool_result_blocks})

        # Bail out after max iterations.
        return AgentReply(
            text=(
                "I hit the tool iteration cap before finishing - the question might need to be "
                "split into smaller asks. Try a more specific follow-up."
            ),
            tool_calls_summary=tool_calls_summary,
            input_tokens=total_input,
            output_tokens=total_output,
            iterations=self._max_iter,
        )
