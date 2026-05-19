"""AI agent that helps you find things in your AI-tool usage data.

Uses Anthropic tool-use to let Claude query the same parsers/analyzers the CLI uses.
The agent is read-only and operates on your local data - nothing leaves your machine
except the prompts sent to Anthropic.
"""
