# anvil

> How much is your AI spend actually buying you?

A local-first personal tool. Reads your Cursor and Claude Code history, prices it honestly, and lines it up next to the PRs you actually shipped.

```bash
pipx install anvil
anvil init
anvil sync
anvil web
```

[![CI](https://github.com/perfectlyrandom/anvil/actions/workflows/ci.yaml/badge.svg)](https://github.com/perfectlyrandom/anvil/actions/workflows/ci.yaml)

## Status

Pre-alpha. Scaffolding only. See the [roadmap](#roadmap) below.

## What this is

A calibration tool. It reads your local AI-tool transcripts (Claude Code JSONL, Cursor's `ai-code-tracking.db`) plus your GitHub PR history, and shows you whether your AI spend is correlated with your shipping velocity.

It is informed by [the METR study](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) - experienced developers thought AI made them 24% faster while measured outcomes showed 19% slower. anvil is built around honest calibration, not validation.

## What this isn't

- Not a manager dashboard. No team rollups, no leaderboards, no cross-user comparison.
- Not a productivity score. No single-number ROI or grade.
- Not a surveillance tool. No keystroke or focus tracking. Opt-in telemetry only.
- Not a verdict generator. The tool shows you data and asks questions; you decide what it means.

## What you get

- **Cost dashboard** - daily/weekly/monthly AI spend by tool, model, and project.
- **ROI correlation view** - AI spend trend next to your shipping trend. With caveats baked into the UI.
- **Calibration view** - perception-vs-shipping gap over time (the METR view).
- **Cost-oriented tips** - rule-based suggestions to reduce token bloat without sacrificing output.

Coming in v0.2 (~4-6 weeks after launch):

- **Per-bucket prompt ROI** - which of your `.cursor/rules`, `~/.cursor/skills`, MCP servers, and `AGENTS.md` files were active in the sessions that produced your merged PRs. Drop the dead weight.
- **Cache health dashboard** - hit rate, cache-killing pattern detection, fix recommendations.
- **Prompt-aware tips** - 10 more optimization tips beyond MVP cost tips.

Phase 2 (the original "hook into prompts before they hit the model" ask):

- **Optional proxy mode** - mitmproxy + local CA cert for users who want exact prompt-byte capture.
- **SDK wrapper companion** - `anvil-sdk` for notebook / LangChain users writing their own agent code.

## Install (placeholder until v0.1 ships to PyPI)

```bash
# from source for now
git clone https://github.com/perfectlyrandom/anvil.git
cd anvil
pipx install -e .
```

## CLI surface

```text
anvil init                  # one-time setup, prompts for GitHub PAT (stored via keyring)
anvil watch                 # foreground daemon, watches Claude Code / Cursor for new turns
anvil watch --daemon        # background, PID file in ~/.anvil/
anvil web                   # FastAPI dashboard on 127.0.0.1:7331, opens browser
anvil top                   # Rich live TUI ("htop mode")
anvil sync                  # one-shot pull from all sources
anvil query "SELECT ..."    # raw SQL escape hatch (Datasette-style)
anvil config show
anvil telemetry on|off|status
anvil doctor                # checks permissions, keychain, schema version
anvil export                # dump ~/.anvil/data.db to a shareable file
```

## Development

```bash
poetry install --with dev
poetry run pre-commit install
poetry run pytest
poetry run ruff format src tests
poetry run ruff check src tests
poetry run mypy src
```

## Roadmap

- [x] Commit 1: scaffold (this commit)
- [ ] Commit 2: Claude Code JSONL parser - the richest data source
- [ ] Commit 3: GitHub GraphQL connector
- [ ] Commit 4: Cursor `ai-code-tracking.db` parser
- [ ] Commits 5-6: FastAPI JSON endpoints + Next.js dashboard
- [ ] Commit 7: ROI correlation view (MVP killer view)
- [ ] Commit 8: Calibration view (METR view)
- [ ] Commit 9: Tips engine
- [ ] Commit 10: v0.1 launch (PyPI, screencast, HN)

## License

MIT
