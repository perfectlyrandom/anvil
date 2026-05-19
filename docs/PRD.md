# anvil - Product Requirements (v0)

## 1. Problem

Engineers running multiple AI coding tools in parallel cannot answer two basic questions.

1. **How much am I actually spending across all of them?** Cursor's billing lives in a SQLite tracking DB. Claude Code records per-response token usage in JSONL. Codex writes rollout files with its own schema. There is no single number for "today's AI spend".
2. **Is that spend buying me anything?** Vendor dashboards report token consumption. They do not report shipping outcomes. Correlation between AI sessions and merged code is anecdotal ("Cursor saved me time this week") and unfalsifiable.

A related pain point. Skill libraries (Cursor `~/.cursor/skills`, Codex `~/.codex/skills`, Claude Code) inject every skill description into every prompt as overhead. There is no surface that tells you which skills are paying for themselves. The cost of one stale skill in the catalog is small. The cost of 25 stale skills across 1000 sessions is not.

## 2. Goals

| # | Goal | Measurable target |
|---|------|---|
| G1 | Single dollar-cost figure across all three tools, refreshed within seconds | <2s warm refresh on a laptop with ~6 months of history |
| G2 | Per-skill consultation evidence, with false positive rate <10% on hand-validated audit | Quarterly hand-validation of 20 random skill verdicts |
| G3 | Concrete correlation between AI sessions and merged PRs over a configurable window | Per-PR cost estimate displayed for >70% of PRs in the window |
| G4 | Zero transcript or token-level data leaves the user's machine | Verified by `lsof` audit during normal use |
| G5 | Coach surfaces at least one actionable, evidence-backed issue per scan in 90% of runs | Logged via internal counter |

## 3. Non-goals

- **Team / multi-user rollups.** anvil is single-user, local-only. A team surface is a different product (see Future).
- **Real-time cost streaming.** Refreshes are pull-based on tab switch. No daemons, no notifications.
- **Anthropic / OpenAI / Cursor vendor billing reconciliation.** We surface the data they produced. Disputes go to the vendor.
- **A general LLM-cost analyzer for production traffic.** anvil is scoped to interactive AI-assisted coding only.
- **Editor integrations.** This is a separate dashboard, not a Cursor extension.

## 4. Users

Persona descriptions are sharp on purpose. Edge users (e.g. "uses only ChatGPT web") are explicitly not in scope.

- **Heavy AI-coder.** Uses Cursor and at least one of {Claude Code, Codex} daily. Has 10+ skills installed. Pays out of pocket or owns a per-seat budget they have to justify.
- **Tooling-curious engineer.** Wants to know which of their AI tools is pulling weight before recommending one to their team.

Not in scope. Casual AI users, non-engineers, enterprise admins managing fleets.

## 5. Success metrics (90 days after launch)

| Metric | Target | How measured |
|---|---|---|
| Weekly active users (self-installed) | 10 | Anonymous opt-in ping, default off |
| Median time from install to first "useful insight" | <5 min | User study, n=5 |
| # of stale skills deleted within 7 days of being flagged | 60% of flagged | Local audit log (opt-in) |
| Variance in $/PR over a stable 30-day window | ≤ 30% (after baseline) | Computed from anvil's own report |
| User-reported "I would notice if this went away" | 7/10 | Survey, n=10 |

## 6. Solution overview

A local FastAPI app served on `localhost:7331` with five htmx-driven tabs and one chat panel. Reads files in place from the user's home directory.

| Tab | Data sources | Output |
|---|---|---|
| Overview | Cursor transcripts | Per-session token weight, top buckets, heavy turns |
| Cost | Cursor `aiUsage.db`, Claude Code JSONL, Codex rollouts | Per-tool real-cost totals, model breakdown, cache hit rate |
| Coach | All of the above + GitHub | Ranked issues with evidence links (stale skills, oversized prompts, low cache hit, no shipping) |
| Deep | Cursor transcripts | Bucket-level prompt anatomy across all sessions |
| Shipped | `gh` CLI | Merged PRs over N days, repo distribution, meaningful-PR ranking |
| Ask | All of the above | Anthropic agent with tool access to the scanned data |

### Honest-signal commitments (P0)

- **Stale skill detection** parses JSONL events structurally, not by raw text. For Codex, only `response_item.message` events with `role ∈ {user, assistant}` count as consultation evidence. For Claude Code, only `type ∈ {user, assistant}`. Catalog injections (`role = developer`) and tool output dumps (`function_call_output`, `exec_command_end`) are dropped before they reach the signal. Within retained events, any single event naming ≥4 distinct skills is treated as a catalog echo and discarded.
- **Cost** is sourced from the vendor's own ledger where one exists (Cursor's tracking DB) and estimated via published per-model pricing where it doesn't (Codex). Each per-tool cost figure carries a "real" vs "estimated" badge.
- **Stale verdicts disclose their blind spots.** Cursor transcripts do not expose structured tool calls, so silent skill following is undetectable. The UI says so.

## 7. Requirements

### P0 (ship-blocking)

- R1. Parse and merge cost data from Cursor, Claude Code, Codex.
- R2. Compute stale-skill verdicts using structured event parsing.
- R3. Correlate against `gh`-fetched PRs over a configurable window (default 90 days).
- R4. Run entirely on `localhost`. No outbound calls except `gh` and (optionally) Anthropic for the Ask panel.
- R5. Refresh on tab switch in under 2 seconds for a representative dataset (1000 sessions, 30 skills).
- R6. Surface a methodology footnote on every signal that involves inference.

### P1 (next 30 days)

- R7. Configurable Coach severity. Today the thresholds are hardcoded.
- R8. Per-repo cost attribution (today the dashboard is global across all repos).
- R9. Cost-per-PR breakdown in the Shipped tab.
- R10. Export report as JSON / CSV.

### P2 (next 90 days)

- R11. Plugin interface for new AI tools (Aider, Continue, JetBrains AI).
- R12. Cursor silent-following detection (depends on Cursor exposing structured tool calls or an aiUsage-style log).
- R13. Compacted-session uncertainty flag (treat sessions whose Codex history was summarized as "unknown" rather than "consulted=0").

## 8. Open questions and risks

| # | Question / risk | Mitigation |
|---|---|---|
| Q1 | Are the Cursor silent-following sessions a meaningful share of consultations? | Hand-audit 50 Cursor sessions where Cursor's aiUsage.db shows tool-use cost; compare against current "stale" verdict |
| Q2 | Codex `compacted` events lose pre-summary context. How aggressive should the "uncertain" flag be? | Tag flagged-stale skills as "uncertain" only if ≥30% of the relevant sessions are compacted |
| Q3 | Pricing schemas drift (vendor changes per-token rate). How does anvil stay accurate? | Pricing config in a single file, easily PR-able. Stamp every cost figure with the pricing-config date |
| R1 | If a vendor changes their JSONL schema, the parser breaks silently. | Schema-version pinning per parser; surface a "parser stale" banner on schema mismatch |
| R2 | The Ask panel's Anthropic dependency adds an outbound call. | Off by default. Requires explicit `ANVIL_ANTHROPIC_API_KEY` |
| R3 | Multi-user expansion would require centralizing transcript metadata. | Out of scope. If revisited, requires a fresh privacy review |

## 9. Future

Ordered, not committed.

1. Cursor silent-following resolution (depends on Cursor).
2. Diff-quality × cost correlation. Pair specific PRs to the AI sessions that produced them via timestamp / repo / branch matching. Flag the high-cost low-impact ones.
3. Per-repo cost attribution.
4. Compacted-session uncertainty handling.
5. Team rollup product (separate scope; not a feature of anvil).
6. Plugin model for new AI tools.

## 10. Decisions needed

1. **Pricing config ownership.** Hardcoded in `src/anvil/analysis/pricing.py` today. Should it move to a config file shipped with the repo? Decision: yes, P1. Tracking as part of R10's export work.
2. **Telemetry.** Should anvil collect anonymous "tool was useful" metrics to validate G1/G2? Decision: no, ship without; revisit if user-count crosses 50.
3. **Name.** "anvil" is the current name. Keep or change? Open.
